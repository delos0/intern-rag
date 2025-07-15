from typing import List, Tuple
from langchain.schema import BaseRetriever, Document
import logging
from transformers import AutoTokenizer
from langchain.callbacks.base import BaseCallbackHandler
from constants import *

tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)


class LoggingCallback(BaseCallbackHandler):
    def on_retriever_end(self, documents, **kwargs):
        # documents: List[Document]
        logging.info(f"[Retrieval] got {len(documents)} docs")
        for i, doc in enumerate(documents):
            chars = len(doc.page_content)
            toks = len(tokenizer.encode(doc.page_content))
            logging.info(f"  • doc#{i}: {chars} chars / {toks} toks")

    def on_llm_start(self, serialized, prompts, **kwargs):
        # prompts: List[str] or List[BaseMessage]
        for i, p in enumerate(prompts):
            text = p if isinstance(p, str) else p.content
            toks = len(tokenizer.encode(text))
            logging.info(f"[Prompt #{i}] {toks} tokens")
            logging.info(f"  ========Prompt text=======: \n{text}")

    def on_llm_new_token(self, token: str, **kwargs):
        logging.info(f"[Stream token] {token!r}")

    def on_llm_end(self, response, **kwargs):
        # response.generations: List[List[Generation]]
        gen = response.generations[0][0].text
        toks = len(tokenizer.encode(gen))
        logging.info(f"[Generation done] {toks} tokens")

class Retriever(BaseRetriever):
    base_retriever: BaseRetriever
    reranker: CrossEncoder
    prefetch_k: int = PREFETCH_K
    final_k: int = FINAL_K

    """
    base_retriever: e.g. vectordb.as_retriever()
    reranker: a Sentence-Transformers CrossEncoder model
    prefetch_k: how many documents to pull initially
    final_k: how many to return after reranking
    """



    def get_relevant_documents(self, query: str) -> List[Document]:
        # 1) pull more candidates than we'll finally return
        logging.info(f"[Retriever] fetching up to {self.prefetch_k} candidates for query: {query!r}")
        candidates = self.base_retriever.get_relevant_documents(
            query,
            # if your base retriever doesn’t accept k, embed+similarity manually:
            k=self.prefetch_k
        )
        logging.info(f"[Retriever] actually received {len(candidates)} candidates")

        for i, doc in enumerate(candidates):
            char_len = len(doc.page_content)
            tok_len = len(tokenizer.encode(doc.page_content))
            logging.info(f"  • candidate #{i}: {char_len} chars / {tok_len} toks")
            logging.info(doc.page_content)

        # 2) rerank with the cross‐encoder
        pairs: List[Tuple[str,str]] = [(query, doc.page_content) for doc in candidates]
        scores = self.reranker.predict(pairs)  # one score per candidate
        logging.info(f"[Retriever] reranker scores: {[round(s, 2) for s in scores]}")

        # 3) sort & take the top final_k
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, _ in ranked[: self.final_k]]

        logging.info(f"[Retriever] returning top {len(top_docs)} docs: "
                     f"{[round(s, 2) for _, s in ranked[:self.final_k]]}")

        for doc, s in ranked[: self.final_k]:
            logging.info(f"  • score: {s}")
            logging.info(doc.page_content)

        for doc in top_docs:
            logging.info(doc)

        return top_docs
