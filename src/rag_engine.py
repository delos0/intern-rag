import os
import logging

from langchain.llms import ollama
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import streamlit as st
from constants import *
from Retriever import *



def ingest_pdf(doc_path):
    """Load PDF documents."""
    if os.path.exists(doc_path):
        loader = UnstructuredPDFLoader(file_path=doc_path)
        data = loader.load()
        # documents = []
        # for doc in data:
        #     documents.append(doc.page_content)

        logging.info("PDF loaded successfully.")
        return data
    else:
        logging.error(f"PDF file not found at path: {doc_path}")
        st.error("PDF file not found.")
        return None


def split_documents(documents):
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = text_splitter.split_documents(documents)

    logging.info("Documents split into chunks.")
    for i, doc in enumerate(chunks):
        logging.info(f"[Splitter] chunk {i}: {len(doc.page_content)} chars")
    return chunks


@st.cache_resource
def load_vector_db():
    """Load or create the vector database."""
    #ollama.pull(EMBEDDING_MODEL)

    embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)
#    embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


    if os.path.exists(PERSIST_DIRECTORY):
        vector_db = Chroma(
            embedding_function=embedding,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=PERSIST_DIRECTORY,
        )
        logging.info("Loaded existing vector database.")
    else:
        data = ingest_pdf(DOC_PATH)
        if data is None:
            return None

        chunks = split_documents(data)

        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=PERSIST_DIRECTORY,
        )
        logging.info("Vector database created and persisted.")
    return vector_db

def create_rerank_retriever(vector_db):
    base_retriever = vector_db.as_retriever(search_kwargs={"k": PREFETCH_K})

    rerank_retriever = Retriever(
        base_retriever=base_retriever,
        reranker=CROSS_ENCODER,
        prefetch_k=PREFETCH_K,
        final_k=FINAL_K,
    )
    return rerank_retriever


def create_chain(retriever, llm):
    format_docs = RunnableLambda(
        lambda inputs: "\n\n---\n\n".join(
            f"Source: {d.metadata.get('source', '?')}\n{d.page_content}"
            for d in inputs["context"]
        )
    )

    assign_context = RunnablePassthrough.assign(context=format_docs)

    """Create the chain with preserved syntax."""
    template = """
Sen, üniversitemizde staj sürecinde öğrencilere rehberlik eden uzman bir akademik danışmansın.
Sana resmi Staj SSS belgesinden alınan bağlam verilecek;
bu belgede başvuru adımları, son tarihler, gereksinimler
ve daha fazlasıyla ilgili sıkça sorulan soruların cevapları yer alıyor.

--

## Görev Talimatları:

1. **Sadece sağlanan bağlamı kullanarak cevap ver.** Yanıtını tamamen önündeki metne dayandır.
2. **Yetersiz bilgi durumunu nazikçe ele al.** Sağlanan bağlam, güvenle yanıt vermeye yetecek bilgi içermiyorsa, yanıtla:
   > "Materyaller, iyi bir yanıt vermek için yeterli görünmüyor."

---

### Bağlam
{context}

### Kullanıcı Sorusu
{question}
"""


#     template = """
# Sen, üniversitemizde staj sürecinde öğrencilere rehberlik eden uzman bir akademik danışmansın.
#
# ### Örnek
#
# **Bağlam:**
# 1. Öncelikle gerekli alanları doldurarak staj talebini oluşturmalısınız.
# 2. Staj talebinizi oluşturduktan sonra “Staj Talepleri / Liste” sayfasına gidin ve uygun başlangıç/bitiş tarihlerini seçin.
# 3. “Kabul formülü oluştur” tuşuna basıp formu kontrol edin, imzalayın ve PDF olarak yükleyin.
# 4. Muhasebe birimi, imzalı formu staj başlangıcınızdan en az bir hafta önce işleme alır ve 4A işe giriş bildiriminizi yapar.
# 5. Toplamda 60 iş günü; en fazla 45 günlük staj süresini aşmamaya dikkat edin.
#
# **Soru:**
# Eğitime başlamadan önce ne yapmalıyım?
#
# **Cevap:**
# 1. Staj talebini oluşturmak için önce gerekli bütün bilgileri (şirket, tarih vs.) eksiksiz doldurun.
# 2. Talebi kaydettikten sonra “Staj Talepleri / Liste” sayfasından “Kabul formülü oluştur”a tıklayın.
# 3. Oluşan formu hem kendiniz hem de şirket yetkilisi imzalayıp PDF olarak yükleyin.
# 4. Muhasebe birimi, imzalı formu staj başlangıcınızdan bir hafta önce alır ve 4A işe giriş bildiriminizi yapar.
# 5. Toplam çalışma gününüzün 60’ı, staj sürenizin ise 45’i geçmeyeceğini unutmayın.
#
# ---
#
# ### Şimdi senin soruna geçelim
#
# **Bağlam:**
# {context}
#
# **Soru:**
# {question}
#
# Lütfen **sadece** yukarıdaki “Bağlam”ı kullanarak yanıt ver. Eğer verilen bilgiler yeterli değilse, şu ifadeyi kullan:
# > “Materyaller, iyi bir yanıt vermek için yeterli görünmüyor.”
# """

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | assign_context
        | prompt
        | llm
        | StrOutputParser()
    )

    logging.info("Chain created with preserved syntax.")
    return chain
