from sentence_transformers import CrossEncoder

# Constants
DOC_PATH = "../data/FAQ - CMPE INTERN TR.pdf"
# MODEL_NAME = "llama3.2:3b"
MODEL_NAME = "gemma3:4b" # Google
# MODEL_NAME = "llama3.1:8b" # Meta
# MODEL_NAME = "gemma3n:e2b"
# MODEL_NAME = "llama3.2:1b"
# MODEL_NAME = "RefinedNeuro/RN_TR_R2"
# MODEL_NAME = "phi4-mini" # Microsoft
# HF_MODEL = "meta-llama/Llama-3.2-3B"
# HF_MODEL = "meta-llama/Llama-3.1-8B"
# HF_MODEL = "microsoft/Phi-4-mini-instruct"
HF_MODEL = "google/gemma-3-4b-it"
# HF_MODEL = "google/gemma-3n-E2B"
# HF_MODEL = "meta-llama/Llama-3.2-1B"
EMBEDDING_MODEL = "bge-m3" # best multilingual
# EMBEDDING_MODEL = "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"
VECTOR_STORE_NAME = "intern-rag"
PERSIST_DIRECTORY = "../chroma_db"
# CROSS_ENCODER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
CROSS_ENCODER = CrossEncoder("BAAI/bge-reranker-v2-m3") # best multilingual
USER_AVATAR = "👤"
BOT_AVATAR = "🤖"
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 500
PREFETCH_K = 5
FINAL_K = 2