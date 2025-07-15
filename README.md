# Intern-RAG

A Streamlit-based Retrieval-Augmented Generation (RAG) chatbot in Turkish language that helps students with internship FAQs by loading a PDF handbook, building a Chroma vector store, and querying an Ollama-powered LLM.

## Features

- **PDF ingestion** & chunking via `langchain`  
- **Chroma** vector store with Ollama embeddings  
- **Cross-encoder** reranking for top-k results  
- **Chat** UI with persistent history (via `shelve`)  
- **Sidebar** option to clear your conversation

## Requirements

- Python 3.8+  
- [Ollama](https://ollama.com/) installed & models pulled  
- A PDF FAQ at `data/pdf`  

Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

1. **Clone & prepare data**  
   ```bash
   git clone https://github.com/delos0/intern-rag.git
   cd intern-rag
   mkdir data chat chroma_db
   # place your FAQ PDF as data/FAQ - CMPE INTERN TR.pdf
   ```

2. **Run the app**  
   ```bash
   cd src
   streamlit run app.py
   ```

3. **Chat**  
   Open http://localhost:8501 in your browser, ask questions, or click **Delete chat history** in the sidebar to reset.

## Configuration

Edit `src/constants.py` to switch LLM, embedding model, PDF path or retrieval settings:
```python
DOC_PATH = "../data/FAQ - CMPE INTERN TR.pdf"
MODEL_NAME = "gemma3:4b"
EMBEDDING_MODEL = "bge-m3"
VECTOR_STORE_NAME = "intern-rag"
PERSIST_DIRECTORY = "../chroma_db"
```
