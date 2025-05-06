import os

# --- Core RAG Setup ---
DOCS_DIRECTORY = "./long_files"
CHUNK_SIZE = 200
# --- Embedding Model to run on macbook ---
# EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
# INDEX_FILE = "index.faiss"

# --- Embedding Model to run on colab ---
EMBEDDING_MODEL = 'intfloat/multilingual-e5-large-instruct'
INDEX_FILE = "indexE5.faiss"

# --- LLM Configuration ---
# Used for Q&A Generation in pipeline 1 and Answer Generation in pipeline 2 & 4
QA_GENERATION_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free" 
# Ensure the API key is set as an environment variable
TOGETHER_API_KEY = os.getenv('TOGETHER_API_KEY')

# --- Retrieval Settings ---
# Used in pipeline 1 (augmentation), 2 (interactive), 3 (eval), 4 (eval)
RETRIEVAL_K = 5 