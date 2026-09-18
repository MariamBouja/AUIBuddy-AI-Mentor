"""Configuration for the AUIBuddy RAG chatbot."""

# Paths
PDF_PATH = "24-25.pdf"              # AUI course catalog
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "auibuddy_catalog"

# Models (OpenAI API)
LLM_MODEL = "gpt-4.1-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

# Chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", " "]

# Retrieval
VECTOR_K = 6
VECTOR_FETCH_K = 24
VECTOR_LAMBDA_MULT = 0.3
BM25_K = 6
VECTOR_WEIGHT = 0.6
BM25_WEIGHT = 0.4
