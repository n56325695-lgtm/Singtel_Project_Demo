import os
from dotenv import load_dotenv
load_dotenv()

# Qdrant
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_documents")


# # Embedding model
# EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3:latest")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))  # bge-m3 output dimension

# LLM --> this model is available on Groq (cloud)
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Chunking / retrieval
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
TOP_K = int(os.getenv("TOP_K", 5))

# Paths
DOCUMENTS_DIR = "documents"

# LLM (Ollama - local)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:8b")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")  # or "groq"

MIN_IMG_SIZE = int(os.getenv("MIN_IMG_SIZE", 100))            # px, skip logos/icons smaller than this
CAPTION_PROXIMITY = int(os.getenv("CAPTION_PROXIMITY", 150))       # points, max vertical distance to link a caption to an image
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.50))  # tune this once you've logged real scores

