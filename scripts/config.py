import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """アプリケーション設定を管理するクラス"""

    # Ollama設定
    OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
    OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "mxbai-embed-large")
    OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "elyza-jp-8b")

    # Weaviate設定
    WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
    WEAVIATE_COLLECTION_NAME = os.getenv("WEAVIATE_COLLECTION_NAME", "PDFDocuments")

    # RAG設定
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
    TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))
