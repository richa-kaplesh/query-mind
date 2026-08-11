from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # LLM
    groq_api_key: str
    model_name: str = "llama-3.1-8b-instant"
    temperature: float = 0.1

    # Embedder
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # Retriever
    retriever_alpha: float = 0.5
    retriever_top_k: int = 20

    # Reranker
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_k: int = 5

    # Chunker
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Web search
    web_search_max_results: int = 5

    tesseract_path: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


    class Config:
        env_file = ".env"

settings = Settings()