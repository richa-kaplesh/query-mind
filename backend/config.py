from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # LLM
    groq_api_key: str
    model_name: str = "openai/gpt-oss-20b"
    temperature: float = 0.1

    # Jina AI (embeddings + reranking)
    jina_api_key: str = ""
    jina_embed_model: str = "jina-embeddings-v3"
    jina_rerank_model: str = "jina-reranker-v2-base-multilingual"

    # jina-embeddings-v3 default output dimension
    embedding_dimension: int = 1024

    # Retriever
    retriever_alpha: float = 0.5
    retriever_top_k: int = 20

    # Reranker
    reranker_top_k: int = 5

    # Chunker
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Web search
    web_search_max_results: int = 5

    tesseract_path: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    class Config:
        env_file = ".env",
        extra = "ignore"
        protected_namespaces = ('settings_',)

settings = Settings()