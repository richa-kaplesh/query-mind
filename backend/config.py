from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # LLM
    # Optional: only needed if the legacy synchronous Groq path is used.
    # The active agent loop routes through gateway_client, not the Groq SDK directly.
    groq_api_key: str = ""
    model_name: str = "openai/gpt-oss-20b"
    temperature: float = 0.1
    max_upload_mb: int = 30
    session_ttl_seconds: int = 1800
    max_sessions: int = 20            # free: ~10-20, paid: raise
    max_concurrent_ingests: int = 1   # free: 1, paid: 3-5
    upload_dir: str = "uploads"
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

    ocr_provider: str = "tesseract"       # swap point for a future hosted provider
    ocr_timeout_seconds: int = 90
    ocr_dpi: int = 200                    # page rasterization resolution for OCR
    
    tesseract_path: str = ""    
    agent_max_iterations: int = 4
    agent_tool_result_max_chars: int = 4000

    reviewer_max_revisions: int = 1
    reviewer_max_iterations: int = 2
    reviewer_skip_rules_enabled: bool = True
    gateway_url: str = Field(default="https://llm-gateway-qewa.onrender.com",
                             validation_alias=AliasChoices("gateway_url", "llm_gateway_url"))
    gateway_api_key: str = Field(default="",
                                 validation_alias=AliasChoices("gateway_api_key", "llm_gateway_key"))
    # Hard cap on gateway requests for ONE user question (agent loop + reviewer + revisions).
    # Worst case without a cap is 14; this stops a rate-limited run from piling on.
    gateway_max_calls_per_request: int = 14
    gateway_max_concurrent_calls: int = 4

    
    class Config:
        env_file = ".env"
        extra = "ignore"
        protected_namespaces = ('settings_',)

settings = Settings()