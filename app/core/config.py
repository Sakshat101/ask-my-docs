from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5-coder:14b"
    embedding_model: str = "nomic-embed-text"
    qdrant_url: str = "http://localhost:6333"
    vector_backend: str = "faiss"
    log_level: str = "INFO"

    class Config:
        env_file = ".env"

settings = Settings()
