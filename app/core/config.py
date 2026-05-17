from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    cohere_api_key: str = ""
    qdrant_url: str = "http://localhost:6333"
    vector_backend: str = "faiss"
    log_level: str = "INFO"

    class Config:
        env_file = ".env"

settings = Settings()
