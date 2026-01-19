from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # LLM Configuration
    llm_provider: str
    ollama_base_url: str
    ollama_model: str
    
    # Vector Store Configuration
    vector_backend: str
    faiss_index_path: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    
    # API Configuration
    api_host: str
    api_port: int
    debug: bool
    
    # Storage Configuration
    upload_dir: str
    processed_dir: str
    log_level: str
    
    # Database (pgvector) Configuration
    database_url: str
    postgres_user: str
    postgres_password: str
    postgres_db: str
    postgres_port: int
    
    # Security
    secret_key: str
    access_token_expire_minutes: int

    # Celery / Task Queue
    celery_broker_url: str
    celery_result_backend: str

    gemini_api_key:str
    gemini_model:str

    # Apple Sign-In Configuration
    apple_team_id: str = ""
    apple_client_id: str = ""
    apple_key_id: str = ""
    apple_private_key_path: str = ""

    class Config:
        env_file = ".env"

settings = Settings()
