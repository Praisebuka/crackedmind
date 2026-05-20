from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = ConfigDict(
        protected_namespaces=(),
        env_file=".env",
        extra="ignore"
    )

    # LLM
    anthropic_api_key: str = Field(..., env="ANTHROPIC_API_KEY")
    model_name: str = "claude-sonnet-4-20250514"

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"

    # Retrieval
    max_rag_results: int = 5
    max_candidates: int = 50
    max_recommendations: int = 10

    # Security
    jwt_secret_key: str = "change-me-in-production-seriously"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60
    max_input_length: int = 8000

    # Data paths
    index_path: str = "data/faiss.index"
    items_path: str = "data/items.json"

    # Nigerian layer toggle
    naija_layer_enabled: bool = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()
