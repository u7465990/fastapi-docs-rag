"""Application settings loaded from environment variables and `.env`."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration.

    Field names map to upper-case environment variables (e.g. ``llm_api_key``
    reads ``LLM_API_KEY``). Unknown variables are ignored so the same `.env`
    can be shared with other tooling.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "fastapi-docs-rag"
    environment: str = "dev"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])

    # LLM
    llm_provider: str = "openai_compatible"
    llm_model: str = "deepseek-chat"
    llm_api_key: str = ""
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2048

    # Embedding
    embedding_provider: str = "openai_compatible"
    embedding_model: str = "text-embedding-v4"
    embedding_api_key: str = ""
    embedding_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    embedding_dim: int = 1024

    # Reranker
    reranker_provider: str = "local"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "fastapi_docs_current"
    qdrant_build_prefix: str = "fastapi_docs_build"

    # Documentation source and incremental index
    docs_repo_url: str = "https://github.com/fastapi/fastapi.git"
    docs_repo_branch: str = "master"
    docs_checkout_dir: Path = Path("data/raw/fastapi")
    index_refresh_cron: str = "0 3 * * *"
    staleness_threshold_hours: int = 24
    default_docs_version: str = "latest"
    include_deprecated: bool = True
    embedding_cache_dir: Path = Path("data/cache/embeddings")

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Retrieval
    retrieval_dense_top_k: int = 40
    retrieval_sparse_top_k: int = 40
    rerank_top_k: int = 20
    final_top_k: int = 8
    max_context_tokens: int = 8000

    # Observability
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "fastapi-rag-dev"

    # API
    admin_api_key: str = ""
    rate_limit_per_minute: int = 30

    def missing_required(self) -> list[str]:
        """Return required settings that are still empty."""
        missing: list[str] = []
        if not self.llm_api_key:
            missing.append("LLM_API_KEY")
        if not self.embedding_api_key:
            missing.append("EMBEDDING_API_KEY")
        return missing


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a process-wide settings instance."""
    return Settings()
