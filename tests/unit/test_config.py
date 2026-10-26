"""Unit tests for settings loading."""

from pathlib import Path

from langchain_rag.config import Settings


def test_defaults() -> None:
    settings = Settings(_env_file=None, llm_api_key="", embedding_api_key="")

    assert settings.app_name == "fastapi-docs-rag"
    assert settings.qdrant_collection == "fastapi_docs_current"
    assert settings.docs_checkout_dir == Path("data/raw/fastapi")
    assert settings.embedding_dim == 1024
    assert settings.missing_required() == ["LLM_API_KEY", "EMBEDDING_API_KEY"]


def test_environment_overrides(monkeypatch) -> None:
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_API_KEY", "test-llm-key")
    monkeypatch.setenv("EMBEDDING_API_KEY", "test-embedding-key")
    monkeypatch.setenv("FINAL_TOP_K", "6")

    settings = Settings(_env_file=None)

    assert settings.llm_model == "test-model"
    assert settings.final_top_k == 6
    assert settings.missing_required() == []
