"""Tests for the chat client configuration."""

from langchain_rag.config import Settings
from langchain_rag.generation.llm import make_chat_client


def test_max_tokens_is_capped() -> None:
    settings = Settings(
        _env_file=None,
        llm_api_key="test-key",
        llm_max_tokens=2048,
        llm_max_tokens_cap=512,
    )
    client = make_chat_client(settings)

    assert client.max_tokens == 512
    client.close()
