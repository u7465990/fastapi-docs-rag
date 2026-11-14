"""Tests for the OpenAI-compatible embedding client."""

import json

import httpx
import pytest

from langchain_rag.config import Settings
from langchain_rag.indexing.embedder import (
    EmbeddingError,
    OpenAICompatibleEmbedder,
    make_embedder,
)


def _embedder(handler, *, dimension: int = 2, batch_size: int = 2):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    embedder = OpenAICompatibleEmbedder(
        api_key="test-key",
        base_url="https://example.test/v1",
        model="test-embedding",
        dimension=dimension,
        batch_size=batch_size,
        client=client,
    )
    return embedder, client


def test_embed_batches_and_preserves_order() -> None:
    calls: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        calls.append(payload["input"])
        data = [
            {"index": index, "embedding": [float(len(text)), 1.0]}
            for index, text in enumerate(payload["input"])
        ]
        return httpx.Response(200, json={"data": data})

    embedder, client = _embedder(handler)
    try:
        vectors = embedder.embed(["a", "bb", "ccc"])
    finally:
        client.close()

    assert calls == [["a", "bb"], ["ccc"]]
    assert vectors == [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]


def test_dimension_mismatch_is_an_embedding_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"data": [{"index": 0, "embedding": [1.0, 2.0]}]},
        )

    embedder, client = _embedder(handler, dimension=3, batch_size=1)
    try:
        with pytest.raises(EmbeddingError, match="3-d"):
            embedder.embed(["a"])
    finally:
        client.close()


def test_empty_input_does_not_call_provider() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("provider should not be called")

    embedder, client = _embedder(handler)
    try:
        assert embedder.embed([]) == []
    finally:
        client.close()


def test_factory_requires_embedding_key() -> None:
    settings = Settings(
        _env_file=None,
        llm_api_key="",
        embedding_api_key="",
    )
    with pytest.raises(ValueError, match="EMBEDDING_API_KEY"):
        make_embedder(settings)


def test_factory_builds_configured_client() -> None:
    settings = Settings(
        _env_file=None,
        embedding_api_key="key",
        embedding_base_url="https://example.test/v1",
        embedding_model="test-model",
        embedding_dim=8,
        embedding_batch_size=7,
    )
    embedder = make_embedder(settings)

    assert isinstance(embedder, OpenAICompatibleEmbedder)
    assert embedder.model == "test-model"
    assert embedder.dimension == 8
    assert embedder.batch_size == 7
    embedder.close()


def test_dashscope_batch_size_is_capped() -> None:
    settings = Settings(
        _env_file=None,
        embedding_api_key="key",
        embedding_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        embedding_batch_size=32,
    )
    embedder = make_embedder(settings)

    assert embedder.batch_size == 10
    embedder.close()
