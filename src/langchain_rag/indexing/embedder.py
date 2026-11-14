"""Dense embedding clients.

The project is provider-agnostic: any OpenAI-compatible ``/embeddings``
endpoint works. The client is deliberately small so the indexer can be tested
with a fake embedder and so the provider can be swapped without touching the
retrieval code.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Protocol

import httpx

from langchain_rag.config import Settings


class EmbeddingError(RuntimeError):
    """Raised when an embedding provider returns an unusable response."""


class Embedder(Protocol):
    """Minimal interface used by the indexer."""

    dimension: int
    model: str

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one dense vector per input text, preserving order."""


class OpenAICompatibleEmbedder:
    """Client for an OpenAI-compatible ``/embeddings`` endpoint."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        dimension: int,
        batch_size: int = 32,
        timeout: float = 60.0,
        max_retries: int = 3,
        retry_backoff: float = 1.0,
        client: httpx.Client | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("embedding API key is required")
        if dimension <= 0:
            raise ValueError("embedding dimension must be positive")
        if batch_size <= 0:
            raise ValueError("embedding batch_size must be positive")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.dimension = dimension
        self.batch_size = batch_size
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self._client = client or httpx.Client(timeout=timeout)
        self._owns_client = client is None

    @property
    def url(self) -> str:
        if self.base_url.endswith("/embeddings"):
            return self.base_url
        return f"{self.base_url}/embeddings"

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        items = list(texts)
        if not items:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(items), self.batch_size):
            batch = items[start : start + self.batch_size]
            vectors.extend(self._embed_batch(batch))
        return vectors

    def _embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        payload = {"model": self.model, "input": list(texts)}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.post(
                    self.url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
            except httpx.HTTPError as exc:
                if attempt >= self.max_retries:
                    raise EmbeddingError(f"embedding request failed: {exc}") from exc
                time.sleep(self.retry_backoff * (attempt + 1))
                continue

            if response.status_code >= 400:
                retryable = response.status_code == 429 or response.status_code >= 500
                if retryable and attempt < self.max_retries:
                    time.sleep(self.retry_backoff * (attempt + 1))
                    continue
                detail = response.text[:300].replace("\n", " ")
                raise EmbeddingError(
                    f"embedding request failed with HTTP {response.status_code}: {detail}"
                )

            try:
                body = response.json()
                data = body["data"]
                if not isinstance(data, list) or len(data) != len(texts):
                    raise ValueError("unexpected number of embeddings")
                ordered = sorted(data, key=lambda item: int(item.get("index", 0)))
                vectors: list[list[float]] = []
                for item in ordered:
                    embedding = item["embedding"]
                    if not isinstance(embedding, list) or len(embedding) != self.dimension:
                        raise ValueError(
                            f"expected {self.dimension}-d embedding, "
                            f"got {len(embedding) if isinstance(embedding, list) else 'non-list'}"
                        )
                    vectors.append([float(value) for value in embedding])
                return vectors
            except (KeyError, TypeError, ValueError) as exc:
                raise EmbeddingError(f"invalid embedding response: {exc}") from exc

        raise EmbeddingError("embedding request exhausted retries")


def make_embedder(settings: Settings) -> Embedder:
    """Build the configured embedding client."""
    provider = settings.embedding_provider.lower()
    if provider not in {"openai", "openai_compatible"}:
        raise ValueError(f"unsupported embedding provider: {settings.embedding_provider}")
    if not settings.embedding_api_key:
        raise ValueError(
            "EMBEDDING_API_KEY is required; copy .env.example to .env and fill it in"
        )
    batch_size = settings.embedding_batch_size
    # DashScope text-embedding-v4 rejects batches larger than 10.
    if "dashscope.aliyuncs.com" in settings.embedding_base_url and batch_size > 10:
        batch_size = 10
    return OpenAICompatibleEmbedder(
        api_key=settings.embedding_api_key,
        base_url=settings.embedding_base_url,
        model=settings.embedding_model,
        dimension=settings.embedding_dim,
        batch_size=batch_size,
    )
