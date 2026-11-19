"""Redis cache for retrieved chunks, keyed by the active index identity."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

from redis import Redis
from redis.exceptions import RedisError

from fastapi_docs_rag.config import Settings
from fastapi_docs_rag.retrieval.dense import SearchHit

PREFIX = "cache:retrieval"


def make_retrieval_key(
    *,
    query: str,
    limit: int,
    doc_type: str | None,
    docs_version: str | None,
    include_deprecated: bool,
    include_meta: bool,
    include_reference: bool,
    identity: str,
    settings: Settings,
) -> str:
    payload = {
        "query": " ".join(query.lower().split()),
        "limit": limit,
        "doc_type": doc_type,
        "docs_version": docs_version,
        "include_deprecated": include_deprecated,
        "include_meta": include_meta,
        "include_reference": include_reference,
        "identity": identity,
        "embedding_model": settings.embedding_model,
        "reranker": settings.reranker_model if settings.rerank_enabled else None,
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return f"{PREFIX}:{digest}"


class RetrievalCache:
    def __init__(self, client: Redis, *, ttl_seconds: int) -> None:
        self.client = client
        self.ttl_seconds = ttl_seconds

    def get(self, key: str) -> list[SearchHit] | None:
        try:
            raw = self.client.get(key)
        except RedisError:
            return None
        if not raw:
            return None
        try:
            rows = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return None
        if not isinstance(rows, list):
            return None
        try:
            return [SearchHit(**row) for row in rows if isinstance(row, dict)]
        except TypeError:
            return None

    def set(self, key: str, hits: list[SearchHit]) -> None:
        try:
            self.client.setex(
                key,
                self.ttl_seconds,
                json.dumps([asdict(hit) for hit in hits], ensure_ascii=False),
            )
        except RedisError:
            return


def make_retrieval_cache(settings: Settings) -> RetrievalCache:
    return RetrievalCache(
        Redis.from_url(settings.redis_url, decode_responses=True),
        ttl_seconds=settings.retrieval_cache_ttl_seconds,
    )
