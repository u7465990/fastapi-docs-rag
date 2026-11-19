"""Version-aware exact-answer cache backed by Redis."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from redis import Redis
from redis.exceptions import RedisError

from fastapi_docs_rag.config import Settings
from fastapi_docs_rag.indexing.manifest import IndexManifest

DEFAULT_MANIFEST = Path("data/manifests/index_manifest.json")
CACHE_PREFIX = "cache:answer"


def index_identity(manifest_path: Path = DEFAULT_MANIFEST) -> str:
    manifest = IndexManifest.load(manifest_path)
    if manifest is None:
        return "unknown"
    return f"{manifest.source_ref}:{manifest.build_id}"


def make_cache_key(
    *,
    query: str,
    history: Sequence[dict[str, str]] | None,
    identity: str,
    settings: Settings,
    options: dict[str, Any] | None = None,
) -> str:
    payload = {
        "query": " ".join(query.lower().split()),
        "history": [
            {"role": message.get("role"), "content": message.get("content")}
            for message in (history or [])
            if message.get("role") in {"user", "assistant"}
        ],
        "identity": identity,
        "embedding_model": settings.embedding_model,
        "embedding_dim": settings.embedding_dim,
        "reranker_provider": settings.reranker_provider,
        "reranker_model": settings.reranker_model,
        "llm_model": settings.llm_model,
        "prompt_version": "v3",
        "options": options or {},
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return f"{CACHE_PREFIX}:{digest}"


class AnswerCache:
    def __init__(self, client: Redis, *, ttl_seconds: int) -> None:
        self.client = client
        self.ttl_seconds = ttl_seconds

    def get(self, key: str) -> dict[str, Any] | None:
        try:
            raw = self.client.get(key)
        except RedisError:
            return None
        if not raw:
            return None
        try:
            value = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def set(self, key: str, value: dict[str, Any]) -> None:
        try:
            self.client.setex(
                key,
                self.ttl_seconds,
                json.dumps(value, ensure_ascii=False),
            )
        except RedisError:
            # Cache failure must never break the chat path.
            return


def make_answer_cache(settings: Settings) -> AnswerCache:
    client = Redis.from_url(settings.redis_url, decode_responses=True)
    return AnswerCache(client, ttl_seconds=settings.cache_ttl_seconds)
