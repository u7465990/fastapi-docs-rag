"""Tests for the version-aware answer cache."""

from fastapi_docs_rag.config import get_settings
from fastapi_docs_rag.generation.cache import AnswerCache, make_cache_key


class FakeRedis:
    def __init__(self) -> None:
        self.data: dict[str, str] = {}

    def get(self, key: str):
        return self.data.get(key)

    def setex(self, key: str, ttl: int, value: str) -> None:
        assert ttl > 0
        self.data[key] = value


def test_cache_key_changes_with_index_identity() -> None:
    settings = get_settings()
    first = make_cache_key(
        query="Depends 怎么用？",
        history=None,
        identity="git:abc:build-1",
        settings=settings,
    )
    second = make_cache_key(
        query="Depends 怎么用？",
        history=None,
        identity="git:def:build-2",
        settings=settings,
    )

    assert first != second


def test_answer_cache_round_trip() -> None:
    cache = AnswerCache(FakeRedis(), ttl_seconds=60)  # type: ignore[arg-type]

    cache.set("key", {"answer": "ok", "sources": []})

    assert cache.get("key") == {"answer": "ok", "sources": []}
