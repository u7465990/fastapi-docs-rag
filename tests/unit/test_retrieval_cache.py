"""Tests for the retrieval result cache."""

from fastapi_docs_rag.config import get_settings
from fastapi_docs_rag.retrieval.cache import RetrievalCache, make_retrieval_key
from fastapi_docs_rag.retrieval.dense import SearchHit


class FakeRedis:
    def __init__(self) -> None:
        self.data: dict[str, str] = {}

    def get(self, key: str):
        return self.data.get(key)

    def setex(self, key: str, ttl: int, value: str) -> None:
        assert ttl > 0
        self.data[key] = value


def _hit() -> SearchHit:
    return SearchHit(
        chunk_id="chunk-1",
        doc_id="doc-1",
        title="Dependencies",
        text="child",
        parent_text="parent",
        url="https://fastapi.tiangolo.com/tutorial/dependencies/",
        section_path=["Dependencies"],
        anchor=None,
        docs_version="0.162.0",
        source_ref="git:abc",
        is_deprecated=False,
        deprecation_hint=False,
        content_type="prose",
        code_language=None,
        score=0.9,
        dense_score=0.8,
    )


def test_retrieval_cache_round_trip() -> None:
    cache = RetrievalCache(FakeRedis(), ttl_seconds=30)  # type: ignore[arg-type]
    key = "key"

    cache.set(key, [_hit()])
    hits = cache.get(key)

    assert hits is not None
    assert hits[0].chunk_id == "chunk-1"
    assert hits[0].dense_score == 0.8


def test_retrieval_cache_key_changes_with_index_identity() -> None:
    settings = get_settings()
    kwargs = {
        "query": "Depends 怎么用？",
        "limit": 8,
        "doc_type": None,
        "docs_version": None,
        "include_deprecated": True,
        "include_meta": False,
        "include_reference": False,
        "settings": settings,
    }

    first = make_retrieval_key(identity="git:a:build-1", **kwargs)
    second = make_retrieval_key(identity="git:b:build-2", **kwargs)

    assert first != second
