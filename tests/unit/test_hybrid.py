"""Tests for BM25, RRF fusion, and the lexical reranker."""

from fastapi_docs_rag.retrieval.bm25 import BM25Index, expand_query, tokenize
from fastapi_docs_rag.retrieval.dense import SearchHit
from fastapi_docs_rag.retrieval.hybrid import HybridRetriever
from fastapi_docs_rag.retrieval.rerank import CrossEncoderReranker, LexicalReranker


def _hit(chunk_id: str, url: str, title: str, text: str, score: float = 0.0) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        doc_id=chunk_id,
        title=title,
        text=text,
        parent_text=text,
        url=url,
        section_path=[title],
        anchor=None,
        docs_version="0.162.0",
        source_ref="unknown",
        is_deprecated=False,
        deprecation_hint=False,
        content_type="prose",
        code_language=None,
        score=score,
        doc_type="tutorial",
    )


def test_tokenize_splits_camel_and_snake_case() -> None:
    assert tokenize("OAuth2PasswordBearer TestClient background_tasks") == [
        "oauth2passwordbearer",
        "oauth2",
        "password",
        "bearer",
        "testclient",
        "test",
        "client",
        "background",
        "tasks",
    ]


def test_chinese_only_query_does_not_add_english_glossary() -> None:
    assert expand_query("后台任务怎么用？") == "后台任务怎么用？"


def test_bm25_ranks_matching_document_first() -> None:
    index = BM25Index(
        [
            _hit("a", "https://x/deps/", "Dependencies", "Depends dependency injection"),
            _hit("b", "https://x/tasks/", "Background Tasks", "background tasks run later"),
            _hit("c", "https://x/testing/", "Testing", "TestClient testing"),
        ]
    )

    hits = index.search("background tasks", limit=2)

    assert [hit.chunk_id for hit in hits] == ["b"]


def test_hybrid_rrf_merges_dense_and_bm25() -> None:
    a = _hit("a", "https://x/a/", "A", "a", score=0.9)
    b = _hit("b", "https://x/b/", "B", "b", score=0.8)
    c = _hit("c", "https://x/c/", "C", "c", score=0.7)

    class Dense:
        def search(self, query, **kwargs):
            return [a, b]

    class Sparse:
        def search(self, query, **kwargs):
            return [b, c]

    hybrid = HybridRetriever(
        Dense(),
        Sparse(),
        rrf_k=1,
        dense_weight=1.0,
        bm25_weight=1.0,
    )
    hits = hybrid.search("b", limit=2)

    assert hits[0].chunk_id == "b"
    assert {hit.chunk_id for hit in hits} == {"a", "b"}


def test_lexical_reranker_boosts_title_match() -> None:
    generic = _hit("a", "https://x/a/", "Generic", "nothing", score=0.02)
    titled = _hit("b", "https://x/b/", "Background Tasks", "nothing", score=0.02)

    hits = LexicalReranker().rerank("background tasks", [generic, titled], limit=2)

    assert hits[0].chunk_id == "b"


def test_cross_encoder_reranker_uses_model_scores(tmp_path) -> None:
    class FakeEncoder:
        def rerank(self, query, documents):
            assert query
            assert len(documents) == 2
            return [0.1, 0.9]

    reranker = CrossEncoderReranker(
        "fake",
        cache_dir=tmp_path,
        encoder=FakeEncoder(),
    )
    generic = _hit("a", "https://x/a/", "Generic", "nothing", score=0.02)
    titled = _hit("b", "https://x/b/", "Background Tasks", "nothing", score=0.02)

    hits = reranker.rerank("background tasks", [generic, titled], limit=2)

    assert hits[0].chunk_id == "b"
    assert hits[0].rerank_score == 0.9
