"""Dense retrieval tests against an in-memory Qdrant collection."""

from pathlib import Path

from qdrant_client import QdrantClient

from langchain_rag.indexing.chunker import chunk_document
from langchain_rag.indexing.indexer import QdrantIndexer
from langchain_rag.ingestion.parser import parse_markdown
from langchain_rag.ingestion.source import SourceInfo
from langchain_rag.retrieval.dense import DenseRetriever, build_filter


class ConstantEmbedder:
    dimension = 4
    model = "constant"

    def embed(self, texts):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


def _doc(tmp_path: Path):
    docs_root = tmp_path / "docs" / "en" / "docs"
    doc_path = docs_root / "tutorial" / "sample.md"
    doc_path.parent.mkdir(parents=True)
    text = (
        "# Sample Page\n\n## Example { #example }\n\n"
        "Hello world.\n\n```python\nx = 1\n```\n\n"
        "## Old Events\n\n/// warning\n\n`on_event` is deprecated.\n\n///\n"
    )
    doc_path.write_text(text, encoding="utf-8")
    doc = parse_markdown(
        text,
        doc_path=doc_path,
        docs_root=docs_root,
        repo_root=tmp_path,
    )
    return doc


def _retriever(tmp_path: Path):
    doc = _doc(tmp_path)
    chunks = chunk_document(doc)
    client = QdrantClient(":memory:")
    indexer = QdrantIndexer(
        client,
        vector_size=4,
        build_prefix="build",
        alias=None,
    )
    source = SourceInfo("0.162.0", "unknown", "2026-01-01T00:00:00+00:00")
    stats = indexer.index(
        [doc],
        chunks,
        embedder=ConstantEmbedder(),
        source=source,
        build_id="20260101T000000Z",
    )
    return (
        DenseRetriever(client, ConstantEmbedder(), collection=stats.build_collection),
        client,
    )


def test_search_returns_parent_text_and_metadata(tmp_path: Path) -> None:
    retriever, client = _retriever(tmp_path)
    try:
        hits = retriever.search("anything", limit=10)
    finally:
        client.close()

    assert hits
    hit = hits[0]
    assert hit.title == "Sample Page"
    assert hit.url.startswith("https://fastapi.tiangolo.com/tutorial/sample/")
    assert hit.parent_text
    assert hit.docs_version == "0.162.0"
    assert 0.0 <= hit.score <= 1.0


def test_search_filters_deprecated_and_doc_type(tmp_path: Path) -> None:
    retriever, client = _retriever(tmp_path)
    try:
        all_hits = retriever.search("anything", limit=10)
        clean_hits = retriever.search(
            "anything", limit=10, include_deprecated=False
        )
        meta_hits = retriever.search("anything", limit=10, doc_type="meta")
    finally:
        client.close()

    assert any(hit.is_deprecated for hit in all_hits)
    assert clean_hits
    assert all(not hit.is_deprecated for hit in clean_hits)
    assert meta_hits == []


def test_reference_pages_are_excluded_by_default() -> None:
    default_filter = build_filter()
    with_reference = build_filter(include_reference=True)

    assert default_filter is not None
    assert default_filter.must_not is not None
    assert any(
        condition.match.value == "reference"
        for condition in default_filter.must_not
        if hasattr(condition, "match")
    )
    assert with_reference is not None
    assert not any(
        condition.match.value == "reference"
        for condition in (with_reference.must_not or [])
        if hasattr(condition, "match")
    )


def test_empty_query_is_rejected(tmp_path: Path) -> None:
    retriever, client = _retriever(tmp_path)
    try:
        try:
            retriever.search("  ")
        except ValueError as exc:
            assert "empty" in str(exc)
        else:
            raise AssertionError("expected ValueError")
    finally:
        client.close()
