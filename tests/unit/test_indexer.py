"""Tests for the Qdrant indexer without a running Qdrant service."""

import uuid
from pathlib import Path
from types import SimpleNamespace

from fastapi_docs_rag.indexing.chunker import chunk_document
from fastapi_docs_rag.indexing.indexer import QdrantIndexer
from fastapi_docs_rag.ingestion.parser import parse_markdown
from fastapi_docs_rag.ingestion.source import SourceInfo


class FakeEmbedder:
    dimension = 4
    model = "fake-embedding"

    def __init__(self) -> None:
        self.texts: list[str] = []

    def embed(self, texts):
        self.texts = list(texts)
        return [[float(index + 1)] * self.dimension for index in range(len(texts))]


class FakeQdrantClient:
    def __init__(self) -> None:
        self.collection_exists_calls: list[str] = []
        self.created_collections: list[tuple[str, object]] = []
        self.payload_indexes: list[tuple[str, str, object]] = []
        self.upserted: list[object] = []
        self.alias_operations: list[object] = []

    def collection_exists(self, collection_name: str) -> bool:
        self.collection_exists_calls.append(collection_name)
        return False

    def create_collection(self, *, collection_name: str, vectors_config) -> None:
        self.created_collections.append((collection_name, vectors_config))

    def create_payload_index(
        self,
        *,
        collection_name: str,
        field_name: str,
        field_schema,
        wait: bool = True,
    ) -> None:
        self.payload_indexes.append((collection_name, field_name, field_schema))

    def upsert(self, *, collection_name: str, points, wait: bool = True) -> None:
        self.upserted.extend(points)

    def get_aliases(self):
        return SimpleNamespace(aliases=[])

    def update_collection_aliases(self, *, change_aliases_operations) -> None:
        self.alias_operations = list(change_aliases_operations)


def _make_doc(tmp_path: Path):
    docs_root = tmp_path / "docs" / "en" / "docs"
    doc_path = docs_root / "tutorial" / "sample.md"
    doc_path.parent.mkdir(parents=True)
    text = (
        "# Sample Page\n\n## Example { #example }\n\n"
        "Hello world.\n\n```python\nx = 1\n```\n"
    )
    doc_path.write_text(text, encoding="utf-8")
    doc = parse_markdown(
        text,
        doc_path=doc_path,
        docs_root=docs_root,
        repo_root=tmp_path,
    )
    return doc, chunk_document(doc)


def _source() -> SourceInfo:
    return SourceInfo(
        docs_version="0.162.0",
        source_ref="git:deadbeef",
        fetched_at="2026-01-01T00:00:00+00:00",
    )


def test_index_writes_only_children_with_parent_context(tmp_path: Path) -> None:
    doc, chunks = _make_doc(tmp_path)
    children = [chunk for chunk in chunks if chunk.level == "child"]
    parents = {chunk.chunk_id: chunk for chunk in chunks if chunk.level == "parent"}
    embedder = FakeEmbedder()
    client = FakeQdrantClient()
    indexer = QdrantIndexer(
        client,
        vector_size=4,
        build_prefix="fastapi_docs_build",
        alias="fastapi_docs_current",
    )

    stats = indexer.index(
        [doc],
        chunks,
        embedder=embedder,
        source=_source(),
        build_id="20260101T000000Z",
    )

    assert stats.build_collection == "fastapi_docs_build_20260101T000000Z"
    assert stats.alias == "fastapi_docs_current"
    assert stats.children == len(children)
    assert stats.vectors == len(children)
    assert len(client.upserted) == len(children)
    assert embedder.texts == [chunk.embed_text for chunk in children]

    point = client.upserted[0]
    uuid.UUID(point.id)
    payload = point.payload
    child = children[0]
    assert payload["chunk_id"] == child.chunk_id
    assert payload["doc_id"] == doc.doc_id
    assert payload["parent_id"] == child.parent_id
    assert payload["title"] == "Sample Page"
    assert payload["url"] == "https://fastapi.tiangolo.com/tutorial/sample/#example"
    assert payload["docs_version"] == "0.162.0"
    assert payload["source_ref"] == "git:deadbeef"
    assert payload["index_build_id"] == "20260101T000000Z"
    assert payload["doc_type"] == "tutorial"
    assert payload["content_type"] in {"prose", "code", "mixed"}
    assert payload["text"] == child.text
    assert payload["parent_text"] == parents[child.parent_id].text
    assert len(payload["content_sha256"]) == 64

    collection, vectors_config = client.created_collections[0]
    assert collection == stats.build_collection
    assert vectors_config["dense"].size == 4
    assert client.payload_indexes
    assert client.alias_operations
    assert client.alias_operations[-1].create_alias.alias_name == "fastapi_docs_current"


def test_index_reuses_vectors_without_embedding(tmp_path: Path) -> None:
    doc, chunks = _make_doc(tmp_path)
    children = [chunk for chunk in chunks if chunk.level == "child"]
    embedder = FakeEmbedder()
    client = FakeQdrantClient()
    indexer = QdrantIndexer(
        client,
        vector_size=4,
        build_prefix="fastapi_docs_build",
        alias="fastapi_docs_current",
    )
    reuse = {chunk.chunk_id: [0.1, 0.2, 0.3, 0.4] for chunk in children}

    stats = indexer.index(
        [doc],
        chunks,
        embedder=embedder,
        source=_source(),
        build_id="20260101T000000Z",
        reuse_vectors=reuse,
    )

    assert stats.reused_vectors == len(children)
    assert stats.embedded_vectors == 0
    assert embedder.texts == []


def test_existing_alias_is_replaced(tmp_path: Path) -> None:
    doc, chunks = _make_doc(tmp_path)
    client = FakeQdrantClient()
    client.get_aliases = lambda: SimpleNamespace(
        aliases=[SimpleNamespace(alias_name="fastapi_docs_current")]
    )
    indexer = QdrantIndexer(
        client,
        vector_size=4,
        build_prefix="build",
        alias="fastapi_docs_current",
    )

    indexer.index(
        [doc],
        chunks,
        embedder=FakeEmbedder(),
        source=_source(),
        build_id="20260101T000000Z",
    )

    assert len(client.alias_operations) == 2
    assert client.alias_operations[0].delete_alias.alias_name == "fastapi_docs_current"
    assert client.alias_operations[1].create_alias.alias_name == "fastapi_docs_current"


def test_no_alias_skips_alias_switch(tmp_path: Path) -> None:
    doc, chunks = _make_doc(tmp_path)
    client = FakeQdrantClient()
    indexer = QdrantIndexer(
        client,
        vector_size=4,
        build_prefix="build",
        alias=None,
    )

    stats = indexer.index(
        [doc],
        chunks,
        embedder=FakeEmbedder(),
        source=_source(),
        build_id="20260101T000000Z",
    )

    assert stats.alias is None
    assert not client.alias_operations
