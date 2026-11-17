"""Tests for the versioned index manifest."""

from pathlib import Path

from fastapi_docs_rag.indexing.manifest import IndexManifest


def test_manifest_roundtrips(tmp_path: Path) -> None:
    manifest = IndexManifest(
        build_id="20260101T000000Z",
        build_collection="fastapi_docs_build_20260101T000000Z",
        source_ref="git:deadbeef",
        docs_version="0.162.0",
        embedding_model="fake-embedding",
        embedding_dim=4,
        created_at="2026-01-01T00:00:00+00:00",
        previous_build_id=None,
        previous_source_ref=None,
        documents={"tutorial/sample.md": "doc-hash"},
        chunks={"chunk-1": "chunk-hash"},
    )
    path = tmp_path / "manifests" / "index_manifest.json"

    manifest.save(path)
    loaded = IndexManifest.load(path)

    assert loaded == manifest
    assert IndexManifest.load(tmp_path / "missing.json") is None
