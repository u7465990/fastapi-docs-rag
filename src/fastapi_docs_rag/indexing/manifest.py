"""Persistent manifest for version-aware incremental indexing."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from fastapi_docs_rag.indexing.chunker import Chunk
from fastapi_docs_rag.ingestion.parser import ParsedDocument
from fastapi_docs_rag.ingestion.source import SourceInfo


@dataclass(frozen=True)
class IndexManifest:
    build_id: str
    build_collection: str
    source_ref: str
    docs_version: str
    embedding_model: str
    embedding_dim: int
    created_at: str
    previous_build_id: str | None
    previous_source_ref: str | None
    documents: dict[str, str]
    chunks: dict[str, str]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.as_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path) -> IndexManifest | None:
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            build_id=str(data["build_id"]),
            build_collection=str(data["build_collection"]),
            source_ref=str(data["source_ref"]),
            docs_version=str(data["docs_version"]),
            embedding_model=str(data["embedding_model"]),
            embedding_dim=int(data["embedding_dim"]),
            created_at=str(data["created_at"]),
            previous_build_id=data.get("previous_build_id"),
            previous_source_ref=data.get("previous_source_ref"),
            documents={str(k): str(v) for k, v in data.get("documents", {}).items()},
            chunks={str(k): str(v) for k, v in data.get("chunks", {}).items()},
        )


def save_manifest_history(manifest: IndexManifest, live_path: Path) -> Path:
    """Keep every build manifest for version/history inspection."""
    history_path = live_path.parent / "history" / f"{manifest.build_id}.json"
    manifest.save(history_path)
    return history_path


def document_hashes(documents: Sequence[ParsedDocument]) -> dict[str, str]:
    return {
        document.relative_path.as_posix(): document.content_sha256
        for document in documents
    }


def chunk_hashes(chunks: Sequence[Chunk]) -> dict[str, str]:
    return {
        chunk.chunk_id: hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()
        for chunk in chunks
        if chunk.level == "child"
    }


def build_manifest(
    *,
    build_id: str,
    build_collection: str,
    source: SourceInfo,
    embedding_model: str,
    embedding_dim: int,
    documents: Sequence[ParsedDocument],
    chunks: Sequence[Chunk],
    previous: IndexManifest | None,
) -> IndexManifest:
    return IndexManifest(
        build_id=build_id,
        build_collection=build_collection,
        source_ref=source.source_ref,
        docs_version=source.docs_version,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        previous_build_id=previous.build_id if previous else None,
        previous_source_ref=previous.source_ref if previous else None,
        documents=document_hashes(documents),
        chunks=chunk_hashes(chunks),
    )


def diff_manifest(
    previous: IndexManifest | None,
    documents: Sequence[ParsedDocument],
    chunks: Sequence[Chunk],
) -> dict[str, list[str] | int]:
    """Compute document/chunk changes against a previous manifest."""
    current_docs = document_hashes(documents)
    current_chunks = chunk_hashes(chunks)
    old_docs = previous.documents if previous else {}
    old_chunks = previous.chunks if previous else {}

    added_docs = sorted(set(current_docs) - set(old_docs))
    deleted_docs = sorted(set(old_docs) - set(current_docs))
    changed_docs = sorted(
        path
        for path in set(current_docs) & set(old_docs)
        if current_docs[path] != old_docs[path]
    )
    added_chunks = sorted(set(current_chunks) - set(old_chunks))
    deleted_chunks = sorted(set(old_chunks) - set(current_chunks))
    changed_chunks = sorted(
        chunk_id
        for chunk_id in set(current_chunks) & set(old_chunks)
        if current_chunks[chunk_id] != old_chunks[chunk_id]
    )
    return {
        "added_docs": added_docs,
        "changed_docs": changed_docs,
        "deleted_docs": deleted_docs,
        "added_chunks": added_chunks,
        "changed_chunks": changed_chunks,
        "deleted_chunks": deleted_chunks,
        "reused_chunks": len(set(current_chunks) & set(old_chunks))
        - len(changed_chunks),
    }
