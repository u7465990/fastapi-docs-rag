"""Build a versioned Qdrant collection from parsed documents and chunks.

Only child chunks are embedded and written; each point carries its parent text
so retrieval can return the larger context without a separate doc store.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    CreateAlias,
    CreateAliasOperation,
    DeleteAlias,
    DeleteAliasOperation,
    Distance,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from fastapi_docs_rag.indexing.chunker import Chunk
from fastapi_docs_rag.indexing.embedder import Embedder
from fastapi_docs_rag.ingestion.parser import ParsedDocument
from fastapi_docs_rag.ingestion.source import SourceInfo

DENSE_VECTOR_NAME = "dense"
UPSERT_BATCH_SIZE = 128
_POINT_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "fastapi-docs-rag/points")
_COLLECTION_SAFE_RE = re.compile(r"[^A-Za-z0-9_-]+")

PAYLOAD_INDEXES: tuple[tuple[str, PayloadSchemaType], ...] = (
    ("doc_type", PayloadSchemaType.KEYWORD),
    ("docs_version", PayloadSchemaType.KEYWORD),
    ("content_type", PayloadSchemaType.KEYWORD),
    ("is_deprecated", PayloadSchemaType.BOOL),
    ("deprecation_hint", PayloadSchemaType.BOOL),
)


@dataclass(frozen=True)
class IndexStats:
    build_id: str
    build_collection: str
    alias: str | None
    documents: int
    children: int
    vectors: int
    reused_vectors: int = 0
    embedded_vectors: int = 0


def point_id_for(chunk_id: str) -> str:
    """Return a deterministic UUID accepted by Qdrant."""
    return str(uuid.uuid5(_POINT_NAMESPACE, chunk_id))


def build_payloads(
    documents: Sequence[ParsedDocument],
    chunks: Sequence[Chunk],
    *,
    source: SourceInfo,
    build_id: str,
) -> list[tuple[Chunk, dict[str, Any]]]:
    """Build one payload per child chunk, with its parent text attached."""
    docs = {document.doc_id: document for document in documents}
    parents = {chunk.chunk_id: chunk for chunk in chunks if chunk.level == "parent"}
    payloads: list[tuple[Chunk, dict[str, Any]]] = []

    for chunk in chunks:
        if chunk.level != "child":
            continue
        document = docs.get(chunk.doc_id)
        if document is None:
            raise ValueError(f"document metadata missing for chunk {chunk.chunk_id}")
        parent = parents.get(chunk.parent_id)
        payloads.append(
            (
                chunk,
                {
                    "chunk_id": chunk.chunk_id,
                    "parent_id": chunk.parent_id,
                    "doc_id": chunk.doc_id,
                    "title": document.title,
                    "section_path": list(chunk.section_path),
                    "anchor": chunk.anchor,
                    "url": chunk.url,
                    "source_path": document.relative_path.as_posix(),
                    "product": "fastapi",
                    "language": "python",
                    "doc_type": document.doc_type,
                    "docs_version": source.docs_version,
                    "source_ref": source.source_ref,
                    "fetched_at": source.fetched_at,
                    "index_build_id": build_id,
                    "is_deprecated": chunk.is_deprecated,
                    "deprecation_hint": chunk.deprecation_hint,
                    "content_type": chunk.content_type,
                    "code_language": chunk.code_language,
                    "content_sha256": hashlib.sha256(
                        chunk.text.encode("utf-8")
                    ).hexdigest(),
                    "token_count": chunk.token_count,
                    "text": chunk.text,
                    "parent_text": parent.text if parent else chunk.text,
                },
            )
        )
    return payloads


class QdrantIndexer:
    """Index chunks into a build collection and atomically move an alias."""

    def __init__(
        self,
        client: QdrantClient,
        *,
        vector_size: int,
        build_prefix: str,
        alias: str | None,
        upsert_batch_size: int = UPSERT_BATCH_SIZE,
    ) -> None:
        if vector_size <= 0:
            raise ValueError("vector_size must be positive")
        if upsert_batch_size <= 0:
            raise ValueError("upsert_batch_size must be positive")
        self.client = client
        self.vector_size = vector_size
        self.build_prefix = build_prefix
        self.alias = alias
        self.upsert_batch_size = upsert_batch_size

    def build_collection_name(self, build_id: str) -> str:
        safe = _COLLECTION_SAFE_RE.sub("-", build_id).strip("-") or "build"
        return f"{self.build_prefix}_{safe}"

    def index(
        self,
        documents: Sequence[ParsedDocument],
        chunks: Sequence[Chunk],
        *,
        embedder: Embedder,
        source: SourceInfo,
        build_id: str,
        reuse_vectors: Mapping[str, Sequence[float]] | None = None,
    ) -> IndexStats:
        payloads = build_payloads(documents, chunks, source=source, build_id=build_id)
        if not payloads:
            raise ValueError("no child chunks to index")

        vectors: list[list[float] | None] = [None] * len(payloads)
        missing_indices: list[int] = []
        missing_texts: list[str] = []
        for index, (chunk, _payload) in enumerate(payloads):
            candidate = (
                reuse_vectors.get(chunk.chunk_id) if reuse_vectors is not None else None
            )
            if candidate is not None and len(candidate) == self.vector_size:
                vectors[index] = [float(value) for value in candidate]
            else:
                missing_indices.append(index)
                missing_texts.append(chunk.embed_text)

        if missing_texts:
            embedded = embedder.embed(missing_texts)
            if len(embedded) != len(missing_texts):
                raise ValueError(
                    f"embedder returned {len(embedded)} vectors for "
                    f"{len(missing_texts)} chunks"
                )
            for index, vector in zip(missing_indices, embedded, strict=True):
                vectors[index] = vector

        final_vectors = [vector for vector in vectors if vector is not None]
        if len(final_vectors) != len(payloads):
            raise ValueError("failed to assign a vector to every child chunk")

        build_collection = self.build_collection_name(build_id)
        self._create_collection(build_collection)
        points = [
            PointStruct(
                id=point_id_for(chunk.chunk_id),
                vector={DENSE_VECTOR_NAME: vector},
                payload=payload,
            )
            for (chunk, payload), vector in zip(payloads, final_vectors, strict=True)
        ]
        for start in range(0, len(points), self.upsert_batch_size):
            self.client.upsert(
                collection_name=build_collection,
                points=points[start : start + self.upsert_batch_size],
                wait=True,
            )
        self._create_payload_indexes(build_collection)
        alias = self._activate_alias(build_collection)

        return IndexStats(
            build_id=build_id,
            build_collection=build_collection,
            alias=alias,
            documents=len(documents),
            children=len(payloads),
            vectors=len(final_vectors),
            reused_vectors=len(payloads) - len(missing_texts),
            embedded_vectors=len(missing_texts),
        )

    def _create_collection(self, collection_name: str) -> None:
        if self.client.collection_exists(collection_name):
            self.client.delete_collection(collection_name)
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config={
                DENSE_VECTOR_NAME: VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                )
            },
        )

    def _create_payload_indexes(self, collection_name: str) -> None:
        for field_name, field_schema in PAYLOAD_INDEXES:
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=field_schema,
                wait=True,
            )

    def _activate_alias(self, collection_name: str) -> str | None:
        if not self.alias:
            return None
        aliases = self.client.get_aliases().aliases or []
        existing = {alias.alias_name for alias in aliases}
        operations: list[CreateAliasOperation | DeleteAliasOperation] = []
        if self.alias in existing:
            operations.append(
                DeleteAliasOperation(
                    delete_alias=DeleteAlias(alias_name=self.alias)
                )
            )
        operations.append(
            CreateAliasOperation(
                create_alias=CreateAlias(
                    collection_name=collection_name,
                    alias_name=self.alias,
                )
            )
        )
        self.client.update_collection_aliases(change_aliases_operations=operations)
        return self.alias
