"""Dense retrieval over the versioned Qdrant child chunks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from fastapi_docs_rag.indexing.embedder import Embedder
from fastapi_docs_rag.indexing.indexer import DENSE_VECTOR_NAME


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    parent_text: str
    url: str
    section_path: list[str]
    anchor: str | None
    docs_version: str
    source_ref: str
    is_deprecated: bool
    deprecation_hint: bool
    content_type: str
    code_language: str | None
    score: float
    doc_type: str = "unknown"
    dense_score: float | None = None
    rerank_score: float | None = None


def build_filter(
    *,
    doc_type: str | None = None,
    docs_version: str | None = None,
    include_deprecated: bool = True,
    include_meta: bool = False,
    include_reference: bool = False,
) -> Filter | None:
    """Build the payload filter used by dense retrieval."""
    must: list[Any] = []
    must_not: list[Any] = []

    if doc_type:
        must.append(FieldCondition(key="doc_type", match=MatchValue(value=doc_type)))
    else:
        if not include_meta:
            must_not.append(
                FieldCondition(key="doc_type", match=MatchValue(value="meta"))
            )
        if not include_reference:
            must_not.append(
                FieldCondition(key="doc_type", match=MatchValue(value="reference"))
            )
    if docs_version:
        must.append(
            FieldCondition(key="docs_version", match=MatchValue(value=docs_version))
        )
    if not include_deprecated:
        must.append(
            FieldCondition(key="is_deprecated", match=MatchValue(value=False))
        )
    if not must and not must_not:
        return None
    return Filter(must=must or None, must_not=must_not or None)


def hit_from_point(point: Any) -> SearchHit:
    payload = point.payload or {}
    raw_score = getattr(point, "score", None)
    score = float(raw_score) if raw_score is not None else 0.0
    return SearchHit(
        chunk_id=str(payload.get("chunk_id", "")),
        doc_id=str(payload.get("doc_id", "")),
        title=str(payload.get("title", "")),
        text=str(payload.get("text", "")),
        parent_text=str(payload.get("parent_text", "")),
        url=str(payload.get("url", "")),
        section_path=list(payload.get("section_path") or []),
        anchor=payload.get("anchor"),
        docs_version=str(payload.get("docs_version", "unknown")),
        source_ref=str(payload.get("source_ref", "unknown")),
        is_deprecated=bool(payload.get("is_deprecated", False)),
        deprecation_hint=bool(payload.get("deprecation_hint", False)),
        content_type=str(payload.get("content_type", "prose")),
        code_language=payload.get("code_language"),
        score=score,
        doc_type=str(payload.get("doc_type", "unknown")),
        dense_score=score if raw_score is not None else None,
    )


class DenseRetriever:
    def __init__(
        self,
        client: QdrantClient,
        embedder: Embedder,
        *,
        collection: str,
    ) -> None:
        self.client = client
        self.embedder = embedder
        self.collection = collection

    def search(
        self,
        query: str,
        *,
        limit: int = 8,
        doc_type: str | None = None,
        docs_version: str | None = None,
        include_deprecated: bool = True,
        include_meta: bool = False,
        include_reference: bool = False,
    ) -> list[SearchHit]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if limit <= 0:
            raise ValueError("limit must be positive")

        vector = self.embedder.embed([query])[0]
        response = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            using=DENSE_VECTOR_NAME,
            query_filter=build_filter(
                doc_type=doc_type,
                docs_version=docs_version,
                include_deprecated=include_deprecated,
                include_meta=include_meta,
                include_reference=include_reference,
            ),
            limit=limit,
            with_payload=True,
        )
        return [self._hit(point) for point in response.points]

    @staticmethod
    def _hit(point: Any) -> SearchHit:
        return hit_from_point(point)
