"""Hybrid dense + BM25 retrieval fused with Reciprocal Rank Fusion."""

from __future__ import annotations

import time
from dataclasses import replace

from fastapi_docs_rag.retrieval.bm25 import BM25Index
from fastapi_docs_rag.retrieval.dense import DenseRetriever, SearchHit
from fastapi_docs_rag.retrieval.rerank import LexicalReranker, Reranker


class HybridRetriever:
    def __init__(
        self,
        dense: DenseRetriever,
        bm25: BM25Index,
        *,
        dense_top_k: int = 20,
        bm25_top_k: int = 20,
        rrf_k: int = 60,
        dense_weight: float = 2.0,
        bm25_weight: float = 0.5,
        reranker: Reranker | None = None,
        rerank_top_k: int = 20,
        rerank_skip_score: float = 0.75,
    ) -> None:
        self.dense = dense
        self.bm25 = bm25
        self.dense_top_k = dense_top_k
        self.bm25_top_k = bm25_top_k
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.reranker = reranker
        self.rerank_top_k = rerank_top_k
        self.rerank_skip_score = rerank_skip_score

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
        timings: dict[str, float] | None = None,
    ) -> list[SearchHit]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if limit <= 0:
            raise ValueError("limit must be positive")

        dense_hits = self.dense.search(
            query,
            limit=self.dense_top_k,
            doc_type=doc_type,
            docs_version=docs_version,
            include_deprecated=include_deprecated,
            include_meta=include_meta,
            include_reference=include_reference,
            timings=timings,
        )
        started = time.perf_counter()
        bm25_hits = self.bm25.search(
            query,
            limit=self.bm25_top_k,
            doc_type=doc_type,
            docs_version=docs_version,
            include_deprecated=include_deprecated,
            include_meta=include_meta,
            include_reference=include_reference,
        )
        if timings is not None:
            timings["bm25_ms"] = (time.perf_counter() - started) * 1000

        fused: dict[str, float] = {}
        by_id: dict[str, SearchHit] = {}
        for rank, hit in enumerate(dense_hits, start=1):
            key = hit.chunk_id or hit.url
            fused[key] = fused.get(key, 0.0) + self.dense_weight / (self.rrf_k + rank)
            by_id.setdefault(key, hit)
        for rank, hit in enumerate(bm25_hits, start=1):
            key = hit.chunk_id or hit.url
            fused[key] = fused.get(key, 0.0) + self.bm25_weight / (self.rrf_k + rank)
            by_id.setdefault(key, hit)

        ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)
        candidates = [replace(by_id[key], score=score) for key, score in ranked]
        if self.reranker is not None:
            candidates = candidates[: self.rerank_top_k]
            if (
                dense_hits
                and dense_hits[0].dense_score is not None
                and dense_hits[0].dense_score >= self.rerank_skip_score
            ):
                if timings is not None:
                    timings["rerank_ms"] = 0.0
                    timings["rerank_skipped"] = 1.0
                return candidates[:limit]
            started = time.perf_counter()
            try:
                reranked = self.reranker.rerank(query, candidates, limit=limit)
            except Exception:  # noqa: BLE001 - model failure must not break chat
                reranked = LexicalReranker().rerank(query, candidates, limit=limit)
            if timings is not None:
                timings["rerank_ms"] = (time.perf_counter() - started) * 1000
            # Keep the dense top-1 for the lexical fallback; a real
            # cross-encoder is allowed to reorder it.
            if not any(hit.rerank_score is not None for hit in reranked):
                dense_top = next(
                    (
                        hit
                        for hit in candidates
                        if (hit.chunk_id or hit.url)
                        == (dense_hits[0].chunk_id or dense_hits[0].url)
                    ),
                    None,
                )
                if dense_top is not None and limit > 1:
                    tail = [
                        hit
                        for hit in reranked
                        if (hit.chunk_id or hit.url)
                        != (dense_top.chunk_id or dense_top.url)
                    ]
                    reranked = [dense_top] + tail
            return reranked[:limit]
        return candidates[:limit]
