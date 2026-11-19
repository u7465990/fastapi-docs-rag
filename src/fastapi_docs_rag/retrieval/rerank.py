"""Lightweight lexical reranker for the hybrid candidate set.

This is intentionally not a cross-encoder: it needs no model download and runs
on CPU. It is a transparent feature reranker, useful as an intermediate
baseline before adding a real cross-encoder.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol

from fastapi_docs_rag.retrieval.bm25 import expand_query, tokenize
from fastapi_docs_rag.retrieval.dense import SearchHit


class Reranker(Protocol):
    def rerank(
        self,
        query: str,
        hits: list[SearchHit],
        *,
        limit: int,
    ) -> list[SearchHit]: ...


class LexicalReranker:
    def __init__(self, *, title_weight: float = 0.05, phrase_weight: float = 0.01):
        self.title_weight = title_weight
        self.phrase_weight = phrase_weight

    def rerank(
        self,
        query: str,
        hits: list[SearchHit],
        *,
        limit: int,
    ) -> list[SearchHit]:
        query_terms = set(tokenize(expand_query(query)))
        if not query_terms:
            return hits[:limit]

        query_lower = query.lower()
        rescored: list[SearchHit] = []
        for hit in hits:
            title_text = f"{hit.title} {' '.join(hit.section_path)}".lower()
            title_terms = set(tokenize(title_text))
            url_terms = set(tokenize(hit.url))
            title_overlap = len(query_terms & title_terms) / max(1, len(query_terms))
            url_overlap = len(query_terms & url_terms) / max(1, len(query_terms))
            phrase_hits = sum(
                1
                for term in query_terms
                if len(term) >= 4 and term in title_text
            )
            boost = (
                self.title_weight * title_overlap
                + self.title_weight * 0.4 * url_overlap
                + self.phrase_weight * min(phrase_hits, 3)
            )
            if query_lower and query_lower in title_text:
                boost += self.title_weight
            rescored.append(replace(hit, score=hit.score + boost))
        return sorted(rescored, key=lambda item: item.score, reverse=True)[:limit]


class CrossEncoderReranker:
    """Semantic cross-encoder reranker backed by FastEmbed ONNX.

    The model is loaded lazily; if FastEmbed or the model is unavailable, the
    caller can fall back to LexicalReranker instead of failing the request.
    """

    def __init__(
        self,
        model_name: str,
        *,
        cache_dir: Path,
        threads: int = 4,
        encoder: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.threads = threads
        self._encoder = encoder
        self._failed = False

    def _get_encoder(self) -> Any:
        if self._failed:
            raise RuntimeError("cross-encoder is unavailable; using lexical fallback")
        if self._encoder is None:
            try:
                from fastembed.rerank.cross_encoder import TextCrossEncoder

                self.cache_dir.mkdir(parents=True, exist_ok=True)
                self._encoder = TextCrossEncoder(
                    model_name=self.model_name,
                    cache_dir=str(self.cache_dir),
                    threads=self.threads,
                    lazy_load=True,
                )
            except Exception:
                self._failed = True
                raise
        return self._encoder

    def rerank(
        self,
        query: str,
        hits: list[SearchHit],
        *,
        limit: int,
    ) -> list[SearchHit]:
        if not hits:
            return []
        documents = [
            f"{hit.title}\n{' > '.join(hit.section_path)}\n{hit.text}"
            for hit in hits
        ]
        scores = list(self._get_encoder().rerank(query, documents))
        if len(scores) != len(hits):
            raise RuntimeError(
                f"reranker returned {len(scores)} scores for {len(hits)} hits"
            )
        rescored = [
            replace(hit, score=float(score), rerank_score=float(score))
            for hit, score in zip(hits, scores, strict=True)
        ]
        return sorted(rescored, key=lambda item: item.score, reverse=True)[:limit]
