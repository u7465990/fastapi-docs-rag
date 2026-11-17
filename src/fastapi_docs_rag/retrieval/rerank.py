"""Lightweight lexical reranker for the hybrid candidate set.

This is intentionally not a cross-encoder: it needs no model download and runs
on CPU. It is a transparent feature reranker, useful as an intermediate
baseline before adding a real cross-encoder.
"""

from __future__ import annotations

from dataclasses import replace

from fastapi_docs_rag.retrieval.bm25 import expand_query, tokenize
from fastapi_docs_rag.retrieval.dense import SearchHit


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
