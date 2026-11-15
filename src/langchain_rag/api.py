"""FastAPI application entry point."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from functools import lru_cache
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

from langchain_rag import __version__
from langchain_rag.config import Settings, get_settings
from langchain_rag.generation.chat import (
    REFUSAL,
    build_messages,
    estimate_chat_tokens,
)
from langchain_rag.generation.llm import ChatClient, make_chat_client
from langchain_rag.generation.quota import (
    DailyQuota,
    QuotaExceeded,
    QuotaUnavailable,
    make_daily_quota,
)
from langchain_rag.indexing.embedder import make_embedder
from langchain_rag.retrieval.dense import DenseRetriever, SearchHit
from langchain_rag.security import (
    RateLimiter,
    RateLimitExceeded,
    RateLimitUnavailable,
    check_access_token,
    client_identifier,
    make_rate_limiter,
)

_settings = get_settings()

app = FastAPI(
    title="FastAPI Docs RAG",
    version=__version__,
    description="Version-aware RAG over FastAPI documentation.",
)

_cors_origins = list(_settings.cors_origins)
if _settings.environment == "dev":
    for _origin in ("http://localhost:5173", "http://127.0.0.1:5173"):
        if _origin not in _cors_origins:
            _cors_origins.append(_origin)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "service": _settings.app_name,
        "version": __version__,
        "docs": "/docs",
        "health": "/healthz",
    }


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    """Liveness probe. Does not check external dependencies."""
    return {"status": "ok"}


async def _check_qdrant(settings: Settings) -> dict[str, Any]:
    url = settings.qdrant_url.rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=1.5) as client:
            response = await client.get(f"{url}/readyz")
        return {"ok": response.status_code == 200, "status": response.status_code}
    except Exception as exc:  # noqa: BLE001 - probe reports any failure
        return {"ok": False, "error": type(exc).__name__}


async def _check_redis(settings: Settings) -> dict[str, Any]:
    try:
        import redis.asyncio as redis

        client = redis.from_url(
            settings.redis_url,
            socket_connect_timeout=1.5,
            socket_timeout=1.5,
        )
        try:
            pong = await client.ping()
        finally:
            await client.aclose()
        return {"ok": bool(pong)}
    except Exception as exc:  # noqa: BLE001 - probe reports any failure
        return {"ok": False, "error": type(exc).__name__}


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=8, ge=1, le=50)
    doc_type: str | None = None
    docs_version: str | None = None
    include_deprecated: bool = True
    include_meta: bool = False
    include_reference: bool = False


class SearchResult(BaseModel):
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


class SearchResponse(BaseModel):
    query: str
    count: int
    results: list[SearchResult]


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=6, ge=1, le=10)
    doc_type: str | None = None
    docs_version: str | None = None
    include_deprecated: bool = True
    include_meta: bool = False
    include_reference: bool = False


class ChatSource(BaseModel):
    index: int
    title: str
    url: str
    section_path: list[str]
    anchor: str | None
    docs_version: str
    is_deprecated: bool
    deprecation_hint: bool
    snippet: str


class ChatResponse(BaseModel):
    query: str
    answer: str
    model: str
    sources: list[ChatSource]


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _sources_for_hits(hits: list[SearchHit]) -> list[ChatSource]:
    return [
        ChatSource(
            index=index,
            title=hit.title,
            url=hit.url,
            section_path=hit.section_path,
            anchor=hit.anchor,
            docs_version=hit.docs_version,
            is_deprecated=hit.is_deprecated,
            deprecation_hint=hit.deprecation_hint,
            snippet=(hit.parent_text or hit.text)[:600],
        )
        for index, hit in enumerate(hits, start=1)
    ]


@lru_cache(maxsize=1)
def _get_retriever() -> DenseRetriever:
    settings = get_settings()
    embedder = make_embedder(settings)
    client = QdrantClient(url=settings.qdrant_url)
    return DenseRetriever(
        client,
        embedder,
        collection=settings.qdrant_collection,
    )


@lru_cache(maxsize=1)
def _get_chat_client() -> ChatClient:
    return make_chat_client(get_settings())


@lru_cache(maxsize=1)
def _get_quota_guard() -> DailyQuota:
    return make_daily_quota(get_settings())


@lru_cache(maxsize=1)
def _get_embedding_quota() -> DailyQuota:
    settings = get_settings()
    return make_daily_quota(
        settings,
        namespace="embedding",
        request_limit=settings.embedding_daily_request_limit,
        token_budget=settings.embedding_daily_token_budget,
    )


@lru_cache(maxsize=1)
def _get_rate_limiter() -> RateLimiter:
    return make_rate_limiter(get_settings())


def _guard_http(http_request: Request, scope: str) -> None:
    settings = get_settings()
    check_access_token(http_request, settings.api_access_token)
    try:
        _get_rate_limiter().check(scope, client_identifier(http_request))
    except RateLimitExceeded as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except RateLimitUnavailable as exc:
        raise HTTPException(
            status_code=503,
            detail="rate limiter unavailable",
        ) from exc


def _reserve_embedding(query: str) -> None:
    estimated = max(1, len(query) // 3 + 16)
    try:
        _get_embedding_quota().reserve(estimated)
    except QuotaExceeded as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except QuotaUnavailable as exc:
        raise HTTPException(
            status_code=503,
            detail="embedding quota unavailable",
        ) from exc


@app.post("/api/v1/search", response_model=SearchResponse)
def search(http_request: Request, payload: SearchRequest) -> SearchResponse:
    """Dense retrieval over the current index; no generation."""
    _guard_http(http_request, "search")
    _reserve_embedding(payload.query)
    try:
        retriever = _get_retriever()
        hits: list[SearchHit] = retriever.search(
            payload.query,
            limit=payload.limit,
            doc_type=payload.doc_type,
            docs_version=payload.docs_version,
            include_deprecated=payload.include_deprecated,
            include_meta=payload.include_meta,
            include_reference=payload.include_reference,
        )
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - dependency failures become 503
        raise HTTPException(
            status_code=503,
            detail="retrieval service unavailable",
        ) from exc

    return SearchResponse(
        query=payload.query,
        count=len(hits),
        results=[
            SearchResult.model_validate(hit, from_attributes=True) for hit in hits
        ],
    )


@app.post("/api/v1/chat", response_model=ChatResponse)
def chat(http_request: Request, payload: ChatRequest) -> ChatResponse:
    """Retrieve context, generate a grounded answer, and return cited sources."""
    _guard_http(http_request, "chat")
    _reserve_embedding(payload.query)
    try:
        retriever = _get_retriever()
        hits: list[SearchHit] = retriever.search(
            payload.query,
            limit=payload.limit,
            doc_type=payload.doc_type,
            docs_version=payload.docs_version,
            include_deprecated=payload.include_deprecated,
            include_meta=payload.include_meta,
            include_reference=payload.include_reference,
        )
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - dependency failures become 503
        raise HTTPException(
            status_code=503,
            detail="retrieval service unavailable",
        ) from exc

    settings = get_settings()
    if not hits:
        return ChatResponse(
            query=payload.query,
            answer=REFUSAL,
            model=settings.llm_model,
            sources=[],
        )

    messages = build_messages(payload.query, hits)
    max_output = min(settings.llm_max_tokens, settings.llm_max_tokens_cap)
    try:
        _get_quota_guard().reserve(
            estimate_chat_tokens(messages, max_output)
        )
    except QuotaExceeded as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except QuotaUnavailable as exc:
        raise HTTPException(
            status_code=503,
            detail="quota service unavailable",
        ) from exc

    try:
        client = _get_chat_client()
        answer = client.complete(messages)
    except Exception as exc:  # noqa: BLE001 - provider failures become 503
        raise HTTPException(
            status_code=503,
            detail="generation service unavailable",
        ) from exc

    return ChatResponse(
        query=payload.query,
        answer=answer,
        model=client.model,
        sources=_sources_for_hits(hits),
    )


@app.post("/api/v1/chat/stream")
def chat_stream(http_request: Request, payload: ChatRequest) -> StreamingResponse:
    """SSE version of /api/v1/chat: sources first, then token chunks."""
    _guard_http(http_request, "chat")
    _reserve_embedding(payload.query)
    try:
        retriever = _get_retriever()
        hits: list[SearchHit] = retriever.search(
            payload.query,
            limit=payload.limit,
            doc_type=payload.doc_type,
            docs_version=payload.docs_version,
            include_deprecated=payload.include_deprecated,
            include_meta=payload.include_meta,
            include_reference=payload.include_reference,
        )
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - dependency failures become 503
        raise HTTPException(
            status_code=503,
            detail="retrieval service unavailable",
        ) from exc

    settings = get_settings()
    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }
    if not hits:
        def refusal_stream() -> Iterator[str]:
            yield _sse("sources", {"sources": []})
            yield _sse("token", {"text": REFUSAL})
            yield _sse("usage", {"input_tokens": None, "output_tokens": None})
            yield _sse("done", {"model": settings.llm_model})

        return StreamingResponse(
            refusal_stream(),
            media_type="text/event-stream",
            headers=headers,
        )

    messages = build_messages(payload.query, hits)
    max_output = min(settings.llm_max_tokens, settings.llm_max_tokens_cap)
    try:
        _get_quota_guard().reserve(
            estimate_chat_tokens(messages, max_output)
        )
    except QuotaExceeded as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except QuotaUnavailable as exc:
        raise HTTPException(
            status_code=503,
            detail="quota service unavailable",
        ) from exc

    try:
        client = _get_chat_client()
    except Exception as exc:  # noqa: BLE001 - provider failures become 503
        raise HTTPException(
            status_code=503,
            detail="generation service unavailable",
        ) from exc

    sources = _sources_for_hits(hits)

    def token_stream() -> Iterator[str]:
        yield _sse(
            "sources",
            {"sources": [source.model_dump() for source in sources]},
        )
        try:
            for chunk in client.stream(messages):
                yield _sse("token", {"text": chunk})
        except Exception:  # noqa: BLE001 - errors must be explicit in SSE
            yield _sse("error", {"message": "generation stream failed"})
            return
        # The OpenAI-compatible stream does not expose usage in all providers;
        # keep the event for the fixed SSE contract, with null counts.
        yield _sse("usage", {"input_tokens": None, "output_tokens": None})
        yield _sse("done", {"model": client.model})

    return StreamingResponse(
        token_stream(),
        media_type="text/event-stream",
        headers=headers,
    )


@app.get("/readyz")
async def readyz() -> JSONResponse:
    """Readiness probe: config, Qdrant and Redis."""
    settings = get_settings()
    missing = settings.missing_required()
    qdrant, redis_check = await asyncio.gather(
        _check_qdrant(settings),
        _check_redis(settings),
    )
    checks: dict[str, dict[str, Any]] = {
        "config": {"ok": not missing, "missing": missing},
        "qdrant": qdrant,
        "redis": redis_check,
    }
    ok = all(check.get("ok") for check in checks.values())
    return JSONResponse(
        status_code=200 if ok else 503,
        content={"status": "ok" if ok else "degraded", "checks": checks},
    )
