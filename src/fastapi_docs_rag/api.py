"""FastAPI application entry point."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from functools import lru_cache
from typing import Any

import anyio
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

from fastapi_docs_rag import __version__
from fastapi_docs_rag.config import Settings, get_settings
from fastapi_docs_rag.generation.cache import (
    AnswerCache,
    index_identity,
    make_answer_cache,
    make_cache_key,
)
from fastapi_docs_rag.generation.chat import (
    REFUSAL,
    build_messages,
    check_citations,
    estimate_chat_tokens,
    retrieval_confidence,
    rewrite_query,
    select_context_hits,
)
from fastapi_docs_rag.generation.llm import (
    AsyncChatClient,
    make_async_chat_client,
)
from fastapi_docs_rag.generation.quota import (
    DailyQuota,
    QuotaExceeded,
    QuotaUnavailable,
    make_daily_quota,
)
from fastapi_docs_rag.generation.session import (
    SessionStore,
    SessionUnavailable,
    make_session_store,
)
from fastapi_docs_rag.indexing.embedder import make_embedder
from fastapi_docs_rag.observability.metrics import (
    log_request,
    metrics,
    request_id_from_header,
    reset_request_id,
    set_request_id,
)
from fastapi_docs_rag.retrieval.bm25 import BM25Index
from fastapi_docs_rag.retrieval.cache import (
    RetrievalCache,
    make_retrieval_cache,
    make_retrieval_key,
)
from fastapi_docs_rag.retrieval.dense import DenseRetriever, SearchHit
from fastapi_docs_rag.retrieval.hybrid import HybridRetriever
from fastapi_docs_rag.retrieval.rerank import (
    CrossEncoderReranker,
    LexicalReranker,
    Reranker,
)
from fastapi_docs_rag.retrieval.router import route_doc_type
from fastapi_docs_rag.security import (
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


@app.middleware("http")
async def observe_http(request: Request, call_next):
    request_id = request_id_from_header(request.headers.get("x-request-id"))
    token = set_request_id(request_id)
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        metrics.record_request(
            request.method,
            request.url.path,
            status,
            duration_ms,
        )
        log_request(
            method=request.method,
            path=request.url.path,
            status=status,
            duration_ms=duration_ms,
            client=request.client.host if request.client else "unknown",
        )
        reset_request_id(token)


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


@app.get("/metrics", include_in_schema=False)
def metrics_view() -> PlainTextResponse:
    return PlainTextResponse(
        metrics.render(),
        media_type="text/plain; version=0.0.4",
    )


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
    dense_score: float | None = None
    rerank_score: float | None = None


class SearchResponse(BaseModel):
    query: str
    count: int
    results: list[SearchResult]
    timings: dict[str, float] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=6, ge=1, le=10)
    doc_type: str | None = None
    docs_version: str | None = None
    include_deprecated: bool = True
    include_meta: bool = False
    include_reference: bool = False
    session_id: str | None = Field(default=None, max_length=64)


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
    score: float | None = None
    dense_score: float | None = None
    rerank_score: float | None = None


class ChatResponse(BaseModel):
    query: str
    answer: str
    model: str
    sources: list[ChatSource]
    session_id: str | None = None
    citation_warnings: list[str] = Field(default_factory=list)
    timings: dict[str, float] = Field(default_factory=dict)
    retrieval_query: str | None = None


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
            score=hit.score,
            dense_score=hit.dense_score,
            rerank_score=hit.rerank_score,
        )
        for index, hit in enumerate(hits, start=1)
    ]


@lru_cache(maxsize=1)
def _get_retriever() -> DenseRetriever | HybridRetriever:
    settings = get_settings()
    embedder = make_embedder(settings)
    client = QdrantClient(url=settings.qdrant_url)
    dense = DenseRetriever(
        client,
        embedder,
        collection=settings.qdrant_collection,
    )
    if settings.retrieval_mode.lower() != "hybrid":
        return dense
    bm25 = BM25Index.from_qdrant(client, settings.qdrant_collection)
    reranker: Reranker | None = None
    if settings.rerank_enabled:
        provider = settings.reranker_provider.lower()
        if provider in {"fastembed", "cross_encoder", "cross-encoder"}:
            reranker = CrossEncoderReranker(
                settings.reranker_model,
                cache_dir=settings.reranker_cache_dir,
                threads=settings.reranker_threads,
            )
        else:
            reranker = LexicalReranker()
    return HybridRetriever(
        dense,
        bm25,
        dense_top_k=settings.hybrid_dense_top_k,
        bm25_top_k=settings.hybrid_bm25_top_k,
        rrf_k=settings.hybrid_rrf_k,
        dense_weight=settings.hybrid_dense_weight,
        bm25_weight=settings.hybrid_bm25_weight,
        reranker=reranker,
        rerank_top_k=settings.rerank_top_k,
        rerank_skip_score=settings.rerank_skip_score,
    )


@lru_cache(maxsize=1)
def _get_chat_client() -> AsyncChatClient:
    return make_async_chat_client(get_settings())


_SETTINGS = get_settings()
_LLM_SEMAPHORE = asyncio.Semaphore(_SETTINGS.llm_max_concurrency)
_RETRIEVAL_SEMAPHORE = asyncio.Semaphore(_SETTINGS.retrieval_max_concurrency)


def _retrieval_hits(
    payload: ChatRequest | SearchRequest,
    history: list[dict[str, str]] | None = None,
) -> tuple[list[SearchHit], dict[str, float]]:
    """Blocking retrieval, executed in a worker thread by async routes."""
    settings = get_settings()
    started = time.perf_counter()
    timings: dict[str, float] = {}
    query = payload.query
    if isinstance(payload, ChatRequest):
        query = rewrite_query(
            payload.query,
            history if history is not None else _load_history(payload.session_id),
        )
    doc_type = route_doc_type(payload.query, payload.doc_type)
    cache_key = None
    if settings.retrieval_cache_enabled:
        cache_key = make_retrieval_key(
            query=query,
            limit=payload.limit,
            doc_type=doc_type,
            docs_version=payload.docs_version,
            include_deprecated=payload.include_deprecated,
            include_meta=payload.include_meta,
            include_reference=payload.include_reference,
            identity=index_identity(),
            settings=settings,
        )
        cached = _get_retrieval_cache().get(cache_key)
        if cached is not None:
            timings["retrieval_cache_hit"] = 1.0
            timings["retrieval_ms"] = (time.perf_counter() - started) * 1000
            return cached, timings

    setup_started = time.perf_counter()
    retriever = _get_retriever()
    timings["setup_ms"] = (time.perf_counter() - setup_started) * 1000
    hits = retriever.search(
        query,
        limit=payload.limit,
        doc_type=doc_type,
        docs_version=payload.docs_version,
        include_deprecated=payload.include_deprecated,
        include_meta=payload.include_meta,
        include_reference=payload.include_reference,
        timings=timings,
    )
    if isinstance(payload, ChatRequest):
        hits = select_context_hits(hits, max_tokens=settings.max_context_tokens)
    timings["retrieval_ms"] = (time.perf_counter() - started) * 1000
    if cache_key is not None:
        _get_retrieval_cache().set(cache_key, hits)
    return hits, timings


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


@lru_cache(maxsize=1)
def _get_session_store() -> SessionStore:
    return make_session_store(get_settings())


@lru_cache(maxsize=1)
def _get_answer_cache() -> AnswerCache:
    return make_answer_cache(get_settings())


@lru_cache(maxsize=1)
def _get_retrieval_cache() -> RetrievalCache:
    return make_retrieval_cache(get_settings())


def _cache_options(payload: ChatRequest) -> dict[str, Any]:
    return {
        "limit": payload.limit,
        "doc_type": route_doc_type(payload.query, payload.doc_type),
        "docs_version": payload.docs_version,
        "include_deprecated": payload.include_deprecated,
        "include_meta": payload.include_meta,
        "include_reference": payload.include_reference,
    }


def _cache_key_for(
    payload: ChatRequest,
    history: list[dict[str, str]],
) -> str:
    return make_cache_key(
        query=payload.query,
        history=history,
        identity=index_identity(),
        settings=get_settings(),
        options=_cache_options(payload),
    )


def _cache_lookup(
    payload: ChatRequest,
    history: list[dict[str, str]],
) -> tuple[str, dict[str, Any]] | None:
    settings = get_settings()
    if not settings.cache_enabled:
        return None
    key = _cache_key_for(payload, history)
    cached = _get_answer_cache().get(key)
    return (key, cached) if cached is not None else None


def _cached_response(
    payload: ChatRequest,
    cached: dict[str, Any],
) -> ChatResponse:
    return ChatResponse(
        query=payload.query,
        answer=str(cached.get("answer", "")),
        model=str(cached.get("model", "")),
        sources=[
            ChatSource.model_validate(source)
            for source in cached.get("sources", [])
        ],
        session_id=payload.session_id,
        citation_warnings=list(cached.get("citation_warnings", [])),
        timings={"answer_cache_hit": 1.0},
    )


def _cache_store(
    key: str,
    answer: str,
    model: str,
    sources: list[ChatSource],
    warnings: list[str],
) -> None:
    _get_answer_cache().set(
        key,
        {
            "answer": answer,
            "model": model,
            "sources": [source.model_dump() for source in sources],
            "citation_warnings": warnings,
        },
    )


def _load_history(session_id: str | None) -> list[dict[str, str]]:
    if not session_id:
        return []
    try:
        return _get_session_store().load(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except SessionUnavailable as exc:
        raise HTTPException(
            status_code=503,
            detail="session store unavailable",
        ) from exc


def _save_turn(session_id: str | None, question: str, answer: str) -> None:
    if not session_id:
        return
    try:
        _get_session_store().append_turn(session_id, question, answer)
    except (SessionUnavailable, ValueError):
        # The answer is still useful; do not fail the response if history
        # persistence has a transient problem.
        pass


def _guard_http(http_request: Request, scope: str) -> None:
    settings = get_settings()
    check_access_token(http_request, settings.api_access_token)
    try:
        _get_rate_limiter().check(scope, client_identifier(http_request))
    except RateLimitExceeded as exc:
        metrics.record_quota_rejection("rate_limit")
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
        metrics.record_quota_rejection("embedding")
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except QuotaUnavailable as exc:
        raise HTTPException(
            status_code=503,
            detail="embedding quota unavailable",
        ) from exc
    metrics.record_tokens("embedding", estimated)


def _reserve_llm(tokens: int) -> None:
    _get_quota_guard().reserve(tokens)


def _retrieval_is_confident(hits: list[SearchHit]) -> bool:
    if not hits:
        return False
    settings = get_settings()
    threshold = (
        settings.rerank_min_score
        if any(hit.rerank_score is not None for hit in hits)
        else settings.retrieval_min_score
    )
    return retrieval_confidence(hits) >= threshold


@app.post("/api/v1/search", response_model=SearchResponse)
async def search(http_request: Request, payload: SearchRequest) -> SearchResponse:
    """Async retrieval over the current index; no generation."""
    await anyio.to_thread.run_sync(_guard_http, http_request, "search")
    await anyio.to_thread.run_sync(_reserve_embedding, payload.query)
    try:
        async with _RETRIEVAL_SEMAPHORE:
            hits, timings = await anyio.to_thread.run_sync(_retrieval_hits, payload)
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
        timings=timings,
    )


@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(http_request: Request, payload: ChatRequest) -> ChatResponse:
    """Retrieve context, generate a grounded answer, and return cited sources."""
    started = time.perf_counter()
    await anyio.to_thread.run_sync(_guard_http, http_request, "chat")
    history = await anyio.to_thread.run_sync(_load_history, payload.session_id)
    retrieval_query = rewrite_query(payload.query, history)
    cached = await anyio.to_thread.run_sync(_cache_lookup, payload, history)
    if cached is not None:
        return _cached_response(payload, cached[1])
    await anyio.to_thread.run_sync(_reserve_embedding, payload.query)
    cache_key = _cache_key_for(payload, history) if get_settings().cache_enabled else None
    try:
        async with _RETRIEVAL_SEMAPHORE:
            hits, timings = await anyio.to_thread.run_sync(
                _retrieval_hits,
                payload,
                history,
            )
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - dependency failures become 503
        raise HTTPException(
            status_code=503,
            detail="retrieval service unavailable",
        ) from exc

    settings = get_settings()
    if not _retrieval_is_confident(hits):
        hits = []
    if not hits:
        _save_turn(payload.session_id, payload.query, REFUSAL)
        return ChatResponse(
            query=payload.query,
            answer=REFUSAL,
            model=settings.llm_model,
            sources=[],
            session_id=payload.session_id,
            timings={**timings, "total_ms": (time.perf_counter() - started) * 1000},
            retrieval_query=retrieval_query,
        )

    messages = build_messages(payload.query, hits, history)
    max_output = min(settings.llm_max_tokens, settings.llm_max_tokens_cap)
    estimated_llm = estimate_chat_tokens(messages, max_output)
    try:
        await anyio.to_thread.run_sync(_reserve_llm, estimated_llm)
    except QuotaExceeded as exc:
        metrics.record_quota_rejection("llm")
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except QuotaUnavailable as exc:
        raise HTTPException(
            status_code=503,
            detail="quota service unavailable",
        ) from exc
    metrics.record_tokens("llm", estimated_llm)

    try:
        client = _get_chat_client()
        llm_started = time.perf_counter()
        async with _LLM_SEMAPHORE:
            answer = await client.complete(messages)
            citation_check = check_citations(answer, hits)
            if citation_check.citations == 0 or citation_check.invalid_indices:
                retry_messages = [
                    *messages,
                    {"role": "assistant", "content": answer},
                    {
                        "role": "user",
                        "content": (
                            "上一次回答缺少有效引用。请重新回答，并为每个关键结论"
                            "标注本轮上下文中的 [n]；不要编造编号。"
                        ),
                    },
                ]
                retry_answer = await client.complete(retry_messages)
                retry_check = check_citations(retry_answer, hits)
                if retry_check.citations > citation_check.citations:
                    answer = retry_answer
                    citation_check = retry_check
    except Exception as exc:  # noqa: BLE001 - provider failures become 503
        raise HTTPException(
            status_code=503,
            detail="generation service unavailable",
        ) from exc
    timings["llm_ms"] = (time.perf_counter() - llm_started) * 1000

    await anyio.to_thread.run_sync(
        _save_turn,
        payload.session_id,
        payload.query,
        answer,
    )
    sources = _sources_for_hits(hits)
    if cache_key is not None:
        await anyio.to_thread.run_sync(
            _cache_store,
            cache_key,
            answer,
            client.model,
            sources,
            citation_check.warnings,
        )
    return ChatResponse(
        query=payload.query,
        answer=answer,
        model=client.model,
        sources=sources,
        session_id=payload.session_id,
        citation_warnings=citation_check.warnings,
        timings={**timings, "total_ms": (time.perf_counter() - started) * 1000},
        retrieval_query=retrieval_query,
    )


@app.post("/api/v1/chat/stream")
async def chat_stream(
    http_request: Request,
    payload: ChatRequest,
) -> StreamingResponse:
    """Async SSE: sources first, then token chunks."""
    started = time.perf_counter()
    await anyio.to_thread.run_sync(_guard_http, http_request, "chat")
    history = await anyio.to_thread.run_sync(_load_history, payload.session_id)
    retrieval_query = rewrite_query(payload.query, history)
    cached = await anyio.to_thread.run_sync(_cache_lookup, payload, history)
    if cached is not None:
        cached_data = cached[1]
        cached_sources = [
            ChatSource.model_validate(source)
            for source in cached_data.get("sources", [])
        ]
        cached_warnings = list(cached_data.get("citation_warnings", []))

        async def cached_stream() -> AsyncIterator[str]:
            yield _sse(
                "sources",
                {"sources": [source.model_dump() for source in cached_sources]},
            )
            yield _sse("token", {"text": str(cached_data.get("answer", ""))})
            for warning in cached_warnings:
                yield _sse("warning", {"message": warning})
            yield _sse("usage", {"input_tokens": None, "output_tokens": None})
            yield _sse(
                "done",
                {
                    "model": str(cached_data.get("model", "")),
                    "session_id": payload.session_id,
                    "citation_warnings": cached_warnings,
                    "cache_hit": True,
                    "timings": {"answer_cache_hit": 1.0},
                    "retrieval_query": retrieval_query,
                },
            )

        return StreamingResponse(
            cached_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    await anyio.to_thread.run_sync(_reserve_embedding, payload.query)
    cache_key = _cache_key_for(payload, history) if get_settings().cache_enabled else None
    try:
        async with _RETRIEVAL_SEMAPHORE:
            hits, timings = await anyio.to_thread.run_sync(
                _retrieval_hits,
                payload,
                history,
            )
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - dependency failures become 503
        raise HTTPException(
            status_code=503,
            detail="retrieval service unavailable",
        ) from exc

    settings = get_settings()
    if not _retrieval_is_confident(hits):
        hits = []
    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }
    if not hits:
        await anyio.to_thread.run_sync(
            _save_turn,
            payload.session_id,
            payload.query,
            REFUSAL,
        )

        async def refusal_stream() -> AsyncIterator[str]:
            yield _sse("sources", {"sources": []})
            yield _sse("token", {"text": REFUSAL})
            yield _sse("usage", {"input_tokens": None, "output_tokens": None})
            yield _sse(
                "done",
                {
                    "model": settings.llm_model,
                    "session_id": payload.session_id,
                    "timings": {
                        **timings,
                        "total_ms": (time.perf_counter() - started) * 1000,
                    },
                    "retrieval_query": retrieval_query,
                },
            )

        return StreamingResponse(
            refusal_stream(),
            media_type="text/event-stream",
            headers=headers,
        )

    messages = build_messages(payload.query, hits, history)
    max_output = min(settings.llm_max_tokens, settings.llm_max_tokens_cap)
    estimated_llm = estimate_chat_tokens(messages, max_output)
    try:
        await anyio.to_thread.run_sync(_reserve_llm, estimated_llm)
    except QuotaExceeded as exc:
        metrics.record_quota_rejection("llm")
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except QuotaUnavailable as exc:
        raise HTTPException(
            status_code=503,
            detail="quota service unavailable",
        ) from exc
    metrics.record_tokens("llm", estimated_llm)

    try:
        client = _get_chat_client()
    except Exception as exc:  # noqa: BLE001 - provider failures become 503
        raise HTTPException(
            status_code=503,
            detail="generation service unavailable",
        ) from exc

    sources = _sources_for_hits(hits)

    async def token_stream() -> AsyncIterator[str]:
        yield _sse(
            "sources",
            {"sources": [source.model_dump() for source in sources]},
        )
        parts: list[str] = []
        llm_started = time.perf_counter()
        try:
            async with _LLM_SEMAPHORE:
                async for chunk in client.stream(messages):
                    parts.append(chunk)
                    yield _sse("token", {"text": chunk})
        except Exception:  # noqa: BLE001 - errors must be explicit in SSE
            yield _sse("error", {"message": "generation stream failed"})
            return
        answer = "".join(parts)
        await anyio.to_thread.run_sync(
            _save_turn,
            payload.session_id,
            payload.query,
            answer,
        )
        citation_check = check_citations(answer, hits)
        if cache_key is not None:
            await anyio.to_thread.run_sync(
                _cache_store,
                cache_key,
                answer,
                client.model,
                sources,
                citation_check.warnings,
            )
        for warning in citation_check.warnings:
            yield _sse("warning", {"message": warning})
        # The OpenAI-compatible stream does not expose usage in all providers;
        # keep the event for the fixed SSE contract, with null counts.
        yield _sse("usage", {"input_tokens": None, "output_tokens": None})
        timings["llm_ms"] = (time.perf_counter() - llm_started) * 1000
        yield _sse(
            "done",
            {
                "model": client.model,
                "session_id": payload.session_id,
                "citation_warnings": citation_check.warnings,
                "timings": {
                    **timings,
                    "total_ms": (time.perf_counter() - started) * 1000,
                },
                "retrieval_query": retrieval_query,
            },
        )

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
        "llm": {
            "ok": bool(settings.llm_api_key and settings.llm_model),
            "model": settings.llm_model,
            # A live completion probe would spend quota on every readiness
            # check; this validates configuration only.
            "mode": "config_only",
        },
    }
    ok = all(check.get("ok") for check in checks.values())
    return JSONResponse(
        status_code=200 if ok else 503,
        content={"status": "ok" if ok else "degraded", "checks": checks},
    )
