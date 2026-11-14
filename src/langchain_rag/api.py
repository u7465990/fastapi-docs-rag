"""FastAPI application entry point."""

from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

from langchain_rag import __version__
from langchain_rag.config import Settings, get_settings
from langchain_rag.generation.chat import REFUSAL, generate_answer
from langchain_rag.generation.llm import ChatClient, make_chat_client
from langchain_rag.indexing.embedder import make_embedder
from langchain_rag.retrieval.dense import DenseRetriever, SearchHit

_settings = get_settings()

app = FastAPI(
    title="FastAPI Docs RAG",
    version=__version__,
    description="Version-aware RAG over FastAPI documentation.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.cors_origins,
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


@app.post("/api/v1/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    """Dense retrieval over the current index; no generation."""
    try:
        retriever = _get_retriever()
        hits: list[SearchHit] = retriever.search(
            request.query,
            limit=request.limit,
            doc_type=request.doc_type,
            docs_version=request.docs_version,
            include_deprecated=request.include_deprecated,
            include_meta=request.include_meta,
        )
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - dependency failures become 503
        raise HTTPException(
            status_code=503,
            detail="retrieval service unavailable",
        ) from exc

    return SearchResponse(
        query=request.query,
        count=len(hits),
        results=[
            SearchResult.model_validate(hit, from_attributes=True) for hit in hits
        ],
    )


@app.post("/api/v1/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Retrieve context, generate a grounded answer, and return cited sources."""
    try:
        retriever = _get_retriever()
        hits: list[SearchHit] = retriever.search(
            request.query,
            limit=request.limit,
            doc_type=request.doc_type,
            docs_version=request.docs_version,
            include_deprecated=request.include_deprecated,
            include_meta=request.include_meta,
        )
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - dependency failures become 503
        raise HTTPException(
            status_code=503,
            detail="retrieval service unavailable",
        ) from exc

    if not hits:
        return ChatResponse(
            query=request.query,
            answer=REFUSAL,
            model=get_settings().llm_model,
            sources=[],
        )

    try:
        client = _get_chat_client()
        answer = generate_answer(request.query, hits, client)
    except Exception as exc:  # noqa: BLE001 - provider failures become 503
        raise HTTPException(
            status_code=503,
            detail="generation service unavailable",
        ) from exc

    sources = [
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
    return ChatResponse(
        query=request.query,
        answer=answer,
        model=client.model,
        sources=sources,
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
