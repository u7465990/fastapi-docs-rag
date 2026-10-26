"""FastAPI application entry point."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from langchain_rag import __version__
from langchain_rag.config import Settings, get_settings

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
