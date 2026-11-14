"""Smoke-test the configured embedding and chat providers.

This does not print secrets. Run it after filling `.env`:

    uv run --no-sync python scripts/check_providers.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langchain_rag.config import Settings, get_settings  # noqa: E402
from langchain_rag.indexing.embedder import make_embedder  # noqa: E402


def _chat_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def check_embedding(settings: Settings) -> bool:
    try:
        embedder = make_embedder(settings)
        vectors = embedder.embed(["FastAPI dependency injection"])
    except Exception as exc:  # noqa: BLE001 - report provider failures
        print(f"[embedding] FAILED: {type(exc).__name__}: {str(exc)[:300]}")
        return False
    finally:
        close = getattr(locals().get("embedder"), "close", None)
        if close is not None:
            close()

    print(
        f"[embedding] OK model={settings.embedding_model} "
        f"dim={len(vectors[0])}"
    )
    return True


def check_chat(settings: Settings) -> bool:
    if not settings.llm_api_key:
        print("[chat] SKIPPED: LLM_API_KEY is empty")
        return False
    payload = {
        "model": settings.llm_model,
        "messages": [
            {"role": "user", "content": "Reply with exactly: OK"}
        ],
        "temperature": 0,
        "max_tokens": 16,
    }
    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                _chat_url(settings.llm_base_url),
                headers={"Authorization": f"Bearer {settings.llm_api_key}"},
                json=payload,
            )
    except httpx.HTTPError as exc:
        print(f"[chat] FAILED: {type(exc).__name__}: {str(exc)[:300]}")
        return False

    if response.status_code != 200:
        print(f"[chat] FAILED: HTTP {response.status_code}: {response.text[:300]}")
        return False
    try:
        content = response.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        print(f"[chat] FAILED: unexpected response: {exc}")
        return False
    print(f"[chat] OK model={settings.llm_model} reply={content[:80]!r}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-only", action="store_true")
    parser.add_argument("--chat-only", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    print(
        f"embedding={settings.embedding_model} "
        f"chat={settings.llm_model} "
        f"missing={settings.missing_required()}"
    )

    ok = True
    if not args.chat_only:
        ok = check_embedding(settings) and ok
    if not args.embedding_only:
        ok = check_chat(settings) and ok
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
