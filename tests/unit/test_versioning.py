"""Tests for historical version-to-collection routing."""

from fastapi_docs_rag.config import get_settings
from fastapi_docs_rag.retrieval.versioning import select_collection


def test_known_historical_version_selects_collection() -> None:
    collection, version = select_collection(
        "FastAPI 0.100.0 对 on_startup 做了什么？",
        None,
        get_settings(),
    )

    assert collection == "fastapi_docs_v0100"
    assert version == "0.100.0"


def test_explicit_version_selects_collection() -> None:
    collection, version = select_collection(
        "on_startup 有什么变更？",
        "0.100.0",
        get_settings(),
    )

    assert collection == "fastapi_docs_v0100"
    assert version == "0.100.0"


def test_unknown_version_falls_back_to_current() -> None:
    settings = get_settings()
    collection, version = select_collection(
        "FastAPI 9.9.9 有什么变更？",
        None,
        settings,
    )

    assert collection == settings.qdrant_collection
    assert version is None
