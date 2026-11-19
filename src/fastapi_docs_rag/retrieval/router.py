"""Lightweight query router for version-aware retrieval."""

from __future__ import annotations

import re

VERSION_RE = re.compile(r"\b\d+\.\d+(?:\.\d+)?\b")
VERSION_TERMS = (
    "版本",
    "变更",
    "更新",
    "废弃",
    "弃用",
    "release",
    "changelog",
    "deprecated",
)


def is_version_query(query: str) -> bool:
    if VERSION_RE.search(query):
        return True
    lowered = query.lower()
    return any(term in lowered for term in VERSION_TERMS)


def route_doc_type(query: str, requested: str | None) -> str | None:
    """Route version questions to release notes unless the caller overrides."""
    if requested:
        return requested
    return "release_notes" if is_version_query(query) else None
