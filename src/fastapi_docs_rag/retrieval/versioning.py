"""Version-to-collection registry for historical documentation."""

from __future__ import annotations

import re

from fastapi_docs_rag.config import Settings

VERSION_RE = re.compile(r"\b\d+\.\d+(?:\.\d+)?\b")

# Only versions with a real materialized collection belong here. Do not add a
# version until its collection has been built and validated.
HISTORICAL_COLLECTIONS: dict[str, str] = {
    "0.100.0": "fastapi_docs_v0100",
}


def requested_version(
    query: str,
    explicit: str | None = None,
) -> str | None:
    if explicit:
        return explicit
    match = VERSION_RE.search(query)
    return match.group(0) if match else None


def select_collection(
    query: str,
    explicit_version: str | None,
    settings: Settings,
) -> tuple[str, str | None]:
    """Return (collection, historical version) for this request."""
    version = requested_version(query, explicit_version)
    if version and version in HISTORICAL_COLLECTIONS:
        return HISTORICAL_COLLECTIONS[version], version
    return settings.qdrant_collection, None
