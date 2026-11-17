"""Detect the documentation source snapshot used by the index."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

RELEASE_HEADING_RE = re.compile(r"(?m)^##\s+(\d+\.\d+(?:\.\d+)?)\b")


@dataclass(frozen=True)
class SourceInfo:
    """Version fields written into every Qdrant payload."""

    docs_version: str
    source_ref: str
    fetched_at: str


def detect_source_info(
    repo_root: Path,
    *,
    default_version: str = "unknown",
    fallback_source_ref: str = "",
) -> SourceInfo:
    """Read release-notes version and git HEAD, with a snapshot fallback."""
    git_ref = _git_head(repo_root)
    if git_ref == "unknown" and fallback_source_ref:
        git_ref = (
            fallback_source_ref
            if fallback_source_ref.startswith(("git:", "snapshot:"))
            else f"git:{fallback_source_ref}"
        )
    return SourceInfo(
        docs_version=_release_notes_version(repo_root) or default_version,
        source_ref=git_ref,
        fetched_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )


def _release_notes_version(repo_root: Path) -> str | None:
    path = repo_root / "docs" / "en" / "docs" / "release-notes.md"
    if not path.exists():
        return None
    match = RELEASE_HEADING_RE.search(path.read_text(encoding="utf-8"))
    return match.group(1) if match else None


def _git_head(repo_root: Path) -> str:
    """Return the commit only when *repo_root* is its own git worktree.

    ``data/raw/fastapi`` is a standalone checkout. If it is a plain file
    snapshot, ``git -C`` would otherwise walk up and return the parent project
    repository's HEAD, which is not the documentation snapshot.
    """
    try:
        top = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--show-toplevel"],
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
        if top.returncode != 0:
            return "unknown"
        toplevel = Path(top.stdout.strip()).resolve()
        if toplevel != repo_root.resolve():
            return "unknown"
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    commit = result.stdout.strip()
    if result.returncode != 0 or not commit:
        return "unknown"
    return f"git:{commit}"
