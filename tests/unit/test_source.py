"""Tests for source metadata detection."""

import subprocess
from pathlib import Path

from langchain_rag.ingestion.source import detect_source_info


def test_detects_release_notes_version(tmp_path: Path) -> None:
    docs = tmp_path / "docs" / "en" / "docs"
    docs.mkdir(parents=True)
    (docs / "release-notes.md").write_text(
        "# Release Notes\n\n## 0.162.0 (2026-10-23)\n\n### Features\n\n- Something.\n",
        encoding="utf-8",
    )

    source = detect_source_info(tmp_path)

    assert source.docs_version == "0.162.0"
    assert source.source_ref == "unknown"
    assert source.fetched_at


def test_missing_release_notes_uses_fallback(tmp_path: Path) -> None:
    source = detect_source_info(tmp_path, default_version="unknown")

    assert source.docs_version == "unknown"
    assert source.source_ref == "unknown"


def test_source_ref_ignores_parent_repository(tmp_path: Path) -> None:
    snapshot = tmp_path / "data" / "raw" / "fastapi"
    snapshot.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)

    source = detect_source_info(snapshot)

    # Without its own .git, the snapshot must not inherit the project repo HEAD.
    assert source.source_ref == "unknown"
