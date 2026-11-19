"""Tests for Git-diff change detection and manifest history."""

import subprocess
from pathlib import Path

from fastapi_docs_rag.indexing.manifest import IndexManifest, save_manifest_history
from fastapi_docs_rag.ingestion.change_detector import git_changed_docs


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        check=True,
        text=True,
    )
    return result.stdout.strip()


def test_git_changed_docs_returns_modified_page(tmp_path: Path) -> None:
    repo = tmp_path / "docs-repo"
    docs = repo / "docs" / "en" / "docs"
    docs.mkdir(parents=True)
    page = docs / "sample.md"
    page.write_text("# Sample\n\nold\n", encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "initial")
    previous = _git(repo, "rev-parse", "HEAD")

    page.write_text("# Sample\n\nnew\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "change")
    current = _git(repo, "rev-parse", "HEAD")

    changed = git_changed_docs(
        repo,
        f"git:{previous}",
        f"git:{current}",
    )

    assert changed == ["docs/en/docs/sample.md"]


def test_git_changed_docs_is_none_for_plain_snapshot(tmp_path: Path) -> None:
    assert git_changed_docs(tmp_path, "git:abc", "git:def") is None


def test_manifest_history_is_saved(tmp_path: Path) -> None:
    manifest = IndexManifest(
        build_id="20260101T000000Z",
        build_collection="build-20260101",
        source_ref="git:abc",
        docs_version="0.162.0",
        embedding_model="fake",
        embedding_dim=4,
        created_at="2026-01-01T00:00:00+00:00",
        previous_build_id=None,
        previous_source_ref=None,
        documents={"a": "b"},
        chunks={"c": "d"},
    )
    live = tmp_path / "index_manifest.json"

    history = save_manifest_history(manifest, live)

    assert history.exists()
    assert IndexManifest.load(history) == manifest
