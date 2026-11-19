"""Git-diff change detection for the documentation source.

The content-hash manifest remains the source of truth; this module adds a
commit-level view when the checkout actually has its own ``.git``. A plain file
snapshot returns ``None`` instead of walking up into the project repository.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def _git_root(repo_root: Path) -> Path | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--show-toplevel"],
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    root = Path(result.stdout.strip()).resolve()
    return root if root == repo_root.resolve() else None


def git_changed_docs(
    repo_root: Path,
    previous_ref: str | None,
    current_ref: str | None,
) -> list[str] | None:
    """Return changed docs paths between two git refs, or None without git."""
    if not previous_ref or not current_ref:
        return None
    if not previous_ref.startswith("git:") or not current_ref.startswith("git:"):
        return None
    if _git_root(repo_root) is None:
        return None
    previous = previous_ref.removeprefix("git:")
    current = current_ref.removeprefix("git:")
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "diff",
                "--name-only",
                f"{previous}..{current}",
                "--",
                "docs/en/docs",
                "docs_src",
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]
