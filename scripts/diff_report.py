"""Report document/chunk changes and index staleness."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi_docs_rag.config import get_settings  # noqa: E402
from fastapi_docs_rag.indexing.chunker import chunk_documents  # noqa: E402
from fastapi_docs_rag.indexing.manifest import (  # noqa: E402
    IndexManifest,
    diff_manifest,
)
from fastapi_docs_rag.ingestion.change_detector import git_changed_docs  # noqa: E402
from fastapi_docs_rag.ingestion.parser import iter_documents  # noqa: E402
from fastapi_docs_rag.ingestion.source import detect_source_info  # noqa: E402


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("data/raw/fastapi"))
    parser.add_argument("--docs", type=Path, default=Path("docs/en/docs"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/index_manifest.json"),
    )
    args = parser.parse_args()

    repo_root = args.repo.resolve()
    docs_root = repo_root / args.docs
    if not docs_root.exists():
        raise SystemExit(f"docs not found: {docs_root}")

    documents = list(iter_documents(docs_root=docs_root, repo_root=repo_root))
    chunks = chunk_documents(
        documents,
        release_notes_latest_versions=settings.release_notes_latest_versions,
    )
    previous = IndexManifest.load(args.manifest)
    current = detect_source_info(
        repo_root,
        fallback_source_ref=settings.docs_source_ref
        or (previous.source_ref if previous else ""),
    )
    changes = diff_manifest(previous, documents, chunks)
    git_files = git_changed_docs(
        repo_root,
        previous.source_ref if previous else None,
        current.source_ref,
    )

    stale_hours = None
    if previous is not None:
        try:
            created = datetime.fromisoformat(previous.created_at)
            if created.tzinfo is None:
                created = created.replace(tzinfo=UTC)
            stale_hours = (datetime.now(UTC) - created).total_seconds() / 3600
        except ValueError:
            stale_hours = None

    print(f"source_ref:       {current.source_ref}")
    print(f"docs_version:     {current.docs_version}")
    print(f"previous_ref:     {previous.source_ref if previous else 'none'}")
    print(f"previous_build:   {previous.build_id if previous else 'none'}")
    print(
        "stale_hours:      "
        f"{stale_hours:.1f}" if stale_hours is not None else "stale_hours:      unknown"
    )
    print(f"git_changed:     {len(git_files) if git_files is not None else 'unavailable'}")
    for key in (
        "added_docs",
        "changed_docs",
        "deleted_docs",
        "added_chunks",
        "changed_chunks",
        "deleted_chunks",
        "reused_chunks",
    ):
        value = changes[key]
        print(f"{key + ':':<17}{len(value) if isinstance(value, list) else value}")

    changed_docs = changes["changed_docs"]
    if isinstance(changed_docs, list) and changed_docs:
        print("changed pages:")
        for path in changed_docs:
            print(f"  {path}")


if __name__ == "__main__":
    main()
