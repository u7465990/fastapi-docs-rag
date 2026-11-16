"""Build a Qdrant collection from the FastAPI documentation snapshot."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qdrant_client import QdrantClient  # noqa: E402

from fastapi_docs_rag.config import get_settings  # noqa: E402
from fastapi_docs_rag.indexing.chunker import chunk_documents  # noqa: E402
from fastapi_docs_rag.indexing.embedder import make_embedder  # noqa: E402
from fastapi_docs_rag.indexing.indexer import QdrantIndexer  # noqa: E402
from fastapi_docs_rag.ingestion.parser import iter_documents  # noqa: E402
from fastapi_docs_rag.ingestion.source import detect_source_info  # noqa: E402


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("data/raw/fastapi"))
    parser.add_argument("--docs", type=Path, default=Path("docs/en/docs"))
    parser.add_argument("--parent-tokens", type=int, default=1400)
    parser.add_argument("--child-tokens", type=int, default=400)
    parser.add_argument("--overlap-tokens", type=int, default=64)
    parser.add_argument("--code-max-tokens", type=int, default=800)
    parser.add_argument(
        "--release-notes-versions",
        type=int,
        default=settings.release_notes_latest_versions,
    )
    parser.add_argument("--limit", type=int, default=None, help="index only N documents")
    parser.add_argument("--build-id", default=None)
    parser.add_argument("--no-alias", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo.resolve()
    docs_root = repo_root / args.docs
    if not docs_root.exists():
        raise SystemExit(f"docs not found: {docs_root}")

    documents = list(iter_documents(docs_root=docs_root, repo_root=repo_root))
    if args.limit is not None:
        documents = documents[: args.limit]
    chunks = chunk_documents(
        documents,
        parent_tokens=args.parent_tokens,
        child_tokens=args.child_tokens,
        overlap_tokens=args.overlap_tokens,
        code_max_tokens=args.code_max_tokens,
        release_notes_latest_versions=args.release_notes_versions,
    )
    children = [chunk for chunk in chunks if chunk.level == "child"]
    source = detect_source_info(repo_root)
    build_id = args.build_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    print(f"documents:      {len(documents)}")
    print(f"chunks:         {len(chunks)} (children {len(children)})")
    print(f"docs_version:   {source.docs_version}")
    print(f"source_ref:     {source.source_ref}")
    print(f"fetched_at:     {source.fetched_at}")
    print(f"build_id:       {build_id}")
    print(f"embedding:      {settings.embedding_model} ({settings.embedding_dim}-d)")

    if args.dry_run:
        return

    if not children:
        raise SystemExit("no child chunks to index")

    embedder = make_embedder(settings)
    client = QdrantClient(url=settings.qdrant_url)
    try:
        indexer = QdrantIndexer(
            client,
            vector_size=settings.embedding_dim,
            build_prefix=settings.qdrant_build_prefix,
            alias=None if args.no_alias else settings.qdrant_collection,
        )
        stats = indexer.index(
            documents,
            chunks,
            embedder=embedder,
            source=source,
            build_id=build_id,
        )
    finally:
        client.close()
        close = getattr(embedder, "close", None)
        if close is not None:
            close()

    print(f"build_collection: {stats.build_collection}")
    print(f"alias:            {stats.alias}")
    print(f"vectors:          {stats.vectors}")
    print("index complete")


if __name__ == "__main__":
    main()
