"""Incrementally update the Qdrant index, reusing unchanged vectors."""

from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qdrant_client import QdrantClient  # noqa: E402

from fastapi_docs_rag.config import get_settings  # noqa: E402
from fastapi_docs_rag.indexing.chunker import chunk_documents  # noqa: E402
from fastapi_docs_rag.indexing.embedder import make_embedder  # noqa: E402
from fastapi_docs_rag.indexing.indexer import DENSE_VECTOR_NAME, QdrantIndexer  # noqa: E402
from fastapi_docs_rag.indexing.manifest import (  # noqa: E402
    IndexManifest,
    build_manifest,
    diff_manifest,
)
from fastapi_docs_rag.ingestion.parser import iter_documents  # noqa: E402
from fastapi_docs_rag.ingestion.source import detect_source_info  # noqa: E402


def load_previous_vectors(
    client: QdrantClient,
    collection: str,
) -> dict[str, tuple[str, list[float]]]:
    """Read old child vectors and content hashes from the current collection."""
    if not client.collection_exists(collection):
        return {}
    vectors: dict[str, tuple[str, list[float]]] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        for point in points:
            payload = point.payload or {}
            chunk_id = payload.get("chunk_id")
            content_hash = payload.get("content_sha256")
            if not chunk_id or not content_hash:
                continue
            raw_vector: Any = point.vector
            if isinstance(raw_vector, dict):
                raw_vector = raw_vector.get(DENSE_VECTOR_NAME)
            if isinstance(raw_vector, list) and raw_vector:
                vectors[str(chunk_id)] = (
                    str(content_hash),
                    [float(value) for value in raw_vector],
                )
        if offset is None:
            break
    return vectors


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
    parser.add_argument("--build-id", default=None)
    parser.add_argument("--no-alias", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
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
        parent_tokens=args.parent_tokens,
        child_tokens=args.child_tokens,
        overlap_tokens=args.overlap_tokens,
        code_max_tokens=args.code_max_tokens,
        release_notes_latest_versions=args.release_notes_versions,
    )
    children = [chunk for chunk in chunks if chunk.level == "child"]
    previous = IndexManifest.load(args.manifest)
    source = detect_source_info(
        repo_root,
        fallback_source_ref=settings.docs_source_ref
        or (previous.source_ref if previous else ""),
    )
    changes = diff_manifest(previous, documents, chunks)
    build_id = args.build_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    print(f"documents:       {len(documents)}")
    print(f"children:        {len(children)}")
    print(f"docs_version:    {source.docs_version}")
    print(f"source_ref:      {source.source_ref}")
    print(f"previous_ref:    {previous.source_ref if previous else 'none'}")
    print(f"added_docs:      {len(changes['added_docs'])}")
    print(f"changed_docs:    {len(changes['changed_docs'])}")
    print(f"deleted_docs:    {len(changes['deleted_docs'])}")
    print(f"changed_chunks:  {len(changes['changed_chunks'])}")
    print(f"reused_chunks:   {changes['reused_chunks']}")

    if args.dry_run:
        return
    if not children:
        raise SystemExit("no child chunks to index")

    embedder = make_embedder(settings)
    client = QdrantClient(url=settings.qdrant_url)
    try:
        old_vectors = (
            {}
            if args.rebuild
            else load_previous_vectors(client, settings.qdrant_collection)
        )
        current_hashes = {
            chunk.chunk_id: hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()
            for chunk in children
        }
        reuse_vectors = {
            chunk_id: vector
            for chunk_id, vector in (
                (
                    chunk_id,
                    old_vectors.get(chunk_id),
                )
                for chunk_id in current_hashes
            )
            if vector is not None and vector[0] == current_hashes[chunk_id]
        }
        reuse_map = {
            chunk_id: vector[1]
            for chunk_id, vector in reuse_vectors.items()
        }
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
            reuse_vectors=reuse_map,
        )
    finally:
        client.close()
        close = getattr(embedder, "close", None)
        if close is not None:
            close()

    manifest = build_manifest(
        build_id=build_id,
        build_collection=stats.build_collection,
        source=source,
        embedding_model=settings.embedding_model,
        embedding_dim=settings.embedding_dim,
        documents=documents,
        chunks=chunks,
        previous=previous,
    )
    manifest.save(args.manifest)

    print(f"build_collection: {stats.build_collection}")
    print(f"alias:            {stats.alias}")
    print(f"vectors:          {stats.vectors}")
    print(f"reused_vectors:   {stats.reused_vectors}")
    print(f"embedded_vectors: {stats.embedded_vectors}")
    print(f"manifest:         {args.manifest}")
    print("incremental update complete")


if __name__ == "__main__":
    main()
