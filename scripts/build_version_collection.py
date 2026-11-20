"""Build a small historical release-notes collection for one FastAPI version.

This is intentionally narrow: it materializes the release-notes section for a
pinned version, so version questions can query a real historical collection
without pretending that the full historical docs tree was reconstructed.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.models import Distance, PointStruct, VectorParams  # noqa: E402

from fastapi_docs_rag.config import get_settings  # noqa: E402
from fastapi_docs_rag.indexing.embedder import make_embedder  # noqa: E402
from fastapi_docs_rag.indexing.indexer import DENSE_VECTOR_NAME  # noqa: E402

DEFAULT_RELEASE_NOTES = Path("data/raw/fastapi/docs/en/docs/release-notes.md")
POINT_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "fastapi-docs-rag/history")


def extract_section(text: str, version: str) -> str:
    heading = re.compile(rf"^## {re.escape(version)}(?:\s|$)")
    lines = text.splitlines()
    start = next(
        (index for index, line in enumerate(lines) if heading.match(line)),
        None,
    )
    if start is None:
        raise ValueError(f"release notes section {version} not found")
    end = next(
        (
            index
            for index in range(start + 1, len(lines))
            if lines[index].startswith("## ")
        ),
        len(lines),
    )
    return "\n".join(lines[start:end]).strip()


def chunks(text: str, *, size: int = 1800, overlap: int = 120) -> list[str]:
    result: list[str] = []
    start = 0
    while start < len(text):
        result.append(text[start : start + size])
        if start + size >= len(text):
            break
        start += size - overlap
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="0.100.0")
    parser.add_argument("--collection", default="fastapi_docs_v0100")
    parser.add_argument("--release-notes", type=Path, default=DEFAULT_RELEASE_NOTES)
    args = parser.parse_args()

    settings = get_settings()
    section = extract_section(
        args.release_notes.read_text(encoding="utf-8"),
        args.version,
    )
    pieces = chunks(section)
    if not pieces:
        raise RuntimeError("no chunks extracted")

    embedder = make_embedder(settings)
    vectors = embedder.embed(pieces)
    if len(vectors) != len(pieces):
        raise RuntimeError("embedding count mismatch")

    client = QdrantClient(url=settings.qdrant_url)
    if client.collection_exists(args.collection):
        client.delete_collection(args.collection)
    client.create_collection(
        collection_name=args.collection,
        vectors_config={
            DENSE_VECTOR_NAME: VectorParams(
                size=embedder.dimension,
                distance=Distance.COSINE,
            )
        },
    )

    points = []
    now = datetime.now(UTC).isoformat()
    for index, (piece, vector) in enumerate(zip(pieces, vectors, strict=True)):
        chunk_id = f"release-notes-{args.version}-{index}"
        payload = {
            "chunk_id": chunk_id,
            "parent_id": f"release-notes-{args.version}",
            "doc_id": f"release-notes-{args.version}",
            "title": f"Release Notes {args.version}",
            "section_path": ["Release Notes", args.version],
            "anchor": None,
            "url": "https://fastapi.tiangolo.com/release-notes/",
            "source_path": "docs/en/docs/release-notes.md",
            "product": "fastapi",
            "language": "python",
            "doc_type": "release_notes",
            "docs_version": args.version,
            "source_ref": f"release-notes:{args.version}",
            "fetched_at": now,
            "index_build_id": f"history-{args.version}",
            "is_deprecated": False,
            "deprecation_hint": False,
            "content_type": "prose",
            "code_language": None,
            "content_sha256": hashlib.sha256(piece.encode("utf-8")).hexdigest(),
            "token_count": max(1, len(piece) // 4),
            "text": piece,
            "parent_text": section,
        }
        points.append(
            PointStruct(
                id=str(uuid.uuid5(POINT_NAMESPACE, chunk_id)),
                vector={DENSE_VECTOR_NAME: vector},
                payload=payload,
            )
        )

    client.upsert(collection_name=args.collection, points=points, wait=True)
    print(
        f"built {args.collection}: version={args.version} "
        f"chunks={len(points)} chars={len(section)}"
    )
    client.close()
    embedder.close()


if __name__ == "__main__":
    main()
