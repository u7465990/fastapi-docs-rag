"""Run a few fixed queries against the current Qdrant index."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qdrant_client import QdrantClient  # noqa: E402

from fastapi_docs_rag.config import get_settings  # noqa: E402
from fastapi_docs_rag.indexing.embedder import make_embedder  # noqa: E402
from fastapi_docs_rag.retrieval.dense import DenseRetriever  # noqa: E402

# (Chinese query, accepted URL fragments)
SMOKE_QUERIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "FastAPI 里 on_event 怎么迁移到 lifespan？",
        ("advanced/events/",),
    ),
    (
        "Depends 和 Annotated 怎么一起用？",
        ("tutorial/dependencies/",),
    ),
    (
        "怎么定义带路径参数和查询参数的 GET 接口？",
        ("tutorial/path-params/", "tutorial/query-params/"),
    ),
    (
        "BackgroundTasks 怎么用？",
        ("tutorial/background-tasks/",),
    ),
    (
        "Pydantic v1 的 validator 在 v2 怎么写？",
        ("how-to/migrate-from-pydantic-v1-to-pydantic-v2/",),
    ),
)


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default=settings.qdrant_collection)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--docs-version", default=None)
    parser.add_argument("--exclude-deprecated", action="store_true")
    args = parser.parse_args()

    embedder = make_embedder(settings)
    client = QdrantClient(url=settings.qdrant_url)
    retriever = DenseRetriever(client, embedder, collection=args.collection)

    passed = 0
    try:
        for query, expected in SMOKE_QUERIES:
            hits = retriever.search(
                query,
                limit=args.limit,
                docs_version=args.docs_version,
                include_deprecated=not args.exclude_deprecated,
            )
            urls = [hit.url for hit in hits]
            ok = any(fragment in url for url in urls for fragment in expected)
            passed += int(ok)
            print(f"\n[{'OK' if ok else 'MISS'}] {query}")
            print(f"  expected: {', '.join(expected)}")
            for hit in hits:
                print(f"  {hit.score:.4f}  {hit.url}")
                print(f"          {hit.title} / {hit.section_path}")
    finally:
        client.close()
        embedder.close()

    print(f"\n{passed}/{len(SMOKE_QUERIES)} queries matched an expected URL")


if __name__ == "__main__":
    main()
