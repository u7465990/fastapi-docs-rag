"""Smoke-check parent/child chunking over a local FastAPI checkout."""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langchain_rag.indexing.chunker import chunk_document  # noqa: E402
from langchain_rag.ingestion.parser import iter_documents  # noqa: E402


def _stats(values: list[int]) -> str:
    if not values:
        return "n/a"
    return f"min {min(values)} / median {int(statistics.median(values))} / max {max(values)}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("data/raw/fastapi"))
    parser.add_argument("--docs", type=Path, default=Path("docs/en/docs"))
    parser.add_argument("--parent-tokens", type=int, default=1400)
    parser.add_argument("--child-tokens", type=int, default=400)
    parser.add_argument("--overlap-tokens", type=int, default=64)
    parser.add_argument("--code-max-tokens", type=int, default=800)
    args = parser.parse_args()

    repo_root = args.repo
    documents = list(iter_documents(docs_root=repo_root / args.docs, repo_root=repo_root))
    chunks = [
        chunk
        for document in documents
        for chunk in chunk_document(
            document,
            parent_tokens=args.parent_tokens,
            child_tokens=args.child_tokens,
            overlap_tokens=args.overlap_tokens,
            code_max_tokens=args.code_max_tokens,
        )
    ]
    parents = [chunk for chunk in chunks if chunk.level == "parent"]
    children = [chunk for chunk in chunks if chunk.level == "child"]

    print(f"documents:   {len(documents)}")
    print(f"chunks:      {len(chunks)} (parents {len(parents)}, children {len(children)})")
    print(f"parent size: {_stats([chunk.token_count for chunk in parents])}")
    print(f"child size:  {_stats([chunk.token_count for chunk in children])}")
    print(f"content:     {dict(Counter(chunk.content_type for chunk in children))}")
    print(f"languages:   {dict(Counter(c.code_language for c in children if c.code_language))}")
    print(f"deprecated children: {sum(1 for chunk in children if chunk.is_deprecated)}")
    print(f"children/parent: {len(children) / max(len(parents), 1):.2f}")

    oversized = [chunk for chunk in children if chunk.token_count > args.child_tokens]
    print(f"over child budget: {len(oversized)}")


if __name__ == "__main__":
    main()
