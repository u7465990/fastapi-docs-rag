"""Smoke-check the FastAPI Markdown parser over a local checkout."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langchain_rag.ingestion.parser import iter_documents  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("data/raw/fastapi"))
    parser.add_argument("--docs", type=Path, default=Path("docs/en/docs"))
    args = parser.parse_args()

    repo_root = args.repo
    docs_root = repo_root / args.docs
    documents = list(iter_documents(docs_root=docs_root, repo_root=repo_root))

    types = Counter(document.doc_type for document in documents)
    missing = [
        document.relative_path.as_posix()
        for document in documents
        if "missing code include" in document.markdown
    ]

    print(f"documents:       {len(documents)}")
    print(f"sections:        {sum(len(d.sections) for d in documents)}")
    print(f"code includes:   {sum(len(d.code_includes) for d in documents)}")
    print(f"deprecated docs: {sum(1 for d in documents if d.has_deprecated_section)}")
    print(f"doc types:       {dict(sorted(types.items()))}")
    print(f"missing includes: {len(missing)}")
    for path in missing[:10]:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
