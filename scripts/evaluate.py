"""Evaluate dense retrieval against the committed golden set."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qdrant_client import QdrantClient  # noqa: E402

from fastapi_docs_rag.config import get_settings  # noqa: E402
from fastapi_docs_rag.indexing.embedder import make_embedder  # noqa: E402
from fastapi_docs_rag.retrieval.bm25 import BM25Index  # noqa: E402
from fastapi_docs_rag.retrieval.dense import DenseRetriever  # noqa: E402
from fastapi_docs_rag.retrieval.hybrid import HybridRetriever  # noqa: E402
from fastapi_docs_rag.retrieval.rerank import LexicalReranker  # noqa: E402

DEFAULT_GOLDEN = Path("data/eval/golden_set.jsonl")


def load_cases(path: Path) -> list[dict]:
    cases = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
        if not case.get("question") or not case.get("expected_urls"):
            raise SystemExit(f"missing question/expected_urls at {path}:{line_number}")
        cases.append(case)
    return cases


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(len(ordered) * fraction) - 1))
    return ordered[index]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    settings = get_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--collection", default=settings.qdrant_collection)
    parser.add_argument("--mode", choices=("dense", "hybrid"), default="dense")
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.top_k <= 0:
        raise SystemExit("--top-k must be positive")

    cases = load_cases(args.golden)
    embedder = make_embedder(settings)
    client = QdrantClient(url=settings.qdrant_url)
    dense = DenseRetriever(client, embedder, collection=args.collection)
    if args.mode == "hybrid":
        retriever = HybridRetriever(
            dense,
            BM25Index.from_qdrant(client, args.collection),
            dense_top_k=settings.hybrid_dense_top_k,
            bm25_top_k=settings.hybrid_bm25_top_k,
            rrf_k=settings.hybrid_rrf_k,
            dense_weight=settings.hybrid_dense_weight,
            bm25_weight=settings.hybrid_bm25_weight,
            reranker=LexicalReranker() if args.rerank else None,
        )
    else:
        retriever = dense
    results = []
    latencies: list[float] = []
    category_ranks: dict[str, list[int | None]] = defaultdict(list)

    try:
        for case in cases:
            start = time.perf_counter()
            hits = retriever.search(case["question"], limit=args.top_k)
            latency_ms = (time.perf_counter() - start) * 1000
            latencies.append(latency_ms)
            expected = case["expected_urls"]
            rank = None
            for index, hit in enumerate(hits, start=1):
                if any(fragment in hit.url for fragment in expected):
                    rank = index
                    break
            results.append(
                {
                    "id": case.get("id"),
                    "question": case["question"],
                    "category": case.get("category", "uncategorized"),
                    "expected_urls": expected,
                    "rank": rank,
                    "top_url": hits[0].url if hits else None,
                    "top_score": hits[0].score if hits else None,
                    "latency_ms": latency_ms,
                }
            )
            category_ranks[case.get("category", "uncategorized")].append(rank)
    finally:
        client.close()
        embedder.close()

    total = len(results)
    recall_at = {
        k: sum(1 for result in results if result["rank"] is not None and result["rank"] <= k)
        / total
        for k in (1, 5, 10)
    }
    mrr = sum(
        1.0 / result["rank"] for result in results if result["rank"] is not None
    ) / total

    print(f"cases: {total}")
    for k, value in recall_at.items():
        print(f"Recall@{k}: {value:.3f}")
    print(f"MRR: {mrr:.3f}")
    print(
        "latency_ms: "
        f"p50={percentile(latencies, 0.50):.0f} "
        f"p95={percentile(latencies, 0.95):.0f} "
        f"max={max(latencies):.0f}"
    )

    category_summary = {}
    print("Recall@5 by category:")
    for category, ranks in sorted(category_ranks.items()):
        value = sum(1 for rank in ranks if rank is not None and rank <= 5) / len(ranks)
        category_summary[category] = {
            "cases": len(ranks),
            "recall_at_5": value,
        }
        print(f"  {category}: {value:.3f} ({len(ranks)} cases)")

    misses = [result for result in results if result["rank"] is None]
    print(f"misses: {len(misses)}")
    for result in misses:
        print(f"  MISS {result['id']}: {result['question']}")
        print(f"       expected={result['expected_urls']} top={result['top_url']}")

    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "collection": args.collection,
        "mode": args.mode,
        "rerank": args.rerank,
        "top_k": args.top_k,
        "cases": total,
        "metrics": {
            "recall_at_1": recall_at[1],
            "recall_at_5": recall_at[5],
            "recall_at_10": recall_at[10],
            "mrr": mrr,
            "latency_p50_ms": percentile(latencies, 0.50),
            "latency_p95_ms": percentile(latencies, 0.95),
            "latency_max_ms": max(latencies),
            "misses": len(misses),
        },
        "categories": category_summary,
        "results": results,
    }
    output = args.output or Path("data/reports") / (
        "retrieval_eval_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + ".json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"report: {output}")


if __name__ == "__main__":
    main()
