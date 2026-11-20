"""Evaluate end-to-end generation, citations, and refusal behavior."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qdrant_client import QdrantClient  # noqa: E402

from fastapi_docs_rag.config import Settings, get_settings  # noqa: E402
from fastapi_docs_rag.generation.chat import REFUSAL, generate_answer  # noqa: E402
from fastapi_docs_rag.generation.llm import make_chat_client  # noqa: E402
from fastapi_docs_rag.indexing.embedder import make_embedder  # noqa: E402
from fastapi_docs_rag.retrieval.bm25 import BM25Index  # noqa: E402
from fastapi_docs_rag.retrieval.dense import DenseRetriever, SearchHit  # noqa: E402
from fastapi_docs_rag.retrieval.hybrid import HybridRetriever  # noqa: E402
from fastapi_docs_rag.retrieval.rerank import LexicalReranker  # noqa: E402
from fastapi_docs_rag.retrieval.router import route_doc_type  # noqa: E402
from fastapi_docs_rag.retrieval.versioning import select_collection  # noqa: E402

DEFAULT_GOLDEN = Path("data/eval/generation_set.jsonl")
CITATION_RE = re.compile(r"\[(\d+)\]")
REFUSAL_MARKERS = (
    "没有找到",
    "没有足够",
    "未找到",
    "无法回答",
    "不知道",
    "无法提供",
    "不直接提供",
    "不支持",
)


def load_cases(path: Path) -> list[dict]:
    cases = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
        if not case.get("question"):
            raise SystemExit(f"missing question at {path}:{line_number}")
        cases.append(case)
    return cases


def build_retriever(
    settings: Settings,
    client: QdrantClient,
    collection: str | None = None,
) -> DenseRetriever | HybridRetriever:
    target = collection or settings.qdrant_collection
    embedder = make_embedder(settings)
    dense = DenseRetriever(client, embedder, collection=target)
    if settings.retrieval_mode.lower() != "hybrid":
        return dense
    bm25 = BM25Index.from_qdrant(client, target)
    return HybridRetriever(
        dense,
        bm25,
        dense_top_k=settings.hybrid_dense_top_k,
        bm25_top_k=settings.hybrid_bm25_top_k,
        rrf_k=settings.hybrid_rrf_k,
        dense_weight=settings.hybrid_dense_weight,
        bm25_weight=settings.hybrid_bm25_weight,
        reranker=LexicalReranker() if settings.rerank_enabled else None,
    )


def has_refusal(answer: str) -> bool:
    return answer.strip() == REFUSAL or any(marker in answer for marker in REFUSAL_MARKERS)


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
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="evaluate only first N cases")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    cases = load_cases(args.golden)[args.start :]
    if args.limit:
        cases = cases[: args.limit]

    qdrant = QdrantClient(url=settings.qdrant_url)
    retrievers: dict[str, DenseRetriever | HybridRetriever] = {}
    chat_client = make_chat_client(settings)
    results = []
    latencies: list[float] = []
    answerable = 0
    no_answer = 0

    try:
        for case in cases:
            question = case["question"]
            should_refuse = bool(case.get("should_refuse", False))
            force_no_context = bool(case.get("force_no_context", False))
            start = time.perf_counter()
            collection, _historical = select_collection(
                question,
                case.get("docs_version"),
                settings,
            )
            retriever = retrievers.get(collection)
            if retriever is None:
                retriever = build_retriever(settings, qdrant, collection)
                retrievers[collection] = retriever
            try:
                hits: list[SearchHit] = (
                    []
                    if force_no_context
                    else retriever.search(
                        question,
                        limit=settings.final_top_k,
                        doc_type=route_doc_type(question, case.get("doc_type")),
                    )
                )
                answer = generate_answer(question, hits, chat_client)
                error = None
            except Exception as exc:  # noqa: BLE001 - report provider failures
                hits = []
                answer = ""
                error = str(exc)[:300]
            latency_ms = (time.perf_counter() - start) * 1000
            latencies.append(latency_ms)

            source_urls = [hit.url for hit in hits]
            expected = case.get("expected_urls", [])
            keywords = case.get("expected_keywords", [])
            source_recall = (
                any(fragment in url for fragment in expected for url in source_urls)
                if expected
                else None
            )
            cited = [int(match) for match in CITATION_RE.findall(answer)]
            valid_citations = [
                index for index in cited if 1 <= index <= len(source_urls)
            ]
            citation_validity = (
                len(valid_citations) / len(cited)
                if cited
                else (1.0 if should_refuse else 0.0)
            )
            cited_urls = [
                source_urls[index - 1] for index in valid_citations
            ]
            citation_hit = (
                any(fragment in url for fragment in expected for url in cited_urls)
                if expected
                else None
            )
            keyword_coverage = (
                sum(1 for keyword in keywords if keyword.lower() in answer.lower())
                / len(keywords)
                if keywords
                else 1.0
            )
            answer_pass = keyword_coverage == 1.0
            refused = has_refusal(answer)
            if should_refuse:
                no_answer += 1
                case_pass = refused and error is None
            else:
                answerable += 1
                case_pass = (
                    error is None
                    and bool(source_recall)
                    and answer_pass
                    and citation_validity == 1.0
                    and bool(citation_hit)
                    and not refused
                )
            results.append(
                {
                    "id": case.get("id"),
                    "question": question,
                    "category": case.get("category"),
                    "should_refuse": should_refuse,
                    "answer": answer,
                    "sources": source_urls,
                    "expected_urls": expected,
                    "expected_keywords": keywords,
                    "error": error,
                    "latency_ms": latency_ms,
                    "source_recall": source_recall,
                    "citation_validity": citation_validity,
                    "citation_hit": citation_hit,
                    "keyword_coverage": keyword_coverage,
                    "refused": refused,
                    "pass": case_pass,
                }
            )
            status = "PASS" if case_pass else "FAIL"
            print(f"[{status}] {case.get('id')} {question}")
            if not case_pass:
                print(f"       answer={answer[:140]!r}")
                print(
                    "       source_recall="
                    f"{source_recall} citation_hit={citation_hit} "
                    f"keyword={keyword_coverage:.2f} refused={refused}"
                )
    finally:
        chat_client.close()
        qdrant.close()

    total = len(results)
    passed = sum(1 for result in results if result["pass"])
    answerable_results = [result for result in results if not result["should_refuse"]]
    no_answer_results = [result for result in results if result["should_refuse"]]
    metrics = {
        "cases": total,
        "answerable": answerable,
        "no_answer": no_answer,
        "passed": passed,
        "pass_rate": passed / total if total else 0.0,
        "source_recall": (
            sum(1 for result in answerable_results if result["source_recall"])
            / len(answerable_results)
            if answerable_results
            else 0.0
        ),
        "citation_validity": (
            sum(result["citation_validity"] for result in answerable_results)
            / len(answerable_results)
            if answerable_results
            else 0.0
        ),
        "citation_hit": (
            sum(1 for result in answerable_results if result["citation_hit"])
            / len(answerable_results)
            if answerable_results
            else 0.0
        ),
        "keyword_coverage": (
            sum(result["keyword_coverage"] for result in answerable_results)
            / len(answerable_results)
            if answerable_results
            else 0.0
        ),
        "refusal_accuracy": (
            sum(1 for result in no_answer_results if result["refused"])
            / len(no_answer_results)
            if no_answer_results
            else 0.0
        ),
        "latency_p50_ms": percentile(latencies, 0.50),
        "latency_p95_ms": percentile(latencies, 0.95),
        "latency_max_ms": max(latencies) if latencies else 0.0,
    }

    print()
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    output = args.output or Path("data/reports") / (
        "generation_eval_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + ".json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "model": settings.llm_model,
                "metrics": metrics,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"report: {output}")


if __name__ == "__main__":
    main()
