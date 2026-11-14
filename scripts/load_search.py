"""Concurrent smoke load test for the search endpoint."""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time

import httpx


async def _one(client: httpx.AsyncClient, url: str, query: str, limit: int):
    start = time.perf_counter()
    response = await client.post(url, json={"query": query, "limit": limit})
    elapsed_ms = (time.perf_counter() - start) * 1000
    return response.status_code, elapsed_ms


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/api/v1/search")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument(
        "--query",
        default="Depends 和 Annotated 怎么一起用？",
    )
    args = parser.parse_args()
    if args.concurrency <= 0:
        raise SystemExit("concurrency must be positive")

    async with httpx.AsyncClient(timeout=60.0) as client:
        results = await asyncio.gather(
            *(
                _one(client, args.url, args.query, args.limit)
                for _ in range(args.concurrency)
            )
        )

    statuses = [status for status, _ in results]
    latencies = sorted(latency for _, latency in results)
    p95_index = max(0, int(len(latencies) * 0.95) - 1)
    print(f"requests: {len(results)} concurrency={args.concurrency}")
    print(f"statuses: {statuses}")
    print(
        "latency_ms: "
        f"min={latencies[0]:.0f} "
        f"p50={statistics.median(latencies):.0f} "
        f"p95={latencies[p95_index]:.0f} "
        f"max={latencies[-1]:.0f}"
    )
    if any(status != 200 for status in statuses):
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
