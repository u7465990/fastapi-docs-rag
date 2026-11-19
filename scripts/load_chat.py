"""Small async load tester for retrieval and chat endpoints."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((p / 100) * (len(ordered) - 1))))
    return ordered[index]


async def one_request(
    client: httpx.AsyncClient,
    *,
    url: str,
    payload: dict[str, object],
    stream: bool,
    headers: dict[str, str] | None = None,
) -> tuple[float, int | None, str | None, float | None]:
    started = time.perf_counter()
    try:
        if not stream:
            response = await client.post(url, json=payload, headers=headers)
            elapsed = (time.perf_counter() - started) * 1000
            return elapsed, response.status_code, None, None

        ttft: float | None = None
        async with client.stream(
            "POST",
            url,
            json=payload,
            headers=headers,
        ) as response:
            if response.status_code >= 400:
                await response.aread()
                return (
                    (time.perf_counter() - started) * 1000,
                    response.status_code,
                    f"HTTP {response.status_code}",
                    None,
                )
            async for line in response.aiter_lines():
                if line.startswith("event: token") and ttft is None:
                    ttft = (time.perf_counter() - started) * 1000
            elapsed = (time.perf_counter() - started) * 1000
            return elapsed, response.status_code, None, ttft
    except Exception as exc:  # noqa: BLE001 - load test reports failures
        return (time.perf_counter() - started) * 1000, None, str(exc), None


async def run(args: argparse.Namespace) -> dict[str, object]:
    semaphore = asyncio.Semaphore(args.concurrency)
    latencies: list[float] = []
    ttfts: list[float] = []
    errors: list[str] = []
    statuses: dict[str, int] = {}

    async def limited(index: int) -> None:
        async with semaphore:
            elapsed, status, error, ttft = await one_request(
                client,
                url=args.url.rstrip("/") + args.path,
                payload={"query": f"{args.query} #{index}", "limit": args.limit},
                stream=args.stream,
                headers=(
                    {"X-Real-IP": f"10.0.{index // 250}.{index % 250 + 1}"}
                    if args.unique_ip
                    else None
                ),
            )
            latencies.append(elapsed)
            if ttft is not None:
                ttfts.append(ttft)
            if status is not None:
                statuses[str(status)] = statuses.get(str(status), 0) + 1
            if error:
                errors.append(error)

    timeout = httpx.Timeout(args.timeout)
    async with httpx.AsyncClient(timeout=timeout) as client:
        started = time.perf_counter()
        await asyncio.gather(*(limited(i) for i in range(args.requests)))
        wall_ms = (time.perf_counter() - started) * 1000

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "url": args.url.rstrip("/") + args.path,
        "concurrency": args.concurrency,
        "requests": args.requests,
        "wall_ms": round(wall_ms, 2),
        "rps": round(args.requests / (wall_ms / 1000), 2),
        "statuses": statuses,
        "errors": len(errors),
        "error_samples": errors[:5],
        "latency_ms": {
            "p50": round(percentile(latencies, 50), 2),
            "p95": round(percentile(latencies, 95), 2),
            "p99": round(percentile(latencies, 99), 2),
            "max": round(max(latencies, default=0), 2),
            "mean": round(statistics.mean(latencies), 2) if latencies else 0,
        },
        "ttft_ms": {
            "p50": round(percentile(ttfts, 50), 2) if ttfts else None,
            "p95": round(percentile(ttfts, 95), 2) if ttfts else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--path", default="/api/v1/search")
    parser.add_argument("--query", default="FastAPI 怎么写测试？")
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument(
        "--unique-ip",
        action="store_true",
        help="send distinct X-Real-IP values to isolate throughput from rate limits",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    result = asyncio.run(run(args))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
