"""Minimal request observability without extra runtime dependencies."""

from __future__ import annotations

import json
import logging
import re
import sys
import uuid
from collections import Counter, defaultdict
from contextvars import ContextVar, Token
from datetime import UTC, datetime
from threading import Lock

_REQUEST_ID: ContextVar[str] = ContextVar("request_id", default="-")
_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{8,64}$")
logger = logging.getLogger("langchain_rag.http")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)


def request_id_from_header(value: str | None) -> str:
    if value and _VALID_REQUEST_ID.fullmatch(value):
        return value
    return f"req-{uuid.uuid4().hex}"


def set_request_id(value: str) -> Token[str]:
    return _REQUEST_ID.set(value)


def reset_request_id(token: Token[str]) -> None:
    _REQUEST_ID.reset(token)


def current_request_id() -> str:
    return _REQUEST_ID.get()


class Metrics:
    """In-process counters and a bounded latency window for /metrics."""

    def __init__(self, max_latency_samples: int = 2000) -> None:
        self.max_latency_samples = max_latency_samples
        self._lock = Lock()
        self._http_requests: Counter[tuple[str, str, int]] = Counter()
        self._latencies: dict[str, list[float]] = defaultdict(list)
        self._tokens: Counter[str] = Counter()
        self._quota_rejections: Counter[str] = Counter()

    def record_request(
        self,
        method: str,
        path: str,
        status: int,
        duration_ms: float,
    ) -> None:
        with self._lock:
            self._http_requests[(method, path, status)] += 1
            samples = self._latencies[path]
            samples.append(duration_ms)
            if len(samples) > self.max_latency_samples:
                del samples[: len(samples) - self.max_latency_samples]

    def record_tokens(self, kind: str, count: int) -> None:
        with self._lock:
            self._tokens[kind] += max(0, count)

    def record_quota_rejection(self, scope: str) -> None:
        with self._lock:
            self._quota_rejections[scope] += 1

    @staticmethod
    def _quantile(values: list[float], fraction: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = min(len(ordered) - 1, max(0, int(len(ordered) * fraction) - 1))
        return ordered[index]

    def render(self) -> str:
        with self._lock:
            requests = dict(self._http_requests)
            latencies = {path: list(values) for path, values in self._latencies.items()}
            tokens = dict(self._tokens)
            rejections = dict(self._quota_rejections)

        lines = [
            "# HELP rag_http_requests_total HTTP requests by method, path and status.",
            "# TYPE rag_http_requests_total counter",
        ]
        for (method, path, status), count in sorted(requests.items()):
            lines.append(
                f'rag_http_requests_total{{method="{method}",path="{path}",'
                f'status="{status}"}} {count}'
            )
        lines.extend(
            [
                "# HELP rag_http_request_duration_ms Request latency by path.",
                "# TYPE rag_http_request_duration_ms summary",
            ]
        )
        for path, values in sorted(latencies.items()):
            count = len(values)
            lines.append(
                f'rag_http_request_duration_ms{{path="{path}",quantile="0.5"}} '
                f"{self._quantile(values, 0.5):.1f}"
            )
            lines.append(
                f'rag_http_request_duration_ms{{path="{path}",quantile="0.95"}} '
                f"{self._quantile(values, 0.95):.1f}"
            )
            lines.append(
                f'rag_http_request_duration_ms_count{{path="{path}"}} {count}'
            )
        lines.extend(
            [
                "# HELP rag_tokens_reserved_total Estimated tokens reserved.",
                "# TYPE rag_tokens_reserved_total counter",
            ]
        )
        for kind, count in sorted(tokens.items()):
            lines.append(f'rag_tokens_reserved_total{{kind="{kind}"}} {count}')
        lines.extend(
            [
                "# HELP rag_quota_rejections_total Quota rejections.",
                "# TYPE rag_quota_rejections_total counter",
            ]
        )
        for scope, count in sorted(rejections.items()):
            lines.append(f'rag_quota_rejections_total{{scope="{scope}"}} {count}')
        return "\n".join(lines) + "\n"


metrics = Metrics()


def log_request(
    *,
    method: str,
    path: str,
    status: int,
    duration_ms: float,
    client: str,
) -> None:
    logger.info(
        json.dumps(
            {
                "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
                "level": "info",
                "event": "http_request",
                "request_id": current_request_id(),
                "method": method,
                "path": path,
                "status": status,
                "duration_ms": round(duration_ms, 1),
                "client": client,
            },
            ensure_ascii=False,
        )
    )


