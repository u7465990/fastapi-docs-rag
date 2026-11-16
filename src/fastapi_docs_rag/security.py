"""Small HTTP security helpers: shared access token and Redis rate limit."""

from __future__ import annotations

import secrets
import time
from typing import Any

from fastapi import HTTPException, Request
from redis import Redis
from redis.exceptions import RedisError

from fastapi_docs_rag.config import Settings

RATE_LIMIT_TTL_SECONDS = 90

_RATE_LUA = """
local current = redis.call('INCR', KEYS[1])
if current == 1 then
  redis.call('EXPIRE', KEYS[1], ARGV[2])
end
if current > tonumber(ARGV[1]) then
  return {0, current}
end
return {1, current}
"""


class RateLimitExceeded(RuntimeError):
    """The caller exceeded its per-minute request budget."""


class RateLimitUnavailable(RuntimeError):
    """The rate limiter store (Redis) is unavailable."""


class RateLimiter:
    def __init__(
        self,
        client: Redis,
        *,
        limit_per_minute: int,
        ttl_seconds: int = RATE_LIMIT_TTL_SECONDS,
    ) -> None:
        if limit_per_minute <= 0:
            raise ValueError("limit_per_minute must be positive")
        self.client = client
        self.limit_per_minute = limit_per_minute
        self.ttl_seconds = ttl_seconds

    def check(self, scope: str, identifier: str) -> int:
        minute = int(time.time() // 60)
        key = f"rate:{scope}:{identifier}:{minute}"
        try:
            result: Any = self.client.eval(
                _RATE_LUA,
                1,
                key,
                self.limit_per_minute,
                self.ttl_seconds,
            )
        except RedisError as exc:
            raise RateLimitUnavailable(f"rate limit store unavailable: {exc}") from exc
        if not isinstance(result, (list, tuple)) or len(result) < 2:
            raise RateLimitUnavailable("rate limit store returned an invalid response")
        if not int(result[0]):
            raise RateLimitExceeded(
                f"rate limit exceeded ({int(result[1])}/{self.limit_per_minute} per minute)"
            )
        return int(result[1])


def make_rate_limiter(settings: Settings) -> RateLimiter:
    client = Redis.from_url(settings.redis_url, decode_responses=True)
    return RateLimiter(
        client,
        limit_per_minute=settings.rate_limit_per_minute,
    )


def client_identifier(request: Request) -> str:
    """Use the proxy-provided real IP, falling back to the socket peer."""
    # Nginx sets X-Real-IP. Do not trust X-Forwarded-For from arbitrary
    # clients when the API is exposed directly.
    return request.headers.get("x-real-ip") or (
        request.client.host if request.client else "unknown"
    )


def check_access_token(request: Request, expected_token: str) -> None:
    """Require a shared token only when API_ACCESS_TOKEN is configured."""
    if not expected_token:
        return
    header = request.headers.get("authorization", "")
    provided = ""
    if header.lower().startswith("bearer "):
        provided = header[7:].strip()
    else:
        provided = request.headers.get("x-api-key", "").strip()
    if not provided or not secrets.compare_digest(provided, expected_token):
        raise HTTPException(status_code=401, detail="invalid access token")
