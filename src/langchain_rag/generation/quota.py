"""Application-level daily LLM quota backed by Redis.

Provider billing is not enough: free quotas, rate limits, and account arrears
are all external. This guard reserves an estimated token budget *before* the
LLM call, so the application itself stops spending at a configured line.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from redis import Redis
from redis.exceptions import RedisError

from langchain_rag.config import Settings

QUOTA_TTL_SECONDS = 2 * 24 * 60 * 60

_RESERVE_LUA = """
local requests = tonumber(redis.call('GET', KEYS[1]) or '0')
local tokens = tonumber(redis.call('GET', KEYS[2]) or '0')
local request_limit = tonumber(ARGV[1])
local estimate = tonumber(ARGV[2])
local token_limit = tonumber(ARGV[3])
local ttl = tonumber(ARGV[4])

if requests >= request_limit then
  return {0, requests, tokens}
end
if tokens + estimate > token_limit then
  return {0, requests, tokens}
end

redis.call('INCR', KEYS[1])
redis.call('INCRBY', KEYS[2], estimate)
redis.call('EXPIRE', KEYS[1], ttl)
redis.call('EXPIRE', KEYS[2], ttl)
return {1, requests + 1, tokens + estimate}
"""


class QuotaExceeded(RuntimeError):
    """The configured daily request or token budget is exhausted."""


class QuotaUnavailable(RuntimeError):
    """The quota store (Redis) is unavailable."""


@dataclass(frozen=True)
class QuotaState:
    requests: int
    tokens: int


class DailyQuota:
    def __init__(
        self,
        client: Redis,
        *,
        request_limit: int,
        token_budget: int,
        ttl_seconds: int = QUOTA_TTL_SECONDS,
    ) -> None:
        if request_limit <= 0:
            raise ValueError("request_limit must be positive")
        if token_budget <= 0:
            raise ValueError("token_budget must be positive")
        self.client = client
        self.request_limit = request_limit
        self.token_budget = token_budget
        self.ttl_seconds = ttl_seconds

    def reserve(self, estimated_tokens: int) -> QuotaState:
        """Atomically reserve one request and its estimated token cost."""
        estimate = max(1, estimated_tokens)
        day = datetime.now(UTC).strftime("%Y%m%d")
        request_key = f"quota:llm:requests:{day}"
        token_key = f"quota:llm:tokens:{day}"
        try:
            result: Any = self.client.eval(
                _RESERVE_LUA,
                2,
                request_key,
                token_key,
                self.request_limit,
                estimate,
                self.token_budget,
                self.ttl_seconds,
            )
        except RedisError as exc:
            raise QuotaUnavailable(f"quota store unavailable: {exc}") from exc

        if not isinstance(result, (list, tuple)) or len(result) < 3:
            raise QuotaUnavailable("quota store returned an invalid response")
        allowed = int(result[0])
        requests = int(result[1])
        tokens = int(result[2])
        if not allowed:
            raise QuotaExceeded(
                "daily LLM quota exhausted "
                f"(requests {requests}/{self.request_limit}, "
                f"tokens {tokens}/{self.token_budget})"
            )
        return QuotaState(requests=requests, tokens=tokens)


def make_daily_quota(settings: Settings) -> DailyQuota:
    client = Redis.from_url(settings.redis_url, decode_responses=True)
    return DailyQuota(
        client,
        request_limit=settings.llm_daily_request_limit,
        token_budget=settings.llm_daily_token_budget,
    )
