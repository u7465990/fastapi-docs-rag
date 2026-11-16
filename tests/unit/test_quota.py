"""Tests for the Redis-backed daily LLM quota."""

import pytest
from redis.exceptions import RedisError

from fastapi_docs_rag.generation.quota import DailyQuota, QuotaExceeded, QuotaUnavailable


class FakeRedis:
    def __init__(self) -> None:
        self.requests = 0
        self.tokens = 0

    def eval(self, script, numkeys, *args):
        values = list(map(int, args[numkeys:]))
        request_limit, estimate, token_limit, _ttl = values
        if self.requests >= request_limit or self.tokens + estimate > token_limit:
            return [0, self.requests, self.tokens]
        self.requests += 1
        self.tokens += estimate
        return [1, self.requests, self.tokens]


class BrokenRedis:
    def eval(self, script, numkeys, *args):
        raise RedisError("redis down")


def test_reserves_requests_and_tokens() -> None:
    quota = DailyQuota(
        FakeRedis(),
        request_limit=2,
        token_budget=100,
    )

    first = quota.reserve(60)
    second = quota.reserve(30)

    assert (first.requests, first.tokens) == (1, 60)
    assert (second.requests, second.tokens) == (2, 90)


def test_token_budget_stops_before_overspend() -> None:
    quota = DailyQuota(
        FakeRedis(),
        request_limit=10,
        token_budget=100,
    )

    quota.reserve(80)

    with pytest.raises(QuotaExceeded, match="quota exhausted"):
        quota.reserve(30)


def test_request_budget_stops() -> None:
    quota = DailyQuota(
        FakeRedis(),
        request_limit=1,
        token_budget=1000,
    )

    quota.reserve(10)

    with pytest.raises(QuotaExceeded):
        quota.reserve(10)


def test_redis_failure_is_quota_unavailable() -> None:
    quota = DailyQuota(
        BrokenRedis(),
        request_limit=10,
        token_budget=1000,
    )

    with pytest.raises(QuotaUnavailable, match="unavailable"):
        quota.reserve(10)
