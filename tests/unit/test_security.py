"""Tests for HTTP rate limiting and the shared access token."""

import pytest
from fastapi import HTTPException
from redis.exceptions import RedisError
from starlette.requests import Request

from langchain_rag.security import (
    RateLimiter,
    RateLimitExceeded,
    RateLimitUnavailable,
    check_access_token,
    client_identifier,
)


class FakeRedis:
    def __init__(self) -> None:
        self.count = 0

    def eval(self, script, numkeys, *args):
        limit = int(args[1])
        self.count += 1
        return [1 if self.count <= limit else 0, self.count]


class BrokenRedis:
    def eval(self, script, numkeys, *args):
        raise RedisError("redis down")


def _request(headers=None, client=("1.2.3.4", 1234)) -> Request:
    encoded = [
        (key.lower().encode(), value.encode())
        for key, value in (headers or {}).items()
    ]
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": encoded,
            "client": client,
            "query_string": b"",
            "scheme": "http",
            "server": ("test", 80),
        }
    )


def test_rate_limit_allows_then_blocks() -> None:
    limiter = RateLimiter(FakeRedis(), limit_per_minute=2)

    assert limiter.check("search", "1.2.3.4") == 1
    assert limiter.check("search", "1.2.3.4") == 2
    with pytest.raises(RateLimitExceeded):
        limiter.check("search", "1.2.3.4")


def test_rate_limit_store_failure_is_unavailable() -> None:
    limiter = RateLimiter(BrokenRedis(), limit_per_minute=2)

    with pytest.raises(RateLimitUnavailable):
        limiter.check("search", "1.2.3.4")


def test_access_token_is_optional_when_empty() -> None:
    check_access_token(_request(), "")


def test_access_token_accepts_bearer_and_x_api_key() -> None:
    check_access_token(_request({"authorization": "Bearer secret"}), "secret")
    check_access_token(_request({"x-api-key": "secret"}), "secret")


def test_access_token_rejects_missing_or_wrong() -> None:
    with pytest.raises(HTTPException) as missing:
        check_access_token(_request(), "secret")
    assert missing.value.status_code == 401

    with pytest.raises(HTTPException) as wrong:
        check_access_token(_request({"authorization": "Bearer wrong"}), "secret")
    assert wrong.value.status_code == 401


def test_real_ip_header_wins() -> None:
    request = _request({"x-real-ip": "203.0.113.9"})

    assert client_identifier(request) == "203.0.113.9"
