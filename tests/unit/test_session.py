"""Tests for the Redis conversation-session store."""

import pytest
from redis.exceptions import RedisError

from langchain_rag.generation.session import SessionStore, SessionUnavailable


class FakePipeline:
    def __init__(self, client):
        self.client = client
        self.operations = []

    def rpush(self, key, *values):
        self.operations.append(("rpush", key, values))
        return self

    def ltrim(self, key, start, end):
        self.operations.append(("ltrim", key, start, end))
        return self

    def expire(self, key, ttl):
        self.operations.append(("expire", key, ttl))
        return self

    def execute(self):
        for operation in self.operations:
            name = operation[0]
            if name == "rpush":
                _, key, values = operation
                self.client.data.setdefault(key, []).extend(values)
            elif name == "ltrim":
                _, key, start, end = operation
                # Redis LRANGE/LTRIM end is inclusive; Python slices are not.
                slice_end = None if end == -1 else end + 1
                self.client.data[key] = self.client.data[key][start:slice_end]
            elif name == "expire":
                pass
        self.operations = []
        return []


class FakeRedis:
    def __init__(self):
        self.data = {}

    def lrange(self, key, start, end):
        values = self.data.get(key, [])
        slice_end = None if end == -1 else end + 1
        return values[start:slice_end]

    def pipeline(self):
        return FakePipeline(self)


class BrokenRedis:
    def lrange(self, key, start, end):
        raise RedisError("down")

    def pipeline(self):
        raise RedisError("down")


def test_append_and_load_turn() -> None:
    store = SessionStore(FakeRedis())

    store.append_turn("session-123", "question", "answer")

    assert store.load("session-123") == [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
    ]


def test_history_is_trimmed_to_max_messages() -> None:
    store = SessionStore(FakeRedis(), max_messages=4)

    store.append_turn("session-123", "q1", "a1")
    store.append_turn("session-123", "q2", "a2")
    store.append_turn("session-123", "q3", "a3")

    assert store.load("session-123") == [
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "q3"},
        {"role": "assistant", "content": "a3"},
    ]


def test_invalid_session_id_is_rejected() -> None:
    store = SessionStore(FakeRedis())

    with pytest.raises(ValueError):
        store.load("short")
    with pytest.raises(ValueError):
        store.append_turn("has spaces", "q", "a")


def test_store_failure_is_unavailable() -> None:
    store = SessionStore(BrokenRedis())

    with pytest.raises(SessionUnavailable):
        store.load("session-123")
    with pytest.raises(SessionUnavailable):
        store.append_turn("session-123", "q", "a")
