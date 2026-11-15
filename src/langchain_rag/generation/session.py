"""Redis-backed short-term conversation history."""

from __future__ import annotations

import json
import re
from typing import Any

from redis import Redis
from redis.exceptions import RedisError

from langchain_rag.config import Settings

SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
MAX_MESSAGES = 12
SESSION_TTL_SECONDS = 7 * 24 * 60 * 60


class SessionUnavailable(RuntimeError):
    """The session store (Redis) is unavailable."""


class SessionStore:
    def __init__(
        self,
        client: Redis,
        *,
        max_messages: int = MAX_MESSAGES,
        ttl_seconds: int = SESSION_TTL_SECONDS,
    ) -> None:
        self.client = client
        self.max_messages = max_messages
        self.ttl_seconds = ttl_seconds

    @staticmethod
    def validate_id(session_id: str) -> None:
        if not SESSION_ID_PATTERN.fullmatch(session_id):
            raise ValueError(
                "session_id must be 8-64 chars of letters, digits, '_' or '-'"
            )

    @staticmethod
    def _key(session_id: str) -> str:
        return f"chat:session:{session_id}:messages"

    def load(self, session_id: str) -> list[dict[str, str]]:
        self.validate_id(session_id)
        try:
            raw = self.client.lrange(self._key(session_id), 0, -1)
        except RedisError as exc:
            raise SessionUnavailable(f"session store unavailable: {exc}") from exc

        messages: list[dict[str, str]] = []
        for item in raw:
            try:
                message: Any = json.loads(item)
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            content = message.get("content")
            if role in {"user", "assistant"} and isinstance(content, str) and content:
                messages.append({"role": role, "content": content})
        return messages[-self.max_messages :]

    def append_turn(self, session_id: str, question: str, answer: str) -> None:
        self.validate_id(session_id)
        key = self._key(session_id)
        try:
            pipe = self.client.pipeline()
            pipe.rpush(
                key,
                json.dumps({"role": "user", "content": question}, ensure_ascii=False),
                json.dumps(
                    {"role": "assistant", "content": answer},
                    ensure_ascii=False,
                ),
            )
            pipe.ltrim(key, -self.max_messages, -1)
            pipe.expire(key, self.ttl_seconds)
            pipe.execute()
        except RedisError as exc:
            raise SessionUnavailable(f"session store unavailable: {exc}") from exc


def make_session_store(settings: Settings) -> SessionStore:
    return SessionStore(Redis.from_url(settings.redis_url, decode_responses=True))
