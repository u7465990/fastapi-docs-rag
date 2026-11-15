"""OpenAI-compatible chat client for answer generation."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator, Sequence
from typing import Protocol

import httpx

from langchain_rag.config import Settings


class LLMError(RuntimeError):
    """Raised when the chat provider returns an unusable response."""


class ChatClient(Protocol):
    model: str

    def complete(self, messages: Sequence[dict[str, str]]) -> str:
        """Return the assistant text for *messages*."""

    def stream(self, messages: Sequence[dict[str, str]]) -> Iterator[str]:
        """Yield assistant text chunks for *messages*."""


class OpenAICompatibleChatClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        timeout: float = 90.0,
        max_retries: int = 3,
        retry_backoff: float = 1.0,
        client: httpx.Client | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("LLM API key is required")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self._client = client or httpx.Client(timeout=timeout)
        self._owns_client = client is None

    @property
    def url(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def complete(self, messages: Sequence[dict[str, str]]) -> str:
        payload = {
            "model": self.model,
            "messages": list(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.post(
                    self.url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
            except httpx.HTTPError as exc:
                if attempt >= self.max_retries:
                    raise LLMError(f"chat request failed: {exc}") from exc
                time.sleep(self.retry_backoff * (attempt + 1))
                continue

            if response.status_code >= 400:
                retryable = response.status_code == 429 or response.status_code >= 500
                if retryable and attempt < self.max_retries:
                    time.sleep(self.retry_backoff * (attempt + 1))
                    continue
                detail = response.text[:300].replace("\n", " ")
                raise LLMError(
                    f"chat request failed with HTTP {response.status_code}: {detail}"
                )

            try:
                content = response.json()["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                raise LLMError(f"invalid chat response: {exc}") from exc
            if not isinstance(content, str) or not content.strip():
                raise LLMError("chat provider returned empty content")
            return content.strip()

        raise LLMError("chat request exhausted retries")

    def stream(self, messages: Sequence[dict[str, str]]) -> Iterator[str]:
        payload = {
            "model": self.model,
            "messages": list(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            with self._client.stream(
                "POST",
                self.url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            ) as response:
                if response.status_code >= 400:
                    response.read()
                    detail = response.text[:300].replace("\n", " ")
                    raise LLMError(
                        f"chat stream failed with HTTP {response.status_code}: {detail}"
                    )
                for line in response.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        event = json.loads(data)
                        delta = event["choices"][0].get("delta", {})
                    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                        continue
                    content = delta.get("content")
                    if isinstance(content, str) and content:
                        yield content
        except httpx.HTTPError as exc:
            raise LLMError(f"chat stream failed: {exc}") from exc


def make_chat_client(settings: Settings) -> ChatClient:
    provider = settings.llm_provider.lower()
    if provider not in {"openai", "openai_compatible"}:
        raise ValueError(f"unsupported LLM provider: {settings.llm_provider}")
    if not settings.llm_api_key:
        raise ValueError("LLM_API_KEY is required; fill it in .env")
    return OpenAICompatibleChatClient(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=min(settings.llm_max_tokens, settings.llm_max_tokens_cap),
    )
