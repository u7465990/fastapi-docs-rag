"""Tests for the chat client configuration and SSE parsing."""

import json

import httpx

from langchain_rag.config import Settings
from langchain_rag.generation.llm import OpenAICompatibleChatClient, make_chat_client


def test_max_tokens_is_capped() -> None:
    settings = Settings(
        _env_file=None,
        llm_api_key="test-key",
        llm_max_tokens=2048,
        llm_max_tokens_cap=512,
    )
    client = make_chat_client(settings)

    assert client.max_tokens == 512
    client.close()


def test_stream_parses_openai_sse_chunks() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["stream"] is True
        lines = [
            'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            "",
            'data: {"choices":[{"delta":{"content":" world"}}]}',
            "",
            "data: [DONE]",
            "",
        ]
        return httpx.Response(200, content="\n".join(lines).encode())

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = OpenAICompatibleChatClient(
        api_key="test-key",
        base_url="https://example.test/v1",
        model="test-model",
        client=http_client,
    )

    try:
        chunks = list(client.stream([{"role": "user", "content": "hi"}]))
    finally:
        http_client.close()

    assert chunks == ["Hello", " world"]
