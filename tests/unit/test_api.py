"""Unit tests for the API skeleton."""

import pytest
from fastapi.testclient import TestClient

import langchain_rag.api as api
from langchain_rag.api import app
from langchain_rag.generation.chat import REFUSAL
from langchain_rag.generation.quota import QuotaExceeded
from langchain_rag.retrieval.dense import SearchHit

client = TestClient(app)


@pytest.fixture(autouse=True)
def _disable_http_security(monkeypatch):
    """Keep endpoint tests independent of Redis; security has its own tests."""
    monkeypatch.setattr(api, "_guard_http", lambda *args, **kwargs: None)
    monkeypatch.setattr(api, "_reserve_embedding", lambda query: None)


class FakeChatClient:
    model = "fake-chat"

    def complete(self, messages):
        assert messages
        return "答案 [1]"

    def stream(self, messages):
        assert messages
        yield "答案"
        yield " [1]"


class FakeQuota:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error

    def reserve(self, estimated_tokens: int) -> None:
        assert estimated_tokens > 0
        if self.error is not None:
            raise self.error


class FakeSessionStore:
    def __init__(self) -> None:
        self.messages: list[dict[str, str]] = []
        self.appended: list[tuple[str, str, str]] = []

    def load(self, session_id: str) -> list[dict[str, str]]:
        assert session_id
        return list(self.messages)

    def append_turn(self, session_id: str, question: str, answer: str) -> None:
        self.appended.append((session_id, question, answer))
        self.messages.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        )


class CapturingChatClient(FakeChatClient):
    def __init__(self) -> None:
        self.calls: list[list[dict[str, str]]] = []

    def complete(self, messages):
        self.calls.append(list(messages))
        return "答案 [1]"


class FakeRetriever:
    def search(self, query: str, **kwargs):
        assert query
        return [
            SearchHit(
                chunk_id="chunk-1",
                doc_id="doc-1",
                title="Dependencies",
                text="child text",
                parent_text="parent text",
                url="https://fastapi.tiangolo.com/tutorial/dependencies/",
                section_path=["Dependencies"],
                anchor=None,
                docs_version="0.162.0",
                source_ref="unknown",
                is_deprecated=False,
                deprecation_hint=False,
                content_type="prose",
                code_language=None,
                score=0.9,
            )
        ]


def test_healthz() -> None:
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root_identifies_service() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["service"] == "fastapi-docs-rag"
    assert response.json()["version"] == "0.1.0"


def test_search_endpoint_returns_hits(monkeypatch) -> None:
    monkeypatch.setattr(api, "_get_retriever", lambda: FakeRetriever())

    response = client.post(
        "/api/v1/search",
        json={"query": "Depends 怎么用？", "limit": 3},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "Depends 怎么用？"
    assert body["count"] == 1
    assert body["results"][0]["url"].endswith("/tutorial/dependencies/")
    assert body["results"][0]["score"] == 0.9


def test_search_endpoint_rejects_empty_query() -> None:
    response = client.post("/api/v1/search", json={"query": ""})

    assert response.status_code == 422


def test_chat_endpoint_returns_answer_and_sources(monkeypatch) -> None:
    monkeypatch.setattr(api, "_get_retriever", lambda: FakeRetriever())
    monkeypatch.setattr(api, "_get_chat_client", lambda: FakeChatClient())
    monkeypatch.setattr(api, "_get_quota_guard", lambda: FakeQuota())

    response = client.post(
        "/api/v1/chat",
        json={"query": "Depends 怎么用？", "limit": 3},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "答案 [1]"
    assert body["model"] == "fake-chat"
    assert len(body["sources"]) == 1
    assert body["sources"][0]["index"] == 1
    assert body["sources"][0]["url"].endswith("/tutorial/dependencies/")


def test_chat_stream_endpoint_emits_sse_events(monkeypatch) -> None:
    monkeypatch.setattr(api, "_get_retriever", lambda: FakeRetriever())
    monkeypatch.setattr(api, "_get_chat_client", lambda: FakeChatClient())
    monkeypatch.setattr(api, "_get_quota_guard", lambda: FakeQuota())

    with client.stream(
        "POST",
        "/api/v1/chat/stream",
        json={"query": "Depends 怎么用？", "limit": 3},
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert "event: sources" in body
    assert "event: token" in body
    assert "event: usage" in body
    assert "event: done" in body
    assert "答案" in body
    assert "/tutorial/dependencies/" in body


def test_chat_endpoint_uses_and_updates_session_history(monkeypatch) -> None:
    store = FakeSessionStore()
    chat_client = CapturingChatClient()
    monkeypatch.setattr(api, "_get_retriever", lambda: FakeRetriever())
    monkeypatch.setattr(api, "_get_chat_client", lambda: chat_client)
    monkeypatch.setattr(api, "_get_quota_guard", lambda: FakeQuota())
    monkeypatch.setattr(api, "_get_session_store", lambda: store)

    first = client.post(
        "/api/v1/chat",
        json={"query": "Depends 怎么用？", "session_id": "session-123"},
    )
    second = client.post(
        "/api/v1/chat",
        json={"query": "那 Annotated 呢？", "session_id": "session-123"},
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["session_id"] == "session-123"
    assert len(chat_client.calls) == 2
    second_messages = chat_client.calls[1]
    assert second_messages[0]["role"] == "system"
    assert {"role": "user", "content": "Depends 怎么用？"} in second_messages
    assert {"role": "assistant", "content": "答案 [1]"} in second_messages
    assert store.appended[0][0] == "session-123"
    assert store.appended[1][1] == "那 Annotated 呢？"


def test_chat_endpoint_returns_429_when_quota_exceeded(monkeypatch) -> None:
    monkeypatch.setattr(api, "_get_retriever", lambda: FakeRetriever())
    monkeypatch.setattr(
        api,
        "_get_chat_client",
        lambda: (_ for _ in ()).throw(AssertionError("LLM must not be called")),
    )
    monkeypatch.setattr(
        api,
        "_get_quota_guard",
        lambda: FakeQuota(QuotaExceeded("quota exhausted")),
    )

    response = client.post("/api/v1/chat", json={"query": "Depends 怎么用？"})

    assert response.status_code == 429
    assert "quota" in response.json()["detail"]


def test_chat_endpoint_refuses_when_no_context(monkeypatch) -> None:
    class EmptyRetriever:
        def search(self, query: str, **kwargs):
            return []

    monkeypatch.setattr(api, "_get_retriever", lambda: EmptyRetriever())
    monkeypatch.setattr(
        api,
        "_get_chat_client",
        lambda: (_ for _ in ()).throw(AssertionError("LLM must not be called")),
    )

    response = client.post("/api/v1/chat", json={"query": "unknown question"})

    assert response.status_code == 200
    assert response.json()["answer"] == REFUSAL
    assert response.json()["sources"] == []
