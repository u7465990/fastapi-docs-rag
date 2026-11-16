"""Tests for grounded answer generation."""

from fastapi_docs_rag.generation.chat import REFUSAL, build_context, generate_answer
from fastapi_docs_rag.retrieval.dense import SearchHit


class FakeChatClient:
    model = "fake-chat"

    def __init__(self) -> None:
        self.messages: list[dict[str, str]] = []

    def complete(self, messages):
        self.messages = list(messages)
        return "答案 [1]"


def _hit(index: int = 1) -> SearchHit:
    return SearchHit(
        chunk_id=f"chunk-{index}",
        doc_id="doc-1",
        title="Lifespan Events",
        text="child text",
        parent_text="Use lifespan instead of on_event.",
        url="https://fastapi.tiangolo.com/advanced/events/",
        section_path=["Lifespan Events", "Alternative Events (deprecated)"],
        anchor="alternative-events-deprecated",
        docs_version="0.162.0",
        source_ref="unknown",
        is_deprecated=True,
        deprecation_hint=False,
        content_type="prose",
        code_language=None,
        score=0.9,
    )


def test_context_has_numbered_sources_without_urls() -> None:
    context = build_context([_hit(1), _hit(2)])

    assert "[1]" in context
    assert "[2]" in context
    assert "Lifespan Events" in context
    assert "0.162.0" in context
    assert "已废弃" in context
    assert "https://" not in context


def test_generate_answer_calls_llm_with_system_and_user_messages() -> None:
    client = FakeChatClient()

    answer = generate_answer("on_event 怎么迁移？", [_hit()], client)

    assert answer == "答案 [1]"
    assert len(client.messages) == 2
    assert client.messages[0]["role"] == "system"
    assert "[1]" in client.messages[1]["content"]
    assert "on_event 怎么迁移？" in client.messages[1]["content"]


def test_no_hits_refuses_without_calling_llm() -> None:
    client = FakeChatClient()

    answer = generate_answer("something", [], client)

    assert answer == REFUSAL
    assert client.messages == []
