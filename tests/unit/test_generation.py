"""Tests for grounded answer generation."""

from fastapi_docs_rag.generation.chat import (
    REFUSAL,
    build_context,
    check_citations,
    generate_answer,
    retrieval_confidence,
    rewrite_query,
    select_context_hits,
)
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


def test_missing_citation_triggers_one_retry() -> None:
    class RetryClient:
        model = "retry-chat"

        def __init__(self) -> None:
            self.calls = 0

        def complete(self, messages):
            self.calls += 1
            return "先这样回答，没有引用。" if self.calls == 1 else "带引用的回答 [1]"

    client = RetryClient()

    answer = generate_answer("Depends 怎么用？", [_hit()], client)  # type: ignore[arg-type]

    assert answer == "带引用的回答 [1]"
    assert client.calls == 2


def test_follow_up_rewrite_uses_previous_user_question() -> None:
    history = [
        {"role": "user", "content": "on_event 怎么迁移？"},
        {"role": "assistant", "content": "用 lifespan。"},
    ]

    rewritten = rewrite_query("那它呢？", history)

    assert "on_event 怎么迁移？" in rewritten
    assert "那它呢？" in rewritten


def test_context_selection_dedupes_urls_and_respects_budget() -> None:
    hits = [_hit(1), _hit(2), _hit(3)]

    selected = select_context_hits(hits, max_tokens=8000, max_per_url=2)

    assert len(selected) == 2
    assert selected[0].url == _hit().url


def test_retrieval_confidence_prefers_rerank_score() -> None:
    hit = _hit()
    hit = SearchHit(
        **{
            **hit.__dict__,
            "dense_score": 0.72,
            "rerank_score": 0.31,
        }
    )

    assert retrieval_confidence([hit]) == 0.31


def test_citation_check_flags_invalid_index() -> None:
    check = check_citations("根据文档 [1]，这里还有 [9]。", [_hit()])

    assert check.invalid_indices == [9]
    assert check.valid_citations == 1
    assert check.warnings
