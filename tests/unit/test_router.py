"""Tests for version intent routing."""

from fastapi_docs_rag.retrieval.router import is_version_query, route_doc_type


def test_version_number_routes_to_release_notes() -> None:
    assert is_version_query("FastAPI 0.100.0 有什么变更？")
    assert route_doc_type("FastAPI 0.100.0 有什么变更？", None) == "release_notes"


def test_explicit_doc_type_wins() -> None:
    assert route_doc_type("FastAPI 0.100.0 有什么变更？", "tutorial") == "tutorial"


def test_normal_query_stays_unrouted() -> None:
    assert not is_version_query("Depends 怎么用？")
    assert route_doc_type("Depends 怎么用？", None) is None


def test_migration_question_is_not_automatically_version_routed() -> None:
    assert not is_version_query("on_event 怎么迁移到 lifespan？")
