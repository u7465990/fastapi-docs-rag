"""Unit tests for the API skeleton."""

from fastapi.testclient import TestClient

import langchain_rag.api as api
from langchain_rag.api import app
from langchain_rag.retrieval.dense import SearchHit

client = TestClient(app)


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
