"""Unit tests for the API skeleton."""

from fastapi.testclient import TestClient

from langchain_rag.api import app

client = TestClient(app)


def test_healthz() -> None:
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root_identifies_service() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["service"] == "fastapi-docs-rag"
    assert response.json()["version"] == "0.1.0"
