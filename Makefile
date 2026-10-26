.PHONY: install install-rag dev up down logs lint format test test-unit

install:
	pip install -e ".[dev]"

install-rag:
	pip install -e ".[dev,rag]"

dev:
	python -m langchain_rag.cli

up:
	docker compose up -d qdrant redis

down:
	docker compose down

logs:
	docker compose logs -f qdrant redis

lint:
	ruff check .
	mypy src

format:
	ruff format .

test:
	pytest

test-unit:
	pytest tests/unit
