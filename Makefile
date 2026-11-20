.PHONY: install install-rag dev up down logs lint format test test-unit check-parser check-chunker check-providers ingest update-index diff-report history-report build-history rebuild-index smoke-search load-search load-chat evaluate evaluate-generation web-build

UV ?= uv
# 依赖由 make install 同步；运行时不再 sync，避免把 rag extra 卸掉
RUN = $(UV) run --no-sync

install:
	$(UV) sync --extra dev

install-rag:
	$(UV) sync --extra dev --extra rag

dev:
	$(RUN) python -m fastapi_docs_rag.cli

up:
	docker compose up -d --build

web-build:
	cd apps/web && npm run build

down:
	docker compose down

logs:
	docker compose logs -f qdrant redis api

lint:
	$(RUN) ruff check .
	$(RUN) mypy src

format:
	$(RUN) ruff format .

test:
	$(RUN) pytest

test-unit:
	$(RUN) pytest tests/unit

check-parser:
	$(RUN) python scripts/check_parser.py

check-chunker:
	$(RUN) python scripts/check_chunker.py

check-providers:
	$(RUN) python scripts/check_providers.py

ingest:
	$(RUN) python scripts/ingest.py

update-index:
	$(RUN) python scripts/update_index.py

diff-report:
	$(RUN) python scripts/diff_report.py

history-report:
	$(RUN) python scripts/history_report.py

build-history:
	$(RUN) python scripts/build_version_collection.py --version 0.100.0 --collection fastapi_docs_v0100

rebuild-index:
	$(RUN) python scripts/rebuild_index.py

smoke-search:
	$(RUN) python scripts/smoke_search.py

load-search:
	$(RUN) python scripts/load_search.py

load-chat:
	$(RUN) python scripts/load_chat.py

evaluate:
	$(RUN) python scripts/evaluate.py

evaluate-generation:
	$(RUN) python scripts/evaluate_generation.py
