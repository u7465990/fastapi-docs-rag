# FastAPI 技术文档 RAG

面向 FastAPI 官方技术文档的 RAG 问答系统：支持中文提问检索英文文档、回答可追溯到 chunk → section → URL，并在检索不到时明确拒答。

项目签名是**版本感知的活知识库**：文档变更可感知、增量可索引、回答可标注适用版本。

> 当前状态：Phase 0 环境与骨架已完成，解析器与父子切分器可用（含单元测试）；下一步是 embedding 与索引写入。

## 数据源

- 仓库：[`fastapi/fastapi`](https://github.com/fastapi/fastapi)（MIT）
- 快照：`7a0a3f0b286c6c865cf4398ea0e957a87b8bffa4`（2026-10-23），对应 FastAPI `0.162.0`
- 英文文档：`docs/en/docs/**/*.md`（156 个）
- 代码示例：`docs_src/**/*.py`（485 个）

本地数据位于 `data/raw/fastapi/`，已被 `.gitignore` 排除，不进入仓库。当前工作区是文件快照（不含 `.git`）；需要更新或做增量索引时，先按下面命令重建 sparse checkout：

```bash
git clone --filter=blob:none --no-checkout https://github.com/fastapi/fastapi.git data/raw/fastapi
cd data/raw/fastapi
git sparse-checkout init --cone
git sparse-checkout set docs/en docs_src fastapi
git checkout master
```

> `fastapi/` 只用于解析 `how-to/configure-swagger-ui.md` 中对 `fastapi/openapi/docs.py` 的代码 include，不索引其源码。

## 快速开始

环境由 [uv](https://docs.astral.sh/uv/) 管理（`uv >= 0.13`；CPython 3.12 由 uv 托管，仓库已提交 `uv.lock`）。

```bash
# 1. 依赖（首次）
uv sync --extra dev                 # core + pytest / ruff / mypy
uv sync --extra dev --extra rag     # 接 LangChain / LangGraph 时再装

# 2. 配置
cp .env.example .env                # Windows PowerShell: Copy-Item .env.example .env
# 然后填入 LLM_API_KEY、EMBEDDING_API_KEY

# 3. 依赖服务（只起 Qdrant + Redis）
docker compose up -d qdrant redis

# 4. 启动 API
uv run --no-sync python -m langchain_rag.cli
```

常用命令：

```bash
uv run --no-sync pytest                      # 全部测试
uv run --no-sync pytest tests/unit           # 仅单元测试
uv run --no-sync ruff check .
uv run --no-sync mypy src
uv run --no-sync python scripts/check_parser.py   # 解析器冒烟统计
uv run --no-sync python scripts/check_chunker.py  # 切分统计（token 分布 / 代码块 / 废弃）
```

`make install` / `make test` / `make lint` 是上面的薄封装。

启动后：

- API 文档：http://localhost:8000/docs
- 存活检查：http://localhost:8000/healthz
- 就绪检查：http://localhost:8000/readyz（需要 Qdrant、Redis 和模型 Key）

## 文档

- [项目设计](docs/PROJECT_DESIGN.md)：架构、技术选型、FastAPI 数据解析、检索/生成/评估设计、迭代路线
- [RAG 原理](docs/RAG_PRINCIPLES.md)：各环节的底层机制、公式、失败模式与实现风险

## 计划技术栈

LangChain / LangGraph · Qdrant（dense + sparse hybrid）· BM25 + RRF · Cross-Encoder rerank · FastAPI + SSE · Redis · Vite + React + TypeScript · RAGAS · LangSmith · Docker Compose

## 后续计划

按 Phase 0 → 1 → 2 → 2.5 → 3 → 4 → 5 推进：先搭骨架与 dense baseline，再做 hybrid + rerank 基线，随后实现版本感知增量索引，最后完成 Agentic RAG、评估闭环与工程化。

## 本地环境

- Python 3.12（uv 托管；`requires-python >=3.11,<3.13`）
- uv 0.13+，依赖版本锁定在 `uv.lock`
- Docker Desktop（仅运行 Qdrant + Redis）
- Node.js 24 / npm 11（前端）
- PyPI 默认走清华镜像（`pyproject.toml` 的 `[tool.uv.index]`），海外环境用 `--default-index https://pypi.org/simple` 覆盖
- 访问 GitHub 需要开启加速器（本机 hosts 将 `github.com` 指向 `127.0.0.1`）
