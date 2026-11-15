# FastAPI 技术文档 RAG

面向 FastAPI 官方技术文档的 RAG 问答系统：支持中文提问检索英文文档、回答可追溯到 chunk → section → URL，并在检索不到时明确拒答。

项目签名是**版本感知的活知识库**：文档变更可感知、增量可索引、回答可标注适用版本。

> 当前状态：Phase 1 进行中。解析、父子切分、embedding、Qdrant indexer、dense retrieval（`/api/v1/search`）、RAG 问答（`/api/v1/chat`）、SSE 流式回答（`/api/v1/chat/stream`）、Docker API 和 Nginx 静态托管的聊天页已可用；已有 30 条检索评估和 HTTP 限流/额度锁；下一步是多轮会话与生成质量评估。

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

release notes 默认只把最近 20 个版本切进普通索引（`RELEASE_NOTES_LATEST_VERSIONS`）；历史版本后续通过 git tag 重建，避免 changelog 主导检索。

## 快速开始

环境由 [uv](https://docs.astral.sh/uv/) 管理（`uv >= 0.13`；CPython 3.12 由 uv 托管，仓库已提交 `uv.lock`）。

```bash
# 1. 依赖（首次）
uv sync --extra dev                 # core + pytest / ruff / mypy
uv sync --extra dev --extra rag     # 接 LangChain / LangGraph 时再装

# 2. 配置
cp .env.example .env                # Windows PowerShell: Copy-Item .env.example .env
# 然后填入 LLM_API_KEY、EMBEDDING_API_KEY
uv run --no-sync python scripts/check_providers.py   # 先验证 Key / 模型 / 端点

# 3. 启动 Qdrant + Redis + API（首次会构建 API 镜像）
docker compose up -d --build
docker compose logs -f api

# 4. 建索引（需要 EMBEDDING_API_KEY，且 Qdrant 已启动）
uv run --no-sync python scripts/ingest.py

# 5. 用固定问题跑检索 smoke
uv run --no-sync python scripts/smoke_search.py

# 6. 构建前端并由 nginx 托管（生产模式，不用 vite dev）
cd apps/web
npm install
npm run build
cd ../..
docker compose up -d --build web
# 打开 http://localhost:8080
```

开发时仍可单独跑 `cd apps/web && npm run dev`，Vite 会把 `/api` 代理到 `127.0.0.1:8000`；这个 5173 端口只用于本机开发，不要暴露到公网。

API 默认跑在 Docker 里。调试时也可以停掉 api 容器，改用宿主机：

```bash
docker compose stop api
uv run --no-sync python -m langchain_rag.cli
```

索引默认写入临时 collection `fastapi_docs_build_<ts>`，成功后原子切换 alias `fastapi_docs_current`；旧 build 保留用于回滚。只想看切分和版本字段时加 `--dry-run`，不调用 embedding。

### LLM 额度锁

API 用 Redis 做应用层每日额度锁，调用 LLM 前先预占 token，超限直接返回 429，不再请求模型：

```dotenv
LLM_MAX_TOKENS=1024
LLM_MAX_TOKENS_CAP=512
LLM_DAILY_REQUEST_LIMIT=50
LLM_DAILY_TOKEN_BUDGET=300000
```

`LLM_MAX_TOKENS_CAP` 是单次硬上限；后两项是每天请求数 / token 预算。Redis 不可用时返回 503，不会绕过额度直接调用模型。

### HTTP 限流、embedding 额度和 access token

```dotenv
RATE_LIMIT_PER_MINUTE=30
EMBEDDING_DAILY_REQUEST_LIMIT=500
EMBEDDING_DAILY_TOKEN_BUDGET=200000
API_ACCESS_TOKEN=
```

- `/api/v1/search` 和 `/api/v1/chat` 都按 IP 做每分钟限流；超限返回 429。
- 每次检索/问答都会消耗一次 embedding 额度；embedding 超限同样返回 429。
- `API_ACCESS_TOKEN` 为空时不做鉴权（本地开发默认）；设置后请求需要：
  - `Authorization: Bearer <token>`，或
  - `X-API-Key: <token>`
- 前端如果要在开启 token 后使用，构建时设置 `apps/web/.env`：

```dotenv
VITE_API_TOKEN=同一个token
```

注意：前端 bundle 里的 token 用户可见，只是演示门禁，不是真正的用户登录。真正上线需要账号体系。

Nginx 会设置 `X-Real-IP`，限流按这个 IP 计数；直接暴露 API 时该 header 可被伪造，所以生产必须只让 API 通过 Nginx 访问。

### 流式回答

`POST /api/v1/chat/stream` 返回 `text/event-stream`，顺序固定为：

```text
sources -> token* -> usage -> done
```

- `sources` 先到，前端可以先显示来源
- `token` 是增量文本
- `usage` 当前为 `null` 计数（OpenAI 兼容流不一定返回 usage）
- `done` 表示结束
- 生成异常会发 `error`，不会静默截断

前端用 `fetch` + `ReadableStream` 解析，不能用原生 `EventSource`（它只支持 GET）。Nginx 的 `/api` 已关闭 `proxy_buffering`，否则流会被整段缓冲。

常用命令：

```bash
uv run --no-sync pytest                      # 全部测试
uv run --no-sync pytest tests/unit           # 仅单元测试
uv run --no-sync ruff check .
uv run --no-sync mypy src
uv run --no-sync python scripts/check_parser.py   # 解析器冒烟统计
uv run --no-sync python scripts/check_chunker.py  # 切分统计（token 分布 / 代码块 / 废弃）
uv run --no-sync python scripts/load_search.py    # 8 路并发检索延迟
uv run --no-sync python scripts/evaluate.py       # 30 条 golden set 的检索基线
```

`make install` / `make test` / `make lint` 是上面的薄封装。

启动后：

- API 文档：http://localhost:8000/docs
- 存活检查：http://localhost:8000/healthz
- 就绪检查：http://localhost:8000/readyz（需要 Qdrant、Redis 和模型 Key）

## 文档

- [项目设计](docs/PROJECT_DESIGN.md)：架构、技术选型、FastAPI 数据解析、检索/生成/评估设计、迭代路线
- [RAG 原理](docs/RAG_PRINCIPLES.md)：各环节的底层机制、公式、失败模式与实现风险
- [2026-11-14 工作记录](docs/WORKLOG_2026-11-14.md)：当天的命令、假设、验证结果与决策
- [名词与原理注解](docs/WORKLOG_2026-11-14_EXPLAINED.md)：工作记录中术语、行为、历史改动与设计原理的解释
- [检索评估基线](docs/EVALUATION.md)：30 条 golden set、指标、迭代记录与局限
- [Bad Cases](docs/BAD_CASES.md)：已实测的排序/标注问题与修复记录

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
