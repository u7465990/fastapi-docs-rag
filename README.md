# FastAPI 技术文档 RAG

面向 FastAPI 官方技术文档的 RAG 问答系统：支持中文提问检索英文文档、回答可追溯到 chunk → section → URL，并在检索不到时明确拒答。

项目签名是**版本感知的活知识库**：文档变更可感知、增量可索引、回答可标注适用版本。

> 当前状态：Greenfield。数据源已拉取，代码骨架尚未实现。

## 数据源

- 仓库：[`fastapi/fastapi`](https://github.com/fastapi/fastapi)（MIT）
- 快照：`7a0a3f0b286c6c865cf4398ea0e957a87b8bffa4`（2026-10-23），对应 FastAPI `0.162.0`
- 英文文档：`docs/en/docs/**/*.md`（156 个）
- 代码示例：`docs_src/**/*.py`（485 个）

本地数据位于 `data/raw/fastapi/`，已被 `.gitignore` 排除，不进入仓库。重新获取：

```bash
git clone --filter=blob:none --no-checkout https://github.com/fastapi/fastapi.git data/raw/fastapi
cd data/raw/fastapi
git sparse-checkout init --cone
git sparse-checkout set docs/en docs_src
git checkout master
```

## 文档

- [项目设计](docs/PROJECT_DESIGN.md)：架构、技术选型、FastAPI 数据解析、检索/生成/评估设计、迭代路线
- [RAG 原理](docs/RAG_PRINCIPLES.md)：各环节的底层机制、公式、失败模式与实现风险

## 计划技术栈

LangChain / LangGraph · Qdrant（dense + sparse hybrid）· BM25 + RRF · Cross-Encoder rerank · FastAPI + SSE · Redis · Vite + React + TypeScript · RAGAS · LangSmith · Docker Compose

## 后续计划

按 Phase 0 → 1 → 2 → 2.5 → 3 → 4 → 5 推进：先搭骨架与 dense baseline，再做 hybrid + rerank 基线，随后实现版本感知增量索引，最后完成 Agentic RAG、评估闭环与工程化。

## 本地环境

- Python 3.11（conda 环境 `langchain-rag`）
- Docker Desktop（仅运行 Qdrant + Redis）
- Node.js 24 / npm 11（前端）
- 访问 GitHub 需要开启加速器（本机 hosts 将 `github.com` 指向 `127.0.0.1`）
