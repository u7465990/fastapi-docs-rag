# LangChain 技术文档 RAG 问答系统 · 方案 v0.3

> 版本：v0.3（2026-10-18）  
> 状态：环境已体检；前端选型与项目签名已定，其余待确认  
> 目标：做一个**可演示、可评估、可部署**的版本感知技术文档 RAG 系统，避免沦为“教程复现”。  
> 项目签名：**版本感知的活知识库**——变更可感知、增量可索引、回答可标注版本；hybrid + rerank 是基线，评估与 bad case 闭环是证据。

---

## 0. TL;DR

**结论：当前环境完全可行，建议立即开工。**

- 机器：Windows 10，i7-11390H（4C/8T），16GB RAM，D 盘剩余约 46GB。
- Python 3.11.11（conda），已有 LangChain 1.3.14、LangGraph 1.2.10、FastAPI、Chroma、sentence-transformers、PyTorch 等；但当前环境 `comp4528` 是课程环境，建议新建专用环境。
- Docker Desktop 29.7.2 已运行，Compose 可用；Docker Hub 直连超时，但已配置国内镜像，**Qdrant 镜像已成功拉取**。
- Node 24 / npm 11 可用，前端采用 Vite + React。
- GPU 只有 MX450 2GB，**不适合跑本地大模型**；LLM 使用云端 API，本地只跑轻量 embedding / reranker（可选）。
- 目前没有项目用的 LLM / Embedding API Key（环境里的 Anthropic 变量是 Claude Code 自己的，不建议挪用），需要单独准备。

**推荐最小可行栈：**

`LangChain + LangGraph + Qdrant（Docker）+ Redis + FastAPI + SSE 流式 + Vite + React + RAGAS + LangSmith + Docker Compose + GitHub Actions`

**项目签名（唯一深挖点）：版本感知的活知识库。** 只做到 hybrid + rerank 只能证明“会用框架”，很容易被当成教程复现；本项目额外回答“LangChain 发布新版本后，知识库如何感知、增量更新，并区分新旧 API”。

**建议节奏：** baseline（约 1 周）→ hybrid + rerank（必备基线）→ **版本感知增量索引（签名，3~4 天）** → LangGraph 自适应 RAG → 评估与 bad case 闭环 → 工程化收尾。Phase 6 加分项默认不做。

---

## 1. 项目定位

### 1.1 要解决的问题

面向 **LangChain / LangGraph 官方技术文档** 的问答系统，回答诸如：

- “LangChain 里怎么给对话加记忆？给代码示例。”
- “LangGraph 的 interrupt 和 checkpointer 怎么配合使用？”
- “RAG 的 ParentDocumentRetriever 和普通 splitter 有什么区别？”
- 中文提问，也能检索英文文档并用中文回答。
- 每个回答必须给出**可点击的文档出处**，检索不到时明确说“不知道”，不编造。

### 1.2 为什么这个选题值得做

| 维度 | 价值 |
|---|---|
| 数据质量 | 官方文档结构清晰、有代码块、有稳定 URL，适合做高质量 RAG |
| 评估可行 | 能构造 golden set，检索和生成都能量化评估 |
| 技术覆盖 | 采集、切分、embedding、混合检索、rerank、Agentic RAG、评估、部署全链路 |
| 面试友好 | 有明确指标、有架构取舍、有踩坑和优化过程，能讲 20 分钟 |
| 可扩展 | 后续可加 JS 文档、MCP、多租户、GraphRAG、多模态 |
| 叙事优势 | “用 LangChain 文档服务 LangChain 开发者”的自指故事，天然有记忆点 |
| 差异化抓手 | 官方文档更新频繁，版本时效性是一个真实、可量化、可演示的工程问题 |

> **元叙事**：用 LangChain / LangGraph 文档，构建一个服务 LangChain 开发者、并能跟上文档版本演进的 RAG 工具。
>
> **反教程声明**：baseline 只是对照组，不作为项目成果。最终只押一个签名能力——版本感知的活知识库；hybrid + rerank 是必备基线；评估与 bad case 闭环是证明手段。

### 1.3 范围

**v1 必做（主线）：**

- 数据源：LangChain Python 文档 + LangGraph 文档（以最终确认范围为准）。
- 采集 → 变更检测 → 清洗 → 父子切分 → 索引 → 混合检索 → rerank → 生成 → 引用。
- **项目签名：版本感知的增量更新管道**——Git commit diff / `content_sha256` 变更检测、只重建变更页面、版本元数据与废弃标记、alias 切换与回滚。
- **混合检索（dense + BM25 + RRF）+ rerank**：这是必备基线，不是加分项。
- 多轮对话、查询重写、无答案兜底、版本归因。
- 流式输出、来源展示、反馈收集。
- 100+ 条 golden set（含版本差异 / 废弃 API / 时效性题），RAGAS + 检索指标 + 版本指标。
- `docs/BAD_CASES.md`：10~20 个 bad case 的“现象 → 根因 → 干预 → 前后指标”。
- FastAPI + Docker Compose + 测试 + CI。

**v1 不做（不要挤占主线）：**

- 本地部署大模型。
- 训练 / 微调模型。
- MCP Server、GraphRAG、多租户权限系统、多文档源（仅作为 Phase 6 加分项）。
- 通用网页搜索、实时联网问答。
- 复杂前端设计系统、PWA / 多端适配。

---

## 2. 环境体检结论

### 2.1 实测结果

| 项目 | 实测 | 判断 |
|---|---|---|
| 工作目录 | `D:\ANU\LangChain-rag`，当前为空 | ✅ 可直接初始化 |
| Python | 3.11.11（`E:\Anaconda3\envs\comp4528`） | ✅ 版本合适 |
| 已有包 | LangChain 1.3.14 / LangGraph 1.2.10 / FastAPI 0.116.1 / Chroma 1.5.9 / SentenceTransformers 5.6.1 / torch 2.6.0 | ✅ 可参考，但建议新建环境 |
| Conda | 24.11.2，已有多个环境 | ✅ 建议新建 `langchain-rag` |
| uv / poetry | 未安装 | ⚠️ 可选，`pip install uv` 即可 |
| Docker | 29.7.2，daemon 正常，Compose v5.5.0 | ✅ |
| Docker Hub | 直连超时，已配置 daocloud / 163 等镜像 | ✅ 镜像拉取实测成功 |
| Qdrant 镜像 | `qdrant/qdrant:latest` 已拉取（291MB） | ✅ 可直接起服务 |
| Redis 镜像 | `redis:7` / `redis:7-alpine` 本地已有 | ✅ |
| Node.js | v24.18.0 / npm 11.16.0 | ✅ 前端使用 Vite + React |
| Git | 2.49.0 | ✅ |
| CPU | i7-11390H，4 核 / 8 线程 | ✅ 够用 |
| 内存 | 15.7GB | ⚠️ 够用，但不要同时跑重服务 |
| 磁盘 | D 盘剩余 46.3GB | ✅ |
| GPU | NVIDIA MX450，2GB VRAM（空闲约 1.9GB） | ❌ 不能跑本地 LLM |
| Ollama | 未安装 | — |
| 网络 | PyPI 200，HuggingFace 200 | ✅ |
| API Key | 仅有 `ANTHROPIC_*`（Claude Code 使用），无项目独立 Key | ⚠️ 需准备 |

### 2.2 环境结论

**可以做，而且是“客户端 + 云端模型 + 本地容器”的理想小项目形态。**

需要注意 5 个约束：

1. **LLM 必须走云端 API。** MX450 2GB 跑 7B 模型不现实，也没必要。用 DeepSeek / 通义千问 / OpenAI / Claude 均可。
2. **不要把所有中间件都塞进 Docker。** 16GB 内存下，建议只跑 Qdrant + Redis；FastAPI、采集脚本、评估脚本跑在宿主机。Langfuse 全家桶（Postgres + ClickHouse + Redis + MinIO）后续再考虑，先用 LangSmith 云端的免费额度。
3. **新建专用 conda 环境**，避免污染 `comp4528` 课程环境：
   ```bash
   conda create -n langchain-rag python=3.11 -y
   conda activate langchain-rag
   pip install -U pip uv
   ```
4. **Docker Desktop 建议分配 6GB 左右内存**，并确认 Qdrant / Redis 端口未被占用（6333、6334、6379）。
5. **API Key 不要复用 Claude Code 的环境变量。** 那是 CLI 的会话凭证，不适合应用长期使用。建议单独申请一个项目 Key，写入 `.env`。

### 2.3 建议的本地运行拓扑

```text
宿主机（Windows）
├─ Python 进程：FastAPI :8000
├─ Node 进程：Vite dev server :5173（生产为静态文件）
├─ 采集 / 索引 / 评估脚本
└─ Docker Desktop
   ├─ qdrant/qdrant        :6333 :6334   （向量库）
   └─ redis:7              :6379          （缓存 / 会话）
```

暂不启动：Langfuse、ClickHouse、MinIO、Kafka、Airflow。

---

## 3. 总体架构

### 3.1 离线索引链路

```mermaid
flowchart LR
    A[官方文档 Git 仓库 / 站点] --> B[采集器<br/>GitLoader + sitemap]
    B --> C{变更检测<br/>commit SHA / content_sha256}
    C -->|变更 / 新增| D[解析<br/>HTML → Markdown / MDX 直读]
    C -->|未变更| M[(manifest<br/>增量与版本状态)]
    D --> E[清洗 / 去重 / 版本元数据]
    E --> F[父子切分<br/>标题层级 + 代码块保护]
    F --> G[Embedding<br/>dense]
    F --> H[BM25<br/>sparse]
    G --> I[(Qdrant<br/>dense + sparse)]
    H --> I
    E --> M
    I --> J[校验 + alias 原子切换]
```

### 3.2 在线问答链路

```mermaid
flowchart TD
    U[用户] --> API[FastAPI<br/>/chat/stream]
    API --> G[LangGraph 编排]
    G --> Q[查询理解<br/>多轮改写 / 路由 / 版本意图]
    Q --> R[混合检索<br/>dense + BM25 → RRF]
    R --> RR[Rerank<br/>CrossEncoder]
    RR --> GEN[生成答案<br/>约束 + 引用]
    GEN --> CHK[引用/忠实度/版本归因检查]
    CHK --> API
    R <--> QD[(Qdrant)]
    API <--> RC[(Redis 缓存)]
    G -.trace.-> LS[LangSmith]
    API --> LOG[结构化日志 / 指标]
```

### 3.3 关键设计原则

- **检索和生成解耦**：检索器可以单独评估、单独替换。
- **先 baseline 再优化**：每一层优化都要有评估数据支撑，避免“堆技术”。
- **所有回答可追溯**：chunk → section → URL，前端可点击。
- **失败可降级**：rerank 失败走 RRF；LLM 超时走缓存；检索为空明确拒答。
- **索引可版本化**：临时 collection 校验后原子切 alias，出问题可回滚。
- **版本可感知**：拆分 `docs_version` / `source_ref` / `index_build_id`；废弃页面显式标记，查询可按版本过滤。
- **单点深挖**：签名能力是版本感知的活知识库；hybrid + rerank 是必备基线，不作为差异化卖点。

---

## 4. 技术选型

### 4.1 核心选型表

| 层 | 选择 | 作用 | 备选 | 优先级 |
|---|---|---|---|---|
| 语言 | Python 3.11 | 主语言 | — | P0 |
| 编排 | LangChain 1.x + LangGraph 1.x | 组件 + 有状态工作流 | 纯 SDK | P0 |
| LLM | DeepSeek / Qwen / OpenAI / Claude（可切换） | 答案生成、查询改写、评分 | 任意 OpenAI 兼容模型 | P0 |
| Embedding | Qwen `text-embedding-v4` / OpenAI `text-embedding-3-small` / 本地 BGE-M3 | 向量化 | Cohere、Jina | P0 |
| 向量库 | Qdrant（Docker） | dense + sparse 混合检索、过滤 | pgvector、Milvus、Chroma | P0 |
| 稀疏检索 | Qdrant BM25 sparse vector（FastEmbed） | 关键词 / 代码符号召回 | rank_bm25、Elasticsearch | P0 |
| 融合 | RRF（Reciprocal Rank Fusion） | 多路召回融合 | DBSF | P0 |
| Rerank | BGE reranker（本地 CPU）或 API rerank | 精排 | Cohere Rerank、GTE rerank | P0 |
| 缓存 | Redis 7 | 答案缓存、检索缓存、会话 | 内存 LRU | P1 |
| API | FastAPI + Pydantic v2 + Uvicorn | HTTP / SSE | Flask | P0 |
| 前端 | Vite + React + TypeScript + Tailwind + shadcn/ui | 聊天界面、来源展示、SSE 消费 | Next.js、Streamlit | P1 |
| 可观测 | LangSmith（先）→ Langfuse（可选） | Trace、token、实验 | OpenTelemetry | P0 |
| 评估 | RAGAS + 自定义检索指标 + LLM-as-Judge | 质量闭环 | DeepEval、TruLens | P0 |
| 测试 | pytest + pytest-asyncio + ruff + mypy | 单测 / 集成 / 静态检查 | — | P0 |
| 部署 | Docker + Docker Compose | 一键起服务 | 云 VM / K8s | P0 |
| CI | GitHub Actions + pre-commit | 自动检查 | GitLab CI | P0 |
| MCP | FastMCP（可选） | 把 RAG 暴露成 Agent 工具 | — | P2 |

### 4.2 LLM 选择建议

项目代码通过 `init_chat_model` / `ChatOpenAI(base_url=...)` 抽象，**不要写死厂商**。

建议默认使用 **OpenAI 兼容协议** 的国内模型（DeepSeek / 通义千问），原因：

- 延迟和价格友好；
- 对中文提问友好；
- `langchain-openai` 直接兼容；
- 后续切 OpenAI / Claude 只改配置。

可选组合：

| 用途 | 建议模型 | 说明 |
|---|---|---|
| 答案生成 | 中高质量对话模型 | 质量优先，温度低 |
| 查询改写 / 路由 / 评分 | 小模型或同模型低 token | 控制成本 |
| Embedding | 支持中英双语的 embedding | 与生成模型解耦 |
| Rerank | 本地 cross-encoder 或 API | 先本地，量大再换 API |
| LLM-as-Judge | 与生成模型不同的模型 | 降低自评偏差 |

> 注意：如果选用 DeepSeek 作为主 LLM，它本身没有 embedding 服务，需要单独选择 Qwen / OpenAI / Cohere / 本地 BGE。把 LLM 和 Embedding 分开配置。

### 4.3 Embedding 选择建议

| 方案 | 优点 | 缺点 | 建议 |
|---|---|---|---|
| Qwen `text-embedding-v4` | 中英双语好、API 稳定 | 需要 DashScope Key | **首选** |
| OpenAI `text-embedding-3-small` | 生态成熟、便宜 | 中文一般、需要外网 | 备选 |
| 本地 `BAAI/bge-m3` | 免费、可离线、多语言 | CPU 编码慢、占内存 | 离线/成本敏感时 |
| 本地 `bge-small-zh-v1.5` | 轻量、CPU 可跑 | 仅中文优化 | 不推荐用于英文文档 |

**向量维度一旦建库就不要随意改**，换 embedding 模型 = 重建 collection。建议把 `embedding_model`、`dimension` 写进索引 manifest。

### 4.4 Reranker 选择建议

- 本地：`BAAI/bge-reranker-base`（约 278M，CPU 可接受，top 20 精排）。
- 本地轻量：FlashRank（更小、更快，效果略弱）。
- API：Cohere Rerank / 阿里 GTE rerank（省机器，按量计费）。
- 策略：**先检索 top 20，再 rerank 取 top 6~8**；不要对全库 rerank。

### 4.5 为什么选 Qdrant

- 原生支持 **dense + sparse 混合检索** 和 RRF，正好覆盖 JD 里的 hybrid search。
- 支持 payload 过滤：按产品、语言、版本、文档类型过滤。
- Docker 单容器即可跑，资源占用低。
- Python client 成熟，LangChain 有官方 `QdrantVectorStore`。
- 后续可平滑扩到 Milvus / pgvector，面试时也能讲清取舍。

Chroma 作为本地快速验证的备选；Postgres + pgvector 作为“想要 SQL + 向量一体”的备选。

### 4.6 前端选型（v0.2 已定）

选择 **Vite + React + TypeScript**，配合 Tailwind CSS + shadcn/ui：

- 项目是纯聊天 UI，无 SEO / 首屏渲染需求，不需要 Next.js 的 SSR 与服务端能力。
- FastAPI 已承担后端，前端只消费 `/api/v1/chat/*`，SPA 足够。
- 构建快、静态部署简单，可由 nginx 托管或并入 FastAPI 静态目录。
- 若后续确有多端（小程序）需求，再评估 Taro（同为 React 语法）。

关键实现约束：`/api/v1/chat/stream` 是 **POST** 接口，原生 `EventSource` 只支持 GET 且不能带请求体，前端必须用 `fetch` + `ReadableStream` 解析 SSE（可用 `eventsource-parser` 处理分帧），并按 `sources → token*N → usage → done / error` 事件顺序渲染。

### 4.7 为什么是 LangChain / LangGraph，而不是 LlamaIndex

这是这个选题最容易被追问的“自指困境”，必须提前准备：

- 本项目的核心不是“文档解析”，而是 **LangGraph 有状态编排**：路由、相关性评分、重试上限、引用/版本检查、checkpointer。LlamaIndex 的摄取与检索抽象很强，但不能替代这条需要显式控制的 Agentic RAG 工作流。
- 版本感知的增量索引、混合检索、rerank 都是**与框架无关的工程设计**；检索层抽象成接口，换框架不改变架构。
- 面试时不要回答“因为我只会 LangChain”。正确叙事是：**选 LangChain / LangGraph 是为了展示可控的编排与工程化能力，而不是因为只会调用它的 API。**

### 4.8 反教程声明（项目边界）

- “跑通 LangChain + 向量库 + 文档问答”只是 baseline，不作为成果。
- 主线只保留**一个**签名能力：版本感知的增量索引；hybrid + rerank 是必备基线；评估与 bad case 闭环是证明手段。
- MCP、GraphRAG、多租户、多文档源、PWA / 多端均**不进入 v1 主线**，只在 Phase 6 且主线完成后考虑。

---

## 5. 数据采集与清洗

### 5.1 数据源

v1 建议只覆盖：

- LangChain Python 文档
- LangGraph 文档

可选扩展：

- LangChain JS 文档
- LangChain API Reference
- LangSmith 文档

具体入口以开工时实际可访问的站点为准，优先顺序：

1. **官方文档源码仓库（Markdown / MDX，首选）**：保留代码块和标题结构，最干净；能拿到 Git commit SHA，天然支持变更检测与可复现构建。
2. **站点 `sitemap.xml`**：枚举页面，适合抓 HTML。
3. **从导航链接爬取**：前两者不可用时兜底，限制深度和域名。

> 文档站点会改版，采集器必须做成“可配置入口 + 可重跑 + 有测试”，不要把 URL 写死在业务代码里。
>
> 采集入口必须产出 `source_ref`（Git commit SHA 或页面 ETag / Last-Modified）。没有它就无法做真正的版本感知，只能退化为内容哈希。

### 5.2 采集要求

- 遵守 `robots.txt`。
- 限速（例如 1 req/s），带明确的 User-Agent。
- 原始 HTML / Markdown 落盘缓存，失败可重试。
- **变更检测优先级**：Git commit diff > sitemap `lastmod` > `content_sha256`。
- **定时刷新**：默认每日一次（可配置），每次生成 diff 报告（新增 / 修改 / 删除 / 疑似废弃）。
- 增量更新：只重新处理变更页面；未变更页面的 chunk 与 embedding 直接复用。
- 规范化 URL：去掉 query、锚点、尾斜杠，避免重复页面。
- 同一内容出现在多个 URL 时，保留 canonical 一份。
- 版本字段必须从采集阶段就写入：`docs_version`、`source_ref`、`fetched_at`、`is_deprecated`。

### 5.3 清洗

- 去掉导航、页脚、侧边栏、cookie 提示、广告。
- HTML → Markdown，保留：
  - 标题层级（H1~H4）
  - 代码块及语言标记
  - 表格
  - 提示框 / 警告块
- 统一代码块格式，去掉行号（如果有）。
- 提取元数据：

```json
{
  "doc_id": "hash(canonical_url)",
  "url": "https://...",
  "canonical_url": "https://...",
  "title": "How to add memory",
  "section_path": ["Guides", "Memory", "Short-term memory"],
  "product": "langchain",
  "language": "python",
  "docs_version": "v0.4",
  "source_ref": "git:<commit-sha>",
  "fetched_at": "2026-10-18T10:00:00Z",
  "is_deprecated": false,
  "doc_type": "guide",
  "updated_at": "2026-10-01",
  "content_sha256": "...",
  "index_build_id": "20261018T100000Z"
}
```

> 版本字段不再混用：`docs_version` 表示文档内容对应的库 / 站点版本（无法判断时写 `"unknown"`），`source_ref` 用于变更检测，`index_build_id` 标识本次索引构建。三者用途不同，必须分开。

### 5.4 数据量预估

- 页面：约 1k~3k 页（取决于范围）。
- chunk：约 2w~6w。
- 单机 Qdrant + 本地磁盘完全够用。
- Embedding 成本很低（几十万到几百万 token），可放心先用 API。

---

## 6. 切分与索引

### 6.1 父子切分（Parent-Child / Small-to-Big）

**不要把整页直接切碎后只返回碎片。** 推荐：

- **Parent chunk**：约 1200~1600 tokens，保留完整小节。
- **Child chunk**：约 300~500 tokens，用于 embedding 和召回。
- 命中 child 后，返回对应 parent 给 LLM，兼顾召回精度和上下文完整性。
- 相邻 child 可加 50~80 tokens overlap，避免边界信息丢失。

### 6.2 代码块处理

技术文档 RAG 的关键难点：

- 代码块不能被从中间切断。
- 长代码块（> 800 tokens）按空行 / 函数边界切分，但保留语言标记和所属标题。
- 代码块的 metadata 增加 `content_type: "code"`、`code_language`。
- 检索时可以把“包含代码示例”作为一个过滤/加权条件。

### 6.3 标题上下文注入

每个 child 在 embedding 前拼接上下文头：

```text
[LangChain > Guides > Memory > Short-term memory]
<chunk 正文>
```

这是成本最低、收益通常最明显的优化之一。

可选进阶：**Contextual Retrieval**——用 LLM 为每个 chunk 生成 1~2 句上下文再 embedding。效果可能更好，但有额外成本，先评估再决定是否全量启用。

### 6.4 Qdrant Collection 设计

```text
Collection: langchain_docs_current（alias → langchain_docs_build_<ts>）
Named vectors:
  - dense:  1024/1536 维（由 embedding 模型决定）
  - sparse: BM25
Payload:
  - chunk_id, parent_id, doc_id
  - url, title, section_path
  - product, language, doc_type
  - docs_version, source_ref, fetched_at, updated_at, is_deprecated
  - content_type, code_language
  - text, parent_text（或 parent 存外部 docstore）
  - content_sha256, index_build_id
```

索引流程：

1. 采集 → 变更检测 → 只解析变更页面 → 清洗 → chunk → 生成稳定 ID。
2. 复用 embedding 缓存；仅对变更 chunk 调用 embedding + sparse 编码。
3. 写入临时 collection `langchain_docs_build_<ts>`（全量点写入，但只重算变更向量）。
4. 校验数量、版本覆盖率、抽样检索。
5. 原子切换 alias `langchain_docs_current` → 新 collection。
6. 旧 collection 保留一个版本，方便回滚；输出 diff 报告与陈旧度指标。

### 6.5 幂等与增量

- chunk ID 由 `doc_id + section_hash + chunk_index` 生成。
- 内容哈希不变则跳过 embedding。
- 删除页面时同步删除旧 chunk（按 `doc_id` 前缀）；无法确认是否删除时先标记 `is_deprecated`，保留一个刷新周期后再物理删除。
- 废弃页面不立即删除：打 `is_deprecated=true`，检索时降权或按需过滤，保证历史问题仍可回答并给出迁移提示。
- manifest 记录：`source_ref`、构建时间、页面数、chunk 数、变更 / 新增 / 删除页面、重嵌入 chunk 数、embedding 模型 / 维度、耗时、失败列表。

### 6.6 版本感知的增量更新（项目签名能力）

目标：文档变更后，系统能**感知 → 增量重建 → 可追溯 → 可回滚**，并把“陈旧度”变成可量化指标。

流程：

1. **抓取源版本**：`git fetch` 后取目标分支 commit SHA；或抓 sitemap 的 `lastmod` / ETag。
2. **生成变更集**：对比上次 manifest，产出 `added / modified / deleted / maybe_deprecated` 列表。
3. **只重建变更部分**：变更页面重新解析、切分、embedding；未变更 chunk 复用内容与向量缓存。
4. **写入新 collection**：`langchain_docs_build_<ts>`；全量点写入，但 embedding 只算变更部分（小数据集下比原地更新更安全）。
5. **校验**：点数量、版本覆盖率、抽样检索、关键 golden 问题回归。
6. **原子切换 alias**：`langchain_docs_current` 指向新 collection；旧版本保留用于回滚。
7. **产出报告**：diff 报告、陈旧度、重嵌入比例、构建耗时，写入 `data/manifests/` 与 `data/reports/`。

版本策略：

- 默认回答“当前最新版本”；问题明确提到旧版本时，按 `docs_version` 过滤。
- 上下文同时包含新旧版本时，生成阶段必须显式标注差异，不允许混合成一段结论。
- 无法判断版本时写 `docs_version="unknown"`，回答中说明版本信息缺失，而不是猜测。

验收指标：

- 文档变更到可检索：≤ 24 小时（目标）。
- 增量重建只重嵌入变更 chunk，重嵌入比例显著低于 100%。
- 版本类问题能给出适用版本；废弃 API 能被识别并提示迁移。
- 每次更新都有 diff 报告与可回滚的上一版本。

---

## 7. 检索设计

### 7.1 查询理解

按顺序做，按需启用：

1. **多轮改写**：把“它怎么配置？”改写成独立问题（需要对话历史）。
2. **版本意图识别**：问题提到具体版本或旧 API 时，提取 `docs_version` 并在检索时过滤；同一问题在新旧版本答案不同时，需要同时召回两版。
3. **意图路由**：
   - 文档问答 → 走 RAG
   - 闲聊 → 直接回答
   - 超出范围 → 明确拒答
   - 要代码示例 → 检索时提高 code chunk 权重
4. **多查询（Multi-Query）**：生成 3 个同义查询，扩大召回（LangChain 内置 `MultiQueryRetriever` 思路）。
5. **HyDE**：先生成假设答案再检索。**不要默认开启**，先用评估验证是否真的有收益。

### 7.2 混合检索

```text
Query
├─ dense 检索：top 40
├─ BM25/sparse 检索：top 40
└─ 元数据过滤：product / language / doc_type / docs_version / is_deprecated
        ↓
      RRF 融合（k=60）
        ↓
    候选 top 20
        ↓
     Rerank
        ↓
    最终 top 6~8（受 token 预算约束）
```

为什么需要 hybrid：

- dense 擅长语义相近但用词不同的情况；
- BM25 擅长类名、函数名、参数名、错误码等精确匹配；
- 技术文档里恰恰有大量精确符号，所以混合检索几乎是必做项。

**版本感知检索策略：**

- 问题未指定版本：默认检索当前版本；旧版本页面降权但不排除。
- 问题指定版本：按 `docs_version` 过滤，同时召回当前版本对应页面做差异对比。
- 结果必须携带 `docs_version` / `is_deprecated`，供生成阶段做版本归因与废弃提示。

### 7.3 检索失败策略

- rerank 最高分低于阈值 → 返回“文档中未找到相关内容”，并给出 2~3 个相近主题链接。
- 首次召回为空 → 查询改写后重试一次。
- 仍为空 → 不调用大模型编造，直接返回兜底话术。
- 版本冲突（新旧版本答案不同）→ 必须同时给出两版结论或明确“以下适用于 vX”，不允许混合。
- 记录每次失败 query，进入 bad case 池，用于补 golden set。

### 7.4 返回给 LLM 的上下文

- 每条 context 带编号和来源：
  ```text
  [1] 版本：v0.4 | 来源：https://... | 标题：... | 小节：...
  <正文>
  ```
- 总上下文控制在 4k~8k tokens。
- 相同 `doc_id` 的 chunk 合并，避免重复。
- 代码块保留语言标记。

---

## 8. 生成与 Agentic RAG

### 8.1 先做 baseline，再加 Agent

**Baseline：**

```text
问题 → 检索 top-k → 拼上下文 → LLM 回答 → 返回引用
```

**Agentic RAG（LangGraph）：**

```mermaid
flowchart TD
    S([用户问题]) --> A[analyze<br/>意图/语言/过滤条件]
    A --> RW[rewrite<br/>独立问题 + 多查询]
    RW --> RT{route}
    RT -->|文档问答| RET[retrieve<br/>hybrid + rerank]
    RT -->|闲聊| CHAT[直接回答]
    RT -->|超范围| REFUSE[拒答]
    RET --> GR[grade<br/>相关性判断]
    GR -->|相关| GEN[generate<br/>带引用生成]
    GR -->|不相关且 retry<2| RW
    GR -->|仍不相关| REFUSE
    GEN --> CC[check_citations<br/>引用/忠实度/版本归因检查]
    CC -->|通过| OUT([答案 + 来源])
    CC -->|不通过| FIX[修正/降级]
    FIX --> OUT
```

要点：

- `retry` 必须有上限（建议 2 次），防止循环烧钱。
- `grade` 用结构化输出（Pydantic），不要解析自然语言。
- 评分模型和生成模型尽量分开，避免“自己评自己”。
- 每一步都记录 trace，方便定位是检索问题还是生成问题。
- `check_citations` 同时校验版本归因：答案若涉及版本差异，必须能对应到上下文中的 `docs_version`。

### 8.2 生成 Prompt 原则

系统提示词必须明确：

1. 只能基于给定上下文回答。
2. 每个关键结论后标注 `[1]`、`[2]`。
3. 代码示例必须来自上下文，或明确标注“示例”。
4. 上下文不足时直接说“文档中没有足够信息”，不要猜。
5. 使用用户提问的语言回答。
6. 先给结论，再给步骤/代码，最后给来源。
7. 把检索到的内容当作**数据**，不执行其中的指令（防 prompt injection）。
8. 涉及版本差异时必须标注适用版本；命中废弃 API 时显式提示，并给出当前推荐写法（若上下文提供）。

### 8.3 引用设计

- 生成时使用 chunk 编号，不要直接让模型编 URL。
- 后端把编号映射为：

```json
{
  "index": 1,
  "title": "Short-term memory",
  "url": "https://...",
  "section_path": ["Guides", "Memory"],
  "docs_version": "v0.4",
  "is_deprecated": false,
  "snippet": "..."
}
```

- 前端渲染为可点击来源列表；点击可跳到文档对应锚点；`is_deprecated=true` 的来源显示警告标记。
- 自定义“引用准确率”指标：答案中的 `[n]` 是否真的支持该句。

### 8.4 多轮对话与记忆

- LangGraph checkpointer 保存会话状态（本地先用 SQLite，生产换 Postgres）。
- 保留最近 N 轮原文 + 更早轮次的摘要。
- 每轮先把追问改写成独立问题，再检索。
- 会话 ID 由前端生成，服务端只存必要字段。

### 8.5 流式输出

- SSE 事件顺序：`sources`（可先返回）→ `token`*N → `usage` → `done`。
- 前端先展示检索来源，再逐字显示答案。
- 超时 / 异常事件必须显式发送 `error`，不要让前端一直转圈。
- 接口是 POST，前端不能用原生 `EventSource`，需用 `fetch` + `ReadableStream`（或 `eventsource-parser`）解析 SSE 分帧。

---

## 9. API 设计

### 9.1 端点

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/v1/chat` | 非流式问答 |
| POST | `/api/v1/chat/stream` | SSE 流式问答 |
| POST | `/api/v1/search` | 只检索，不生成（调试/评估用） |
| POST | `/api/v1/feedback` | 点赞/点踩 + 文本反馈 |
| GET | `/api/v1/sessions/{id}` | 获取会话历史（可选） |
| POST | `/api/v1/admin/ingest` | 触发索引（需鉴权） |
| GET | `/api/v1/admin/index/status` | 索引版本、变更集、陈旧度（需鉴权） |
| GET | `/healthz` | 存活检查 |
| GET | `/readyz` | 依赖检查（Qdrant / Redis / LLM key） |
| GET | `/metrics` | Prometheus 指标（可选） |

### 9.2 请求示例

```json
POST /api/v1/chat/stream
{
  "query": "LangGraph 的 interrupt 怎么用？",
  "session_id": "sess_123",
  "filters": {
    "product": "langgraph",
    "language": "python",
    "docs_version": "latest",
    "include_deprecated": false
  },
  "top_k": 8
}
```

### 9.3 SSE 事件

```text
event: sources
data: {"sources":[{"index":1,"title":"...","url":"...","docs_version":"v0.4","is_deprecated":false}]}

event: token
data: {"text":"LangGraph 的 "}

event: usage
data: {"input_tokens":1234,"output_tokens":56,"cost_usd":0.002}

event: done
data: {"message_id":"msg_123"}
```

### 9.4 工程要求

- Pydantic v2 入参/出参校验。
- 全链路 `request_id`，日志可串联。
- 统一错误码：`400` 参数、`401` 鉴权、`429` 限流、`503` 依赖不可用、`500` 未捕获。
- 普通用户接口与 admin 接口分开鉴权。
- 限流：按 IP / session / API key。
- CORS 只允许前端域名。
- `/healthz` 不检查外部依赖；`/readyz` 检查 Qdrant、Redis、API Key 配置。

---

## 10. 评估体系

> 没有评估的 RAG 项目，面试时很难证明“优化有效”。

### 10.1 Golden Set

建议 100+ 条，覆盖：

| 类型 | 示例 | 数量建议 |
|---|---|---|
| 事实型 | “LangGraph 的 checkpointer 支持哪些后端？” | 20 |
| How-to | “怎么给 LangChain agent 加 memory？” | 20 |
| 代码型 | “给出 interrupt_before 的代码示例” | 15 |
| 对比型 | “MultiQueryRetriever 和 ParentDocumentRetriever 区别？” | 10 |
| 多跳 | “先检索 A 再配置 B” | 10 |
| 中文提问 | 英文文档、中文问题 | 10 |
| **版本差异 / 废弃 API** | “v0.3 的 X 在 v0.4 怎么写？迁移步骤是什么？” | 15 |
| **时效性** | “最新文档里 X 还存在吗？哪个版本引入 / 废弃？” | 5~10 |
| 无答案 | 文档里确实没有的问题 | 5~10 |

每条包含：

```json
{
  "question": "...",
  "reference_answer": "...",
  "golden_urls": ["..."],
  "golden_sections": ["..."],
  "language": "zh",
  "category": "version-diff",
  "docs_version": "v0.4",
  "applies_to_version": ">=0.4",
  "deprecated_apis": ["..."],
  "expected_behavior": "answer | refuse | warn_deprecated"
}
```

### 10.2 指标

**检索：**

- Recall@5 / Recall@10
- MRR
- nDCG@10
- 首次命中位置
- 检索延迟 P50/P95

**时效与版本（项目签名）：**

- 平均陈旧度（页面 `fetched_at` 距今天数）
- 索引更新延迟（源变更 → 可检索）
- 增量重建比例（重嵌入 chunk / 总 chunk）
- 版本归因准确率（答案是否正确标注适用版本）
- 废弃 API 识别率 / 误报率

**生成：**

- Faithfulness（答案是否被上下文支持）
- Answer Relevancy（是否回答问题）
- Context Precision / Context Recall
- 引用准确率（自定义）
- 拒答正确率（无答案题）
- 代码可运行率（抽检）

**系统：**

- 首 token 延迟
- 端到端延迟
- 每次查询 token / 成本
- 错误率、超时率

### 10.3 工具与流程

- **RAGAS**：生成质量指标。
- **自定义脚本**：检索指标、引用准确率、拒答率。
- **LangSmith Experiments**：对比不同 chunk size、embedding、rerank、prompt 的效果。
- **LLM-as-Judge**：用于大规模评估，但必须人工校准 20~30 条。
- **CI 门禁**：小规模 smoke eval（20 条，含 3~5 条版本题）跑在 PR；全量评估定期跑。
- **`docs/BAD_CASES.md`（必做交付物）**：记录 10~20 个 bad case，每个包含：现象、query、检索结果、根因、干预、前后指标。
- **失败分析闭环**：baseline 跑测 → 聚类 bad case → 归因（切分 / 检索 / 排序 / 生成 / 版本）→ 单点干预 → 回归评估。不要只展示成功案例。

### 10.4 优化实验矩阵

按优先级做 A/B：

| 实验 | 对比 |
|---|---|
| 切分 | 固定 chunk vs 父子切分 |
| 检索 | dense only vs hybrid |
| 融合 | RRF vs 加权 |
| 精排 | 无 rerank vs bge-reranker |
| 查询 | 原始 query vs 多轮改写 / multi-query |
| 上下文 | 原始 chunk vs 标题增强 / contextual retrieval |
| 生成 | 直接回答 vs 引用约束 + 自检 |
| 增量更新 | 全量重建 vs 变更页增量（耗时 / 成本 / 陈旧度） |
| 版本过滤 | 无版本过滤 vs 版本感知检索 |
| 版本回答 | 普通 prompt vs 版本归因 prompt |

**每次只改一个变量**，记录指标和成本。

---

## 11. 可观测性与成本

### 11.1 可观测性

- **LangSmith**：先接入，记录每次 graph 运行、检索结果、token、延迟。
- 每个 trace 带 metadata：`session_id`、`model`、`embedding_model`、`retriever_version`、`collection`、`index_build_id`、`docs_version`、`top_k`。
- 结构化日志（JSON），关键字段：request_id、query、retrieval_ms、rerank_ms、llm_ms、tokens、cost、error。
- 指标：
  - QPS / 并发
  - P50 / P95 延迟
  - 检索命中数、rerank 分数分布
  - LLM 错误率 / 超时率
  - 每日 token / 成本
  - 索引：`last_successful_ingest_at`、`staleness_hours`、`changed_pages`、`re_embedded_chunks`、`index_build_duration`
- 后续可加 Prometheus + Grafana；初期不必上。

### 11.2 缓存

| 缓存 | Key | TTL | 说明 |
|---|---|---|---|
| 答案缓存 | query + filters + index_build_id | 1h | 相同问题直接返回 |
| 检索缓存 | query + filters | 30min | 减少 embedding / Qdrant 压力 |
| Embedding 缓存 | content_sha256 | 长期 | 增量索引复用 |
| 会话 | session_id | 1h | LangGraph checkpointer |

可选语义缓存：相似度 > 0.95 才命中，且必须校验 `index_build_id`；不要默认开启。

### 11.3 成本控制

- 查询改写 / 路由 / 评分用小模型。
- 上下文只放 top 6~8，不塞满。
- 限制 `max_tokens`。
- 设置单用户日预算 / 全局日预算。
- 开启厂商的 context caching（若支持）复用系统提示。
- 记录每次请求成本，面试时能讲“如何把单次成本从 X 降到 Y”。

---

## 12. 部署与工程化

### 12.1 Docker Compose 拓扑

```yaml
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports: ["6333:6333", "6334:6334"]
    volumes: ["qdrant_data:/qdrant/storage"]

  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
    volumes: ["redis_data:/data"]

  api:
    build: .
    ports: ["8000:8000"]
    env_file: .env
    depends_on: [qdrant, redis]

  web:
    build: ./apps/web      # Vite 构建产物 + nginx
    ports: ["3000:80"]
    depends_on: [api]

volumes:
  qdrant_data:
  redis_data:
```

> Langfuse / Postgres / ClickHouse 先不要放进同一份 compose，等需要时单独一份 `docker-compose.observability.yml`。

### 12.2 CI/CD

GitHub Actions：

1. `ruff check`
2. `mypy`
3. `pytest`（单元 + 集成，集成用 Docker service 起 Qdrant/Redis）
4. 增量索引冒烟：mock 文档变更 → 断言只重嵌入变更 chunk、diff 报告正确、alias 可回滚
5. `npm ci && npm run lint && npm run build`（前端）
6. `ragas` smoke eval（20 条，含 3~5 条版本题；可用 mock LLM 或小模型）
7. 构建 Docker 镜像
8. 可选：推送镜像 / 部署

### 12.3 配置管理

- 所有配置走 `.env` + Pydantic Settings。
- `.env.example` 入库，`.env` 不进 git。
- 关键配置：
  - `LLM_PROVIDER`, `LLM_MODEL`, `LLM_API_KEY`, `LLM_BASE_URL`
  - `EMBEDDING_MODEL`, `EMBEDDING_DIM`, `EMBEDDING_API_KEY`
  - `QDRANT_URL`, `QDRANT_COLLECTION`
  - `DOCS_SOURCE_KIND`, `DOCS_REPO_URL`, `DOCS_REPO_BRANCH`
  - `INDEX_REFRESH_CRON`, `STALENESS_THRESHOLD_HOURS`, `DEFAULT_DOCS_VERSION`
  - `EMBEDDING_CACHE_DIR`
  - `REDIS_URL`
  - `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`
  - `RETRIEVAL_TOP_K`, `RERANK_TOP_K`, `MAX_CONTEXT_TOKENS`
  - `RATE_LIMIT_*`
- 启动时校验必需配置，缺项直接失败（fail fast）。
- 定时调度放在宿主机：Windows 用 Task Scheduler，Linux / CI 用 cron；不要为调度引入 Airflow / Celery。

### 12.4 安全

- 检索到的文档内容视为不可信数据，防 prompt injection。
- 不在日志里打印 API Key、完整 prompt、用户敏感信息。
- admin 接口必须鉴权。
- 限流 + 请求体大小限制。
- 依赖扫描：`pip-audit`、Docker 镜像扫描（可选 Trivy）。
- 遵守文档站 robots.txt，不重新分发全文。

---

## 13. 目录结构

```text
D:\ANU\LangChain-rag\
├─ apps/
│  ├─ api/                     # FastAPI 应用（如拆分部署）
│  └─ web/                     # Vite + React + TypeScript
├─ src/
│  └─ langchain_rag/
│     ├─ config.py             # Pydantic Settings
│     ├─ ingestion/
│     │  ├─ fetcher.py         # sitemap / Git / 爬取
│     │  ├─ change_detector.py # Git diff / sitemap / 内容哈希
│     │  ├─ parser.py          # HTML→Markdown、MDX
│     │  └─ normalizer.py      # 清洗、去重、metadata、版本字段
│     ├─ indexing/
│     │  ├─ chunker.py         # 父子切分、代码保护
│     │  ├─ embedder.py        # dense + sparse
│     │  ├─ indexer.py         # Qdrant 写入 / alias 切换
│     │  └─ updater.py         # 增量重建 / 版本切换（签名能力）
│     ├─ retrieval/
│     │  ├─ query.py           # 改写、multi-query、HyDE
│     │  ├─ hybrid.py          # dense + sparse + RRF
│     │  ├─ reranker.py
│     │  └─ filters.py
│     ├─ generation/
│     │  ├─ prompts.py
│     │  ├─ generator.py
│     │  └─ citations.py
│     ├─ graph/
│     │  ├─ state.py
│     │  ├─ nodes.py
│     │  └─ builder.py         # LangGraph 编排
│     ├─ evaluation/
│     │  ├─ golden_set.py
│     │  ├─ retrieval_metrics.py
│     │  ├─ version_metrics.py # 陈旧度 / 版本归因 / 废弃识别
│     │  ├─ bad_cases.py       # bad case 聚合与报告
│     │  └─ ragas_runner.py
│     └─ observability/
│        ├─ tracing.py
│        └─ logging.py
├─ scripts/
│  ├─ ingest.py
│  ├─ update_index.py          # 定时增量更新（签名能力）
│  ├─ diff_report.py           # 变更 / 陈旧度报告
│  ├─ rebuild_index.py
│  ├─ evaluate.py
│  └─ serve.py
├─ tests/
│  ├─ unit/
│  ├─ integration/
│  └─ e2e/
├─ data/
│  ├─ raw/
│  ├─ processed/
│  ├─ cache/                   # embedding 缓存
│  ├─ manifests/
│  └─ reports/                 # diff / 评估报告
├─ docker/
│  ├─ Dockerfile.api
│  └─ Dockerfile.web
├─ docs/
│  ├─ PROJECT_DESIGN.md
│  ├─ EVALUATION.md
│  └─ BAD_CASES.md
├─ pyproject.toml
├─ docker-compose.yml
├─ Makefile
├─ .env.example
├─ .gitignore
└─ README.md
```

---

## 14. 迭代路线图

> 工期是按“全职投入”估算的乐观值；实际按周推进，宁可砍范围也不要跳过评估。主线顺序不可乱：先 baseline，再检索基线，再做签名能力，最后才是 Agent 与工程化。

### Phase 0：环境与骨架（0.5~1 天）

- 新建 conda 环境 `langchain-rag`。
- 初始化 git、pyproject、目录结构、`.env.example`（含版本 / 增量配置）。
- `docker compose up -d qdrant redis`。
- 跑通 FastAPI `/healthz`。
- **验收**：`make dev` 能起 API，Qdrant dashboard 可访问。

### Phase 1：数据 + dense baseline（3~5 天）

- 探测数据源：能否拿到 Git commit / 版本信息；确定采集入口。
- 采集 1 个产品文档，落盘原始数据；从第一天写入 `docs_version` / `source_ref` / `fetched_at` / `is_deprecated`。
- HTML/MDX 解析、清洗、metadata。
- 固定大小切分 + dense embedding。
- 写入 Qdrant。
- 实现 `/api/v1/search` 和 `/api/v1/chat`。
- 手工构造 20 条问题。
- **验收**：能回答 20 条中 15 条以上；每条有来源；检索延迟 < 500ms；元数据包含版本字段。

### Phase 2：检索基线（3~4 天）

- 父子切分 + 标题上下文。
- BM25 sparse + dense 混合 + RRF。
- 本地 reranker。
- 元数据过滤（含 `docs_version` / `is_deprecated`）。
- 多轮查询改写。
- 先建立 60 条 golden set，跑检索指标（Phase 4 扩展到 100+ 并补版本题）。
- **验收**：Recall@10 ≥ 0.85；相比 Phase 1 有可量化提升；评估报告落盘。hybrid + rerank 到此为止是“必备基线”，不作为差异化卖点。

### Phase 2.5：项目签名——版本感知的活知识库（3~4 天）

- Git commit diff / 内容哈希变更检测，生成 diff 报告。
- 版本字段拆分与废弃标记；无法判断时显式 `unknown`。
- 增量重建：只重嵌入变更 chunk，复用 embedding 缓存；写入新 collection 并原子切 alias。
- 版本类 golden set（15 条）+ 陈旧度 / 更新延迟 / 重嵌入比例指标。
- 回滚演练：保留上一版本 collection，验证可切回。
- **验收**：文档变更 ≤ 24h 可检索；只重嵌入变更 chunk；版本类问题能归因；有前后对比报告与回滚记录。

### Phase 3：Agentic RAG + 对话 + 流式（3~4 天）

- LangGraph 节点：analyze / rewrite / route / retrieve / grade / generate / check。
- 相关性评分 + 最多 2 次重试。
- checkpointer + 会话历史 + 摘要。
- SSE 流式输出 + sources 先行。
- React 聊天界面（Vite + React + TypeScript）。
- **验收**：多轮追问可用；无关问题会拒答；流式首 token < 2s；前端能点击来源；版本信息可见。

### Phase 4：评估与失败分析（2~3 天）

- RAGAS 接入。
- 引用准确率、拒答率、版本归因准确率、成本指标。
- LangSmith trace。
- 建立实验对比表（dense vs hybrid、rerank on/off、增量 vs 全量、版本过滤 on/off）。
- 完成 `docs/BAD_CASES.md`（10~20 个 case）与失败分析闭环。
- CI 接 smoke eval（含版本题）。
- **验收**：每次改检索/提示词都能看到指标变化；有基线报告；bad case 有根因与干预记录。

### Phase 5：工程化收尾（2~3 天）

- Redis 缓存。
- 限流、鉴权、错误处理。
- Docker Compose 一键启动。
- pytest 单元 + 集成测试（含增量 / 版本回归）。
- GitHub Actions。
- README、架构图、演示脚本。
- **验收**：新机器按 README 能 15 分钟内跑起来；CI 全绿。

### Phase 6：加分项（默认不做）

- MCP Server：把 `search_docs` / `answer_question` 暴露给 Claude Code / 其他 Agent。
- GraphRAG 或多文档源（JS 文档、LangSmith 文档）。
- 多租户 / 用户反馈闭环。
- PWA / 移动端适配（如确有多端需求再评估 Taro）。
- 部署到云 VM + 域名 + HTTPS。
- K8s manifests。
- 多模态（截图/表格理解）。

> 只有主线与评估都完成后才考虑 Phase 6；面试叙事以签名能力和 bad case 为主，不要用加分项稀释重点。

---

## 15. JD 关键词覆盖矩阵

| 关键词 | 在项目中的落点 | 优先级 |
|---|---|---|
| RAG 全链路 | 采集 → 切分 → 索引 → 检索 → 生成 → 评估 | P0 |
| LangChain | 组件、retriever、prompt、embedding、loader | P0 |
| LangGraph | 有状态 Agent 编排、条件边、checkpointer | P0 |
| Agent / Agentic RAG | 路由、评分、重试、引用检查 | P0 |
| Vector Database | Qdrant dense + sparse | P0 |
| Embedding | 多语言 embedding、维度管理、增量索引 | P0 |
| Hybrid Search | dense + BM25 + RRF | P0 |
| Rerank | CrossEncoder / API rerank | P0 |
| Query Rewriting | 多轮独立问题、multi-query | P1 |
| HyDE | 可选实验项 | P2 |
| Chunking | 父子切分、代码块保护、标题上下文 | P0 |
| Metadata Filtering | 产品 / 语言 / 版本 / 类型过滤 | P0 |
| 数据版本管理 / 增量索引 | Git diff 变更检测、增量重建、陈旧度指标 | P0（项目签名） |
| Prompt Engineering | 约束生成、引用、拒答 | P0 |
| Context Engineering | 上下文预算、去重、父子返回 | P0 |
| Citations | chunk → URL 映射、引用准确率 | P0 |
| Memory | 多轮对话、摘要、checkpointer | P1 |
| Function Calling / Tool Calling | 路由、评分、MCP 工具 | P1 |
| MCP | 把 RAG 暴露成 MCP Server | P2 |
| FastAPI | API、SSE、Pydantic 校验 | P0 |
| Async / Streaming | 异步检索、SSE 流式 | P0 |
| Redis | 答案缓存、检索缓存、会话 | P1 |
| PostgreSQL / pgvector | 备选向量库 / 生产会话存储 | P2 |
| Docker / Compose | 一键部署 | P0 |
| CI/CD | GitHub Actions | P0 |
| Testing | pytest、集成测试 | P0 |
| Observability | LangSmith / 结构化日志 / 指标 | P0 |
| Evaluation | RAGAS、LLM-as-Judge、golden set | P0 |
| Cost Optimization | 模型路由、缓存、token 预算 | P1 |
| Security | prompt injection、鉴权、限流、密钥 | P1 |
| K8s | 可选部署 manifests | P3 |
| React / Vite | 聊天前端、SSE 消费 | P1 |
| Celery / Airflow | 可选异步索引调度 | P3 |
| GraphRAG | 可选进阶实验 | P3 |

> 说明：不是“全都用上”就等于好项目。本项目只押一个差异化点——**版本感知的活知识库**；hybrid + rerank 是必备基线；评估与 bad case 闭环是证明手段。P2/P3 不进入 v1 主线，面试时也不主动展开。

---

## 16. 风险与对策

| 风险 | 影响 | 对策 |
|---|---|---|
| 没有可用的 LLM Key | 无法开发 | 先确认供应商，准备独立 Key；代码保持 provider 无关 |
| 文档站点改版 / 反爬 | 采集失败 | 优先 Git 源码；sitemap + 缓存；解析器加测试；限制频率 |
| Docker Hub 不稳定 | 拉镜像失败 | 使用国内镜像（已配置并验证）；镜像离线保存 |
| 16GB 内存不足 | 服务互相挤爆 | 只跑 Qdrant + Redis；不跑 Langfuse 全家桶；限制 Docker 内存 |
| 本地 GPU 太弱 | 不能本地推理 | 云端 LLM；本地只做 CPU rerank / embedding |
| 检索质量差 | 答案胡说 | hybrid + rerank + 阈值 + 拒答；先评估再优化 |
| 引用造假 | 可信度崩 | 编号映射由后端完成；引用检查；不直接生成 URL |
| 评估成本高 | 迭代慢 | smoke set + 小模型 judge + 定期全量 |
| LLM 自评偏差 | 指标虚高 | 生成/评分模型分离；人工校准；多指标交叉 |
| 成本失控 | 账单爆炸 | 缓存、限流、预算、小模型路由 |
| 版本漂移 / 文档高频更新 | 答案引用旧 API | Git commit diff + 增量索引；`docs_version` / `is_deprecated` 字段；版本类 golden set；陈旧度告警 |
| 增量逻辑错误（漏删 / 漏更） | 知识库污染 | diff 报告 + 数量校验 + 抽样检索 + alias 回滚 |
| 文档源缺少版本标记 | 无法做版本归因 | 记录 `source_ref` / `fetched_at`；从页面文本抽取版本；无法判断时显式 `unknown` |
| 选型被质疑（为什么不用 LlamaIndex） | 面试失分 | 准备 §4.7 叙事；检索层接口化；强调 LangGraph 编排与版本能力 |
| 沦为教程复现 | 项目无差异化 | 只保留一个签名能力 + bad case 闭环；砍掉 Phase 6 |
| 中文提问效果差 | 体验差 | 多语言 embedding；中文 golden set；查询翻译可选 |
| 版权 / robots | 合规风险 | 遵守 robots、限速、只用于学习演示、不重新分发全文 |

---

## 17. 待确认问题

开工前仍需确认 3 件事（前端形态已于 v0.2 确定）：

1. **LLM / Embedding 供应商和 Key**
   - 推荐：DeepSeek（LLM）+ 通义千问（Embedding/Rerank），或 OpenAI 一套。
   - 需要你提供项目专用 Key（不要用 Claude Code 的环境变量）。
   - 如果不方便用国内 API，也可以先用 OpenAI 或 Claude，代码层面都兼容。

2. **文档范围**
   - A：只要 LangChain Python 文档（最小、最快）
   - B：LangChain Python + LangGraph（推荐）
   - C：再加 JS 文档 / API Reference（数据量更大）

> 版本感知能力依赖数据源可提供 commit / 版本信息。若只能抓渲染后的 HTML，退化为 `fetched_at` + 内容哈希 + 页面文本中的版本标记。

3. **回答语言**
   - 中文提问 → 中文回答（推荐）
   - 中英双语都支持
   - 统一英文

> 前端形态（已确定）：Vite + React + TypeScript + Tailwind CSS + shadcn/ui；不使用 Streamlit / Next.js 作为主方案。如后续确有多端需求，再评估 Taro。

另外请告诉我：**有没有目标岗位的 JD 原文？** 我可以把技术点再对齐一次，避免“做了很多但不对味”。

---

## 18. 下一步

确认上面的问题后，第一批动作：

1. 新建环境、初始化 repo、写 `pyproject.toml` 和 `.env.example`（包含版本 / 增量相关配置）。
2. 起 Qdrant / Redis，跑通连通性。
3. **先探测数据源能否提供 Git commit / 版本信息**，这决定“版本感知”能做到多深；再写采集器，抓 50 页验证解析质量。
4. 从第一天就写入 `docs_version` / `source_ref` / `fetched_at` / `is_deprecated`，避免后期返工。
5. 建立第一个 collection，跑通 dense baseline。
6. 用 20 条问题做第一次人工评估，记录 bad case。
7. 进入 Phase 2：hybrid + rerank 基线；随后立即做 Phase 2.5 版本感知增量索引。

---

## 附录 A：`.env.example` 草案

```dotenv
# LLM
LLM_PROVIDER=openai_compatible
LLM_MODEL=deepseek-chat
LLM_API_KEY=
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=2048

# Embedding
EMBEDDING_PROVIDER=openai_compatible
EMBEDDING_MODEL=text-embedding-v4
EMBEDDING_API_KEY=
EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_DIM=1024

# Reranker
RERANKER_PROVIDER=local
RERANKER_MODEL=BAAI/bge-reranker-base

# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=langchain_docs_current
QDRANT_BUILD_PREFIX=langchain_docs_build

# Docs source / incremental index
DOCS_SOURCE_KIND=git
DOCS_REPO_URL=
DOCS_REPO_BRANCH=main
INDEX_REFRESH_CRON=0 3 * * *
STALENESS_THRESHOLD_HOURS=24
DEFAULT_DOCS_VERSION=latest
INCLUDE_DEPRECATED=false
EMBEDDING_CACHE_DIR=data/cache/embeddings

# Redis
REDIS_URL=redis://localhost:6379/0

# Retrieval
RETRIEVAL_DENSE_TOP_K=40
RETRIEVAL_SPARSE_TOP_K=40
RERANK_TOP_K=20
FINAL_TOP_K=8
MAX_CONTEXT_TOKENS=8000

# Observability
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=langchain-rag-dev

# API
API_HOST=0.0.0.0
API_PORT=8000
ADMIN_API_KEY=
RATE_LIMIT_PER_MINUTE=30
```

---

## 附录 B：参考入口

- LangChain 官方文档：`https://docs.langchain.com/`
- LangChain Python 文档：`https://python.langchain.com/`
- LangChain JS 文档：`https://js.langchain.com/`
- LangGraph 文档：`https://langchain-ai.github.io/langgraph/`
- Qdrant 文档：`https://qdrant.tech/documentation/`
- RAGAS 文档：`https://docs.ragas.io/`
- LangSmith 文档：`https://docs.smith.langchain.com/`

> 实际开工时先探测官方文档 Git 源码是否可用；不可用时再回退到 `sitemap.xml` + HTML 抓取。

---

## 附录 C：面试讲述提纲（做完后填真实数据）

1. 业务问题与数据源特征（技术文档、代码块、精确符号、版本高频更新）。
2. 反教程声明：baseline 只是对照组，不是成果；本项目只押一个签名能力。
3. 架构：离线增量索引 + 在线 Agentic RAG。
4. 签名能力：版本感知的活知识库
   - Git commit diff / 内容哈希变更检测
   - `docs_version` / `source_ref` / `is_deprecated` 版本元数据
   - 只重嵌入变更 chunk、alias 原子切换、可回滚
   - 陈旧度、更新延迟、版本归因指标
5. 检索基线：父子切分 + 标题上下文、dense + BM25 + RRF、rerank、查询改写。
6. 评估闭环：100+ golden set（含版本题）、RAGAS、Recall@10、引用准确率、bad case 归因与回归。
7. 结果：优化前后对比（填真实数字；不要编造）。
8. 取舍与踩坑：为什么用 LangChain 而不是 LlamaIndex、为什么不用 HyDE、为什么不用本地模型、为什么 Qdrant 而不是 pgvector。
9. 下一步：MCP、多文档源等，但明确说明这是加分项而非主线。

---

## 附录 D：三个必答题（面试前准备）

### 1. 为什么用 LangChain 文档做知识库？

- 官方文档结构清晰、有代码块和稳定 URL，适合把精力集中在检索与生成；
- 文档更新频繁，版本时效性是一个真实工程问题，可量化、可演示；
- 可以讲“用 LangChain 文档服务 LangChain 开发者”的自指故事。

### 2. 为什么用 LangChain / LangGraph，而不是 LlamaIndex？

- 本项目要展示的是 **Agentic RAG 的有状态编排**（路由、评分、重试、引用 / 版本检查），LangGraph 是核心；
- LlamaIndex 在摄取与检索抽象上很强，可作为对照，但不能替代这条需要显式控制的工作流；
- 检索层抽象为接口，版本感知索引与混合检索与框架无关；不是“只会 LangChain”。

### 3. 这个项目和网上教程有什么区别？

- 大部分教程停在“能问能答”；本项目把 baseline 当对照组，只深挖一个工程问题；
- 签名能力是版本感知的增量索引，有变更检测、版本元数据、陈旧度指标与回滚机制；
- 有 100+ golden set、版本类题型和 bad case 闭环，能证明每一次优化有效。
