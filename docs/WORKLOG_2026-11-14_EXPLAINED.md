# 2026-11-14 工作记录 · 名词与原理注解

> 配套文档：[WORKLOG_2026-11-14.md](WORKLOG_2026-11-14.md)
>
> 这份文档不重复“今天做了哪些命令”，而是解释：
>
> 1. Worklog 里出现的**名词**是什么意思；
> 2. 每个**行为/步骤**在做什么、为什么这么做、底层原理；
> 3. **今天之前**的 parser/chunker 等改动是什么、为什么改；
> 4. 哪些概念容易混淆，以及当前实现处于什么水平。
>
> 事实来源优先级：`docs/PROJECT_DESIGN.md` > `docs/RAG_PRINCIPLES.md` > 本文档。本文档是学习/复盘注解，不是新的设计规范。

---

## 0. 先建立全局图景

这个项目不是“把文档丢给大模型让它回答”，而是一条有明确分工的流水线。

### 离线索引链路

```text
FastAPI Markdown 文档
  -> 解析 parser
  -> 切分 chunker
  -> embedding 客户端
  -> indexer
  -> Qdrant 向量库
```

### 在线问答链路

```text
用户问题
  -> query embedding
  -> Qdrant 相似度检索
  -> 取出 chunk + parent_text + URL/版本元数据
  -> 拼上下文
  -> LLM 生成
  -> 后端映射 [n] 引用
  -> 前端展示答案和来源
```

### 为什么要拆成两条链路

- 检索和生成是两种完全不同的能力：检索负责“找证据”，生成负责“组织语言”。
- 分开之后，检索可以单独评估、单独替换（dense -> hybrid -> rerank），不被 LLM 绑死。
- 生成阶段只做“阅读 + 推理”，不负责记文档；文档变了只需重建索引，不必重训模型。

---

## 1. 名词表：RAG / 检索 / 生成

| 名词                           | 含义                                                              | 为什么重要                                         |
| ---------------------------- | --------------------------------------------------------------- | --------------------------------------------- |
| **RAG**                      | Retrieval-Augmented Generation，检索增强生成。先从外部知识库找证据，再让 LLM 基于证据回答。 | 解决 LLM 知识过期、不可溯源、会幻觉的问题。                      |
| **Retrieval（检索）**            | 根据 query 从知识库找相关片段。                                             | 决定回答的上限；没检索到正确证据，生成再强也只能编或拒答。                 |
| **Generation（生成）**           | 把检索到的上下文交给 LLM，组织成答案。                                           | 决定答案是否易读、是否引用正确、是否拒答。                         |
| **Grounding（落地/有据）**         | 回答必须建立在给定上下文上，而不是模型记忆。                                          | 本项目的系统提示明确要求“只依据上下文”。                         |
| **Context window（上下文窗口）**    | 模型一次能看多少 token。                                                 | 决定 top-k 能放多少 parent_text，不能无限塞。              |
| **Token**                    | 模型计费和上下文长度的单位，不等于字符。                                            | embedding batch、LLM 额度、max_tokens 都按 token 算。 |
| **Embedding**                | 把一段文本编码成固定长度向量，语义相近的文本向量距离近。                                    | 支撑语义检索；中文 query 能匹配英文文档。                      |
| **Vector（向量）**               | 一串浮点数，例如 1024 维 `[0.12, -0.03, ...]`。                           | Qdrant 存的就是向量 + payload。                      |
| **Dimension（维度）**            | 向量长度，例如 `text-embedding-v4` 的 1024。                             | 换模型/换维度必须重建索引，不能混用。                           |
| **Cosine similarity（余弦相似度）** | 用向量夹角衡量相似度，Qdrant 的 distance 配置。                                | 检索时按相似度排序。                                    |
| **ANN**                      | Approximate Nearest Neighbor，近似最近邻搜索。                           | 向量库用近似算法换速度，不保证绝对精确。                          |
| **Top-k**                    | 取相似度最高的 k 条。                                                    | 决定召回数量和上下文成本。                                 |
| **Dense retrieval**          | 用 embedding 向量做语义检索。                                            | 当前项目已实现；对“换个说法”有效。                            |
| **Sparse retrieval / BM25**  | 基于词频/逆文档频率的关键词检索。                                               | 对类名、函数名、参数名等精确符号有效；当前尚未实现。                    |
| **Hybrid search**            | dense + sparse 两路召回再融合。                                         | 技术文档既需要语义，也需要精确符号。                            |
| **RRF**                      | Reciprocal Rank Fusion，按排名融合多路结果。                               | BM25 分数和余弦分数不可直接相加，RRF 更稳；当前尚未实现。             |
| **Rerank**                   | 对初步召回结果用 cross-encoder 再精排。                                     | 提升 top-k 精度；当前尚未实现。                           |
| **Cross-encoder**            | 把 query 和 passage 一起输入模型打分，精度高但慢。                               | 只适合对 top 20 左右精排，不能全库跑。                       |
| **Recall@k**                 | 前 k 条里是否包含正确证据。                                                 | 检索的核心指标；比“看起来像”更可靠。                           |
| **MRR**                      | Mean Reciprocal Rank，正确结果排名倒数的平均。                               | 衡量“第一条多快命中”。                                  |
| **nDCG@k**                   | 考虑排序位置的检索指标。                                                    | 衡量整体排序质量。                                     |
| **Golden set**               | 人工标注的问题 + 期望来源/答案集合。                                            | 没有它就无法证明优化有效。                                 |
| **Bad case**                 | 失败样本，记录现象、根因、干预、前后指标。                                           | 项目要求 `docs/BAD_CASES.md`；失败分析比展示成功更重要。        |
| **RAGAS**                    | 评估 RAG 生成质量的工具/指标集。                                             | 计划使用，当前未接入。                                   |
| **LLM-as-Judge**             | 用另一个 LLM 评分。                                                    | 成本低但有偏差；需要人工校准。                               |
| **Citation（引用）**             | 答案中的 `[n]` 对应到具体文档 URL。                                         | 让回答可点击、可审计。                                   |
| **Refusal（拒答）**              | 检索不到足够信息时明确说“不知道”。                                              | 比编造更可信；本项目有 `REFUSAL` 文案。                     |
| **Hallucination（幻觉）**        | 模型生成上下文没有的内容。                                                   | RAG 的核心风险；靠 grounding、引用检查、拒答降低。              |
| **Prompt injection**         | 文档里藏“忽略之前指令”之类内容，诱导模型执行。                                        | 系统提示明确把检索内容当数据、不当指令。                          |

---

## 2. 名词表：FastAPI 文档解析

| 名词 | 含义 | 为什么重要 |
|---|---|---|
| **Markdown** | FastAPI 文档的源格式。 | 纯文本、结构清晰，适合解析。 |
| **MkDocs Material** | FastAPI 文档使用的静态站点主题。 | 有 include、admonition、tab 等扩展语法。 |
| **Code include** | `{* ../../docs_src/xxx.py hl[4,9] *}`，文档不内嵌代码，而是引用外部文件。 | 不展开 include，文档里的代码全部丢失。 |
| **`hl[...]`** | 代码高亮行号参数。 | 解析时忽略，不影响代码内容。 |
| **`ln[start:end]`** | 只引用源文件的行范围。 | 需要按行截取；否则会把整份源码塞进来。 |
| **Admonition** | `/// tip`、`/// warning`、`/// note` 等提示块。 | 废弃/迁移信息常在这里；丢了会答错版本。 |
| **Tab group** | `//// tab` 开头的多版本代码切换块（例如 Python 3.10+）。 | 真实文档有；不处理会重复代码、丢 variant 标签。 |
| **Anchor** | 标题后的 `{ #first-steps }`。 | 引用要能跳到具体小节。 |
| **URL mapping** | `docs/en/docs/tutorial/first-steps.md` -> `https://fastapi.tiangolo.com/tutorial/first-steps/`。 | 回答里的来源必须可点击。 |
| **`reference/`** | mkdocstrings 指令 `::: fastapi.FastAPI`，不是实际 API 文本。 | v1 不作为正文索引，否则会索引空内容。 |
| **`release-notes.md`** | FastAPI 版本变更记录。 | 是版本来源，但体量巨大；需要限制进入普通索引的版本数。 |
| **`meta` 页面** | about、contributing、newsletter 等。 | 默认不参与业务问答，避免答非所问。 |
| **`doc_type`** | 页面类型：tutorial / advanced / how-to / reference / release_notes / meta 等。 | 检索过滤、评估分类型统计都靠它。 |
| **Section path** | 标题层级路径，如 `["Lifespan Events", "Alternative Events (deprecated)"]`。 | 引用展示和上下文头需要。 |
| **Code fence** | ``` 围栏代码块。 | 代码块不能被切断；代码里的 `#` 不是标题。 |

---

## 3. 名词表：切分与索引

| 名词 | 含义 | 为什么重要 |
|---|---|---|
| **Chunk** | 文档切出来的片段。 | embedding 和检索的基本单位。 |
| **Child chunk** | 较小片段，约几百 token，用于 embedding/召回。 | 粒度小，语义集中，召回更准。 |
| **Parent chunk** | 由相邻 child 组成的较大片段，返回给 LLM。 | 命中 child 后给 LLM 完整上下文。 |
| **Parent/Child（Small-to-Big）** | 小片段检索、大片段返回。 | 解决“检索要小、生成要大”的矛盾。 |
| **Overlap** | 相邻 child 保留一小段重叠。 | 防止关键句正好落在边界。 |
| **Code atomicity** | 代码块不可从中间切开。 | 切开的代码不可运行，引用也失去意义。 |
| **Section header injection** | embedding 前拼接 `[FastAPI > Tutorial > First Steps]`。 | 让 embedding 知道片段属于哪一节；成本低、收益明显。 |
| **Stable chunk ID** | `doc_id + section_hash + chunk_index`。 | 增量索引时能判断哪些 chunk 变了。 |
| **`content_sha256`** | 内容哈希。 | 内容没变就跳过 embedding；增量/幂等的基础。 |
| **Qdrant** | 向量数据库，存向量和 payload。 | 不是关系型数据库；专门做相似度检索和过滤。 |
| **Collection** | Qdrant 里的一组点（向量 + payload）。 | 一次索引构建对应一个 build collection。 |
| **Alias** | 指向 collection 的别名，如 `fastapi_docs_current`。 | 新 collection 校验后切 alias，实现原子切换/回滚。 |
| **Named vector** | collection 里命名的向量字段，如 `dense`。 | 未来可加 `sparse`，同一个点存多种向量。 |
| **Point** | Qdrant 的一条记录：id + vector + payload。 | 当前一个 child chunk 一个 point。 |
| **Payload** | 点的附加字段：URL、section、版本、文本等。 | 检索后直接返回来源和上下文。 |
| **Payload index** | 对 payload 字段建索引，加速过滤。 | 按 doc_type/version/deprecated 过滤。 |
| **Upsert** | 插入或更新点。 | 写入/重建索引使用。 |
| **Build collection** | `fastapi_docs_build_<ts>` 临时集合。 | 写坏不影响当前 alias。 |
| **Atomic alias switch** | 校验通过后一次性把 alias 指向新集合。 | 避免用户查到半成品；旧集合保留可回滚。 |
| **`docs_version`** | 文档快照对应的 FastAPI 版本，如 `0.162.0`。 | 回答标注适用版本。 |
| **`source_ref`** | 文档 Git commit SHA，如 `git:7a0a3f0...`。 | 变更检测和可追溯。当前 C 数据快照无 `.git`，所以是 `unknown`。 |
| **`index_build_id`** | 本次索引构建标识。 | 区分同一文档版本的不同构建。 |
| **`fetched_at`** | 文档抓取/索引时间。 | 计算陈旧度。 |
| **`is_deprecated`** | 强废弃信号：该 section 自身是废弃 API。 | 检索降权/提示迁移。 |
| **`deprecation_hint`** | 弱废弃信号：正文提到废弃，但不代表整个 section 废弃。 | 避免把“教你怎么废弃”误判成废弃 API。 |
| **`content_type`** | prose / code / mixed。 | 检索加权和展示。 |
| **`code_language`** | python / json / bash 等。 | 代码块展示与过滤。 |

---

## 4. 名词表：模型与 API

| 名词 | 含义 | 为什么重要 |
|---|---|---|
| **LLM** | 大语言模型，负责生成回答/改写/评分。 | 本项目的生成器。 |
| **Embedding model** | 把文本编码成向量的模型。 | 和 LLM 是两件事；DeepSeek 没有 embedding。 |
| **OpenAI-compatible** | 遵循 OpenAI `/chat/completions`、`/embeddings` 协议的 API。 | 代码不写死厂商，Qwen/DeepSeek/SiliconFlow 都能接。 |
| **Provider-agnostic** | 模型供应商可替换。 | 换模型只改 `.env`，不改业务代码。 |
| **API Key** | 调用模型的凭证。 | 不入库、不进 Git、不打印日志。 |
| **Base URL** | API 根地址，如 `https://dashscope.aliyuncs.com/compatible-mode/v1`。 | 不同 provider 不同。 |
| **Batch** | 一次请求提交多条文本。 | 提高 embedding 吞吐；DashScope v4 上限 10。 |
| **`max_tokens`** | 单次生成最多输出多少 token。 | 控制成本；当前有硬上限 512。 |
| **Temperature** | 生成随机性。 | 问答用低温度（0.1）更稳定。 |
| **401 / 403 / 429** | 未授权 / 无权限 / 限流或额度。 | 区分 Key 错、模型未开通、限流。 |
| **Free quota（免费额度）** | 新账号赠送的 token 额度。 | 够开发调试，但不等于无限；仍有限流。 |
| **DashScope / 百炼** | 阿里云的模型平台。 | 当前 embedding/chat 都用它。 |
| **`qwen-flash`** | 百炼的低价 chat 模型。 | 当前可用且便宜；`qwen-turbo` 曾 429。 |

---

## 5. 名词表：工程、Docker、部署

| 名词 | 含义 | 为什么重要 |
|---|---|---|
| **Docker image** | 只读模板，包含运行环境和代码。 | 保证环境一致。 |
| **Docker container** | 镜像的运行实例。 | 真正跑 API/Qdrant/Redis/Nginx 的进程。 |
| **Docker Compose** | 用 YAML 编排多个容器。 | 一条命令起 API + Qdrant + Redis + Web。 |
| **Service** | Compose 里的一个容器定义，如 `api`、`qdrant`。 | 服务名可作为容器内 DNS（`http://qdrant:6333`）。 |
| **Docker network** | 容器之间的私有网络。 | API 用服务名访问 Qdrant/Redis，不依赖宿主机 localhost。 |
| **Port mapping** | `8080:80` 表示宿主机 8080 -> 容器 80。 | 决定外部访问入口。 |
| **Volume** | 持久化数据目录。 | Qdrant/Redis 数据不随容器删除。 |
| **Healthcheck** | 定期检查容器是否健康。 | `docker compose ps` 能看到 healthy。 |
| **Registry mirror** | Docker 镜像加速源。 | Docker Hub 直连慢时用 `docker.1ms.run` 等。 |
| **Tag** | 给本地镜像起名字，如 `qdrant/qdrant:latest`。 | Compose 按 tag 找镜像；从 mirror 拉取后要重新 tag。 |
| **Nginx** | Web 服务器/反向代理。 | 生产环境托管静态文件、转发 `/api`。 |
| **Static hosting** | 只发 HTML/CSS/JS，不跑 Node/Vite。 | 生产前端不需要 dev server。 |
| **Vite** | 前端开发服务器 + 构建工具。 | `npm run build` 产物是 `dist`；`vite dev` 只用于开发。 |
| **`dist`** | 前端构建产物目录。 | Nginx 直接托管。 |
| **SPA fallback** | `try_files $uri $uri/ /index.html`。 | 前端路由刷新不 404。 |
| **Reverse proxy** | Nginx 把 `/api` 转给 API 容器。 | 同源访问，避免 CORS。 |
| **CORS** | 浏览器跨域资源共享策略。 | 前端 5173 调 8000 时需要；同源反代后不需要。 |
| **Origin** | 协议 + 域名 + 端口，如 `http://127.0.0.1:5173`。 | CORS 按 origin 判断。 |
| **Playwright** | 浏览器自动化工具。 | 真实点页面验证，不只 curl HTML。 |
| **P50 / P95** | 50%/95% 请求的延迟分位。 | 比平均值更能反映体验。 |
| **Concurrency（并发）** | 同时处理的请求数。 | 压测关注吞吐和延迟。 |

---

## 6. 名词表：Redis、额度、测试、Git

| 名词 | 含义 | 为什么重要 |
|---|---|---|
| **Redis** | 内存键值存储。 | 额度计数、session、缓存。 |
| **Lua** | Redis 内嵌脚本。 | 把“检查 + 增加 + 过期”做成原子操作。 |
| **Atomicity（原子性）** | 一组操作要么全做、要么全不做。 | 防止并发请求同时通过额度检查。 |
| **TTL** | key 过期时间。 | 每日额度自动过期，不用手动清理。 |
| **Quota（额度）** | 每天允许的请求数/token 数。 | 免费额度/欠费不可控时，应用层硬停。 |
| **Reservation（预占）** | 调用前先扣估算 token。 | 防止并发超支；失败也会消耗预占。 |
| **Fail-closed** | 依赖故障时拒绝服务，而不是放行。 | Redis 挂了返回 503，不绕过额度。 |
| **Smoke test（冒烟）** | 少量固定用例验证主链路。 | 5 条 query 只证明链路通，不证明质量。 |
| **Regression test（回归）** | 防止已修 bug 再次出现。 | 例如 overlap 跨代码块、废弃误报。 |
| **`pytest`** | Python 测试框架。 | 单测/集成测试。 |
| **`ruff`** | Python lint/format。 | 代码风格和静态问题。 |
| **`mypy`** | Python 静态类型检查。 | 提前发现类型错误。 |
| **Git commit** | 一次版本快照。 | 可回滚、可审计。 |
| **Git push** | 推送到远端。 | 当前只推 C 的 `origin/main`。 |
| **`origin/main`** | 远端主分支。 | C 仓库的 GitHub 上游。 |

---

## 7. Worklog 里的行为逐段解释

### 7.1 `docker compose up -d qdrant`

- **做什么**：按 `docker-compose.yml` 启动 Qdrant 服务，`-d` 是后台运行。
- **为什么**：向量库要独立进程，API 通过 HTTP 访问。
- **原理**：Compose 先找镜像；本地没有就拉取；创建容器/网络/volume；映射端口 6333/6334。
- **卡住的原因**：Docker Hub 直连或镜像层下载慢，不是 Compose 语法错。
- **验证**：`docker compose ps`、`docker images`、`readyz`。

### 7.2 `docker pull ...` + `docker tag ...`

- **做什么**：从镜像加速地址拉取，然后改成本地标准名字。
- **为什么**：Compose 里写的是 `qdrant/qdrant:latest`，但直连拉不下来。
- **原理**：镜像名只是本地标签；同一份镜像可以有多个 tag。

### 7.3 `python scripts/ingest.py --limit 3`

- **做什么**：只取前 3 篇文档，跑完整离线链路。
- **为什么**：先验证 Key、网络、Qdrant、alias，不浪费全量 embedding。
- **原理**：解析 -> 切分 -> 只对 child 做 embedding -> 写 build collection -> 切 alias。
- **结果**：3 篇、20 chunks、15 children、15 vectors。

### 7.4 全量 `ingest.py`

- **做什么**：155 篇文档、1661 chunks、1363 children。
- **为什么**：只有全量索引，固定 query 才有意义。
- **结果**：1363 个向量写入 Qdrant，alias 指向新 build。

### 7.5 `smoke_search.py`

- **做什么**：5 条固定 query，打印 top-3 URL 和分数，和期望 URL 比较。
- **为什么**：接 LLM 前先验证检索；否则 chat 错了不知道是哪层。
- **结果**：5/5 命中。
- **局限**：只证明链路和基本语义，不是 Recall 评估。

### 7.6 `check_providers.py`

- **做什么**：真实调用一次 embedding 和一次 chat。
- **为什么**：区分 Key 错、模型未开通、限流、余额。
- **结果**：embedding OK；qwen-turbo 429；qwen-plus/flash OK。

### 7.7 `/api/v1/chat`

- **做什么**：检索 top-k -> 拼上下文 -> LLM -> 返回 answer + sources。
- **为什么**：把“检索 demo”变成“问答产品”。
- **原理**：模型只输出 `[n]`；后端用编号映射 URL，避免模型编 URL。
- **拒答**：没有命中时直接返回 `REFUSAL`，不调用 LLM。

### 7.8 `npm run build`

- **做什么**：TypeScript 检查 + Vite 构建，生成 `dist`。
- **为什么**：生产不能跑 `vite dev`；Nginx 只发静态文件。
- **结果**：`dist/index.html` + JS/CSS。

### 7.9 CORS 问题

- **现象**：浏览器从 `127.0.0.1:5173` 调 `localhost:8000` 被拦。
- **原理**：浏览器同源策略；`localhost` 和 `127.0.0.1` 是不同 origin。
- **修复**：CORS 同时允许两个 origin；生产用 Nginx 同源反代，不再跨域。

### 7.10 Vite `EBUSY`

- **现象**：编辑文件时 watcher 监控临时目录被 Windows 锁住。
- **原理**：编辑器原子写入产生 `*.tmpdir`，Vite watcher 尝试 watch 时 EBUSY。
- **修复**：忽略这些临时目录。

### 7.11 `docker compose build api`

- **做什么**：构建 API 镜像。
- **为什么**：生产用容器，不依赖宿主机 Python 环境。
- **卡点**：基础镜像拉取慢、容器内 PyPI 源不可用。
- **解决**：镜像加速 + `PIP_INDEX_URL=https://pypi.org/simple`。
- **优化**：Dockerfile 先装依赖（stub package），再复制源码 `pip install --no-deps`。

### 7.12 `load_search.py`

- **做什么**：8 个并发请求 `/api/v1/search`。
- **为什么**：验证检索链路在并发下是否可用。
- **结果**：8 × 200；p50 292ms、p95 316ms、max 360ms。
- **局限**：不是 chat 并发；chat 还受 LLM 限流。

### 7.13 Redis 额度锁

- **做什么**：每次 chat 前预占请求数和 token。
- **为什么**：平台没有 0.01 元硬停；免费额度也会被长上下文烧完。
- **原理**：Redis Lua 原子检查 + INCR + EXPIRE；超限 429，Redis 挂 503。
- **验证**：真实 chat 后 key 为 requests=1/tokens=4300；强制 token 到上限后 429。

### 7.14 Nginx 生产前端

- **做什么**：`dist` 交给 Nginx；`/api` 反代到 API。
- **为什么**：不暴露 dev server；同源访问避免 CORS。
- **结果**：`http://localhost:8080` 可访问，Playwright 点查询正常。

---

## 8. 今天之前的改动：理解、含义、为什么

### 8.1 10 月下旬：设计文档 + Phase 0 骨架

提交：

- `733d8a6 Initial commit: project design and RAG principles`
- `fd4cbac Switch project docs to FastAPI documentation knowledge base`
- `c7d9aeb Add Phase 0 skeleton and FastAPI Markdown parser`

含义：

- 项目从“通用 LangChain RAG 教程”转向“FastAPI 官方文档”这一具体知识库。
- 设计文档明确项目签名是**版本感知的活知识库**；hybrid + rerank 是基线；评估 + bad case 是证明手段。
- Phase 0 只搭骨架和解析器，不堆功能；这是正确的顺序，因为解析错会导致后面全错。

### 8.2 11 月 13 日：parser 增强

提交 `dbe686f`：

- 带标题/缩进/嵌套的 admonition
- `//// tab` 多版本代码块
- strong/weak 废弃信号
- 代码围栏内不误判标题

为什么：

- 真实 FastAPI 文档有 `/// note | Technical Details`、`//// tab | Python 3.10+`；
- 不处理这些，提示块类型和版本信息会丢；
- 不处理代码围栏，代码里的 `# comment` 会被误认为标题。

### 8.3 11 月 13 日：chunker

提交 `dbe686f`：

- parent/child 切分
- 代码块原子
- 标题头注入
- 稳定 chunk ID
- overlap

为什么：

- child 用于检索，parent 用于生成；解决“检索要小、生成要大”。
- 代码块切断会不可运行；标题头能让 embedding 知道语义归属。
- 稳定 ID 是增量索引的基础。

### 8.4 11 月 13 日：五个 bug 修复

提交 `447bb80`：

| 问题 | 现象 | 根因 | 修复 |
|---|---|---|---|
| overlap 跨代码块 | child fence 不配对，代码被切成两半 | overlap 从整组文本尾部取，穿过 code block | 只从末尾 prose 取 |
| 段落分隔丢失 | 两个段落粘成一行 | `_join` 对 prose 用 `\n` | 引入 block separator |
| 废弃误报 | “Deprecating parameters” 被标成 deprecated | 标题子串规则太粗 | 区分教学型/参数级/真废弃 |
| parent 粒度太小 | parent 中位数 133 token | 一个 section 一个 parent | 跨 section 打包 |
| release notes 占比 42% | 1830 chunks 来自 changelog | 全量版本进索引 | 只留最近 20 个版本 |

为什么重要：

- 这些不是“代码风格”，而是会污染索引、影响引用和评估的语义问题；
- 例如“教你怎么废弃”被标成 deprecated，会让回答错误地警告用户；
- release notes 不限制会淹没普通问答。

### 8.5 11 月 13 日：embedding + indexer

提交 `f8a72be`：

- OpenAI 兼容 embedding 客户端
- source metadata 检测
- Qdrant build collection + payload + alias

为什么：

- 把“文本块”变成“可检索向量”；
- 版本字段从采集/索引阶段就写入，避免后期返工；
- 临时 collection + alias 切换，保证查询不看到半成品。

### 8.6 11 月 14 日上午：dense retrieval

提交 `7db57f8`：

- `DenseRetriever`
- `/api/v1/search`
- 5 条固定 query smoke

为什么：

- 先证明检索，再证明生成；
- `/search` 是调试/评估接口，不依赖 LLM。

### 8.7 11 月 14 日：chat / 前端 / Docker / 额度 / Nginx

就是今天 Worklog 的内容；核心演进：

```text
检索 -> 生成 -> 前端 -> Docker -> 额度锁 -> 生产静态托管
```

每一步都在把“能跑”往“能演示、能控制成本、能部署”推进。

---

## 9. 常见误解

### 9.1 RAG = 搜索？

不是。搜索只返回文档片段；RAG 还要让 LLM 基于片段生成答案，并处理引用/拒答。

### 9.2 Embedding = LLM？

不是。Embedding 把文本变向量；LLM 生成文本。DeepSeek 有 LLM 但没有 embedding 服务，所以要分开配置。

### 9.3 Qdrant = 文档数据库？

不是。Qdrant 是向量数据库；payload 可以存文本，但它的核心是相似度检索和过滤，不是事务/关系查询。

### 9.4 Redis = 前端多端方案？

不是。Redis 是后端共享状态；多端靠 Capacitor/Tauri 等壳，不靠 Redis。

### 9.5 Vue3 = 多端？

不是。Vue3 只是另一种前端框架；换它不会自动生成 Android/iOS。多端靠同一套 Web 构建 + 原生壳。

### 9.6 dev server = 生产？

不是。Vite dev server 适合开发，不适合公网；生产必须 `npm run build` + Nginx/Caddy。

### 9.7 5/5 smoke = 质量好？

不是。5 条只是冒烟；没有 20–30 条 golden set、Recall、引用/拒答评估，不能声称质量。

### 9.8 免费额度 = 无限/无限流？

不是。免费额度是 token 费用，仍有 RPM/TPM 限流；`qwen-turbo` 429 就是限流。

### 9.9 `source_ref=unknown` = 版本感知完成？

不是。版本签名需要真实 commit SHA 和变更检测；当前 C 数据没有 `.git`，所以版本归因还不完整。

---

## 10. 当前实现水平

### 已完成

- 解析/include/admonition/anchor/URL 映射
- 父子切分、代码块保护、标题头、稳定 ID
- embedding + Qdrant 全量索引 1363 向量
- dense retrieval + `/api/v1/search`
- `/api/v1/chat` + 引用 + 拒答
- React 前端 + Nginx 静态托管
- Docker API + Qdrant + Redis + Web
- Redis 每日 LLM 额度锁
- 基础测试、lint、type check、5 条 smoke、8 并发检索

### 未完成

- 20–30 条 golden set 与检索/生成指标
- 引用检查（目前是后端映射，但没有逐句校验）
- SSE 流式、多轮、session
- hybrid/BM25/RRF/rerank
- 增量索引、版本归因、陈旧度
- HTTPS/域名/鉴权/HTTP 限流
- embedding 额度计数
- 最终 Dockerfile 的干净构建验证
- Capacitor Android/iOS、Tauri/Electron

---

## 11. 怎么用这份文档

- 想理解“今天做了什么”：看 [WORKLOG_2026-11-14.md](WORKLOG_2026-11-14.md)。
- 想理解“这个词是什么意思”：看第 1–6 节的名词表。
- 想理解“为什么这么做”：看第 7 节的行为解释和第 8 节的历史。
- 想理解“现在能不能上线”：看第 9–10 节，以及 Worklog 的“风险/下一步”。
- 想理解整体设计：回到 `docs/PROJECT_DESIGN.md` 和 `docs/RAG_PRINCIPLES.md`。

一句话总结：

> **今天把一条“文档 -> 向量 -> 检索 -> 回答 -> 前端 -> Docker -> 额度锁 -> Nginx”的真实链路跑通了；但“能演示”和“生产上线”之间，还差评估、HTTPS、鉴权、数据版本和干净的部署验证。**
