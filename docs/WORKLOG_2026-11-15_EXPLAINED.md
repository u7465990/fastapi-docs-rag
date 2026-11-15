# 2026-11-15 工作记录 · 新名词与原理注解

> 配套文档：[WORKLOG_2026-11-15.md](WORKLOG_2026-11-15.md)
>
> 昨天的名词表见：[WORKLOG_2026-11-14_EXPLAINED.md](WORKLOG_2026-11-14_EXPLAINED.md)
>
> 本文只解释 **11 月 15 日新增/新涉及** 的概念：评估、HTTP 安全、SSE、多轮、可复现构建、观测、高并发。
> 目标仍是：先说“是什么”，再说“为什么”，最后说“在项目里怎么用”。

---

## 0. 今天新增内容的全局图

```text
评估
  golden set -> evaluate.py -> Recall/MRR -> bad case -> 过滤/标注修正

安全
  request -> access token -> rate limit -> embedding quota -> LLM quota

流式
  /chat/stream -> sources -> token* -> usage -> done

多轮
  session_id -> Redis history -> prompt history -> save turn

构建
  Dockerfile.web: node build -> nginx runtime
  Dockerfile.api: deps layer -> source layer

观测
  request_id -> JSON log -> /metrics -> /readyz
```

---

## 1. 评估相关名词

### 1.1 Golden set（黄金集）

- **含义**：人工标注的问题 + 期望答案/来源集合。
- **为什么**：没有标注，就无法客观判断“检索变好还是变差”。
- **本项目**：`data/eval/golden_set.jsonl`，30 条中文问题，英文文档。
- **字段**：
  - `id`：稳定编号
  - `question`：用户问题
  - `expected_urls`：一个或多个正确文档 URL
  - `category`：tutorial / advanced / how-to / deployment / security / multi-hop / version
  - `language`：问题语言，当前全部 `zh`

### 1.2 Expected URLs（期望 URL）

- **含义**：这条问题至少应该命中哪些文档。
- **为什么用 URL 而不是 chunk id**：URL 是用户能验证的最终出处；chunk 会随切分变化。
- **允许多个**：同一个问题可能有多个正确页面，例如 204 状态码同时存在于 tutorial 和 advanced。
- **命中规则**：检索结果的 `url` 包含任意一个 expected URL，即算命中。

### 1.3 Recall@k（召回率）

- **含义**：前 k 条结果里，至少有一条命中期望 URL 的问题比例。
- **公式**：`命中问题数 / 总问题数`。
- **本项目**：Recall@1 / @5 / @10。
- **为什么**：衡量“正确证据有没有被找出来”。
- **注意**：Recall@5 = 1.0 不代表答案好，只代表这 30 条都能在前 5 找到。

### 1.4 MRR（Mean Reciprocal Rank）

- **含义**：每条问题命中排名的倒数，再取平均。
- **公式**：`mean(1 / rank)`；没命中记 0。
- **例子**：
  - 第 1 名命中：1.0
  - 第 2 名命中：0.5
  - 第 3 名命中：0.333
  - 没命中：0
- **为什么**：Recall 只看“有没有”，MRR 还看“排得靠不靠前”。
- **本项目**：最终 MRR 0.953，说明大多数命中在第 1，少数在第 3/4。

### 1.5 Rank（排名）

- **含义**：期望 URL 第一次出现在第几条。
- **为什么记录**：`rank=1` 和 `rank=5` 都算 Recall@5 命中，但体验差别很大。
- **Near miss**：rank 2–5 的命中，不是完全失败，但排序不够好。

### 1.6 Miss / Near miss

- **Miss**：前 k 条完全没有期望 URL。
- **Near miss**：前 k 条有，但不在第 1。
- **本项目的 near miss**：
  - OAuth2 + JWT 问题被 `simple-oauth2` 抢到 rank 3
  - `OAuth2PasswordBearer + JWT` 多跳题 rank 4
- **为什么单独记录**：它们是 rerank/hybrid 的目标，不应该混在“miss”里。

### 1.7 Annotation error（标注错误）

- **含义**：golden set 的期望答案写错了或不完整，导致检索被误判。
- **本项目例子**：204 问题只标了 tutorial，漏了 advanced。
- **为什么重要**：指标变化可能来自“标注变准”，不是“代码变好”。
- **处理**：修正 `expected_urls`，并在 bad case 里明确写“这是标注修正”。

### 1.8 Reference page（参考页）

- **含义**：`docs/en/docs/reference/**`，由 mkdocstrings 生成，内容偏 API 签名/参数表。
- **为什么默认排除**：
  - 它和 tutorial 语义很近，容易抢普通问答的 top-1
  - 设计文档规定 reference 默认不参与正文检索
- **实现**：`build_filter(include_reference=False)` 默认 `must_not doc_type=reference`。
- **开关**：`include_reference=true` 可以显式打开。

### 1.9 Evaluation harness（评估脚手架）

- **含义**：可重复运行的评估脚本，而不是手工看几条结果。
- **本项目**：`scripts/evaluate.py`。
- **为什么**：后续改过滤、rerank、hybrid、prompt，都可以跑同一套数据对比。

### 1.10 Baseline（基线）

- **含义**：优化前的对照结果。
- **本项目**：dense retrieval + 当前索引 + 默认过滤。
- **为什么**：没有 baseline，“优化”无法证明有效。

---

## 2. HTTP 安全与额度名词

### 2.1 Rate limit（限流）

- **含义**：限制单位时间内允许的请求数。
- **本项目**：每个 IP、每个 endpoint、每分钟 30 次。
- **为什么**：防止刷接口、保护 embedding/LLM 额度、避免单用户拖垮服务。
- **实现**：Redis key `rate:{scope}:{identifier}:{minute}`。

### 2.2 Fixed window（固定窗口）

- **含义**：按自然分钟计数，例如 13:01:00–13:01:59 一个窗口。
- **优点**：实现简单，Redis `INCR + EXPIRE`。
- **缺点**：窗口边界可能短时双倍流量。
- **替代**：滑动窗口、令牌桶；当前规模固定窗口够用。

### 2.3 Scope（限流作用域）

- **含义**：限流按哪个维度分组。
- **本项目**：`search` / `chat` 分开，各自每分钟 30 次。
- **为什么**：检索和问答成本不同，不能共用同一个计数。

### 2.4 429 Too Many Requests

- **含义**：请求过多/额度耗尽。
- **本项目两种 429**：
  - 限流超限
  - 每日 LLM 或 embedding 额度耗尽
- **前端**：应展示可读错误，不要静默失败。

### 2.5 503 Service Unavailable

- **含义**：依赖服务不可用。
- **本项目**：
  - Redis 不可用 -> 限流/额度无法判断 -> 503
  - Qdrant 不可用 -> 检索失败 -> 503
- **原则**：fail-closed，不绕过限制。

### 2.6 Fail-closed（故障关闭）

- **含义**：安全依赖故障时，拒绝服务，而不是放行。
- **为什么**：Redis 挂了就绕过额度，可能导致费用失控。
- **代价**：可用性下降；但上线初期更安全。

### 2.7 Access token（访问令牌）

- **含义**：调用 API 需要携带的共享密钥。
- **本项目**：
  - `Authorization: Bearer <token>`
  - 或 `X-API-Key: <token>`
- **配置**：`API_ACCESS_TOKEN` 为空时本地不鉴权。
- **限制**：共享 token 不是用户登录；前端 `VITE_API_TOKEN` 会暴露在 bundle 里。

### 2.8 Embedding quota（embedding 额度）

- **含义**：每天允许多少次 query embedding / 多少 token。
- **为什么**：`/search` 和 `/chat` 都会调用 embedding；只锁 LLM 不够。
- **本项目配置**：
  - `EMBEDDING_DAILY_REQUEST_LIMIT=500`
  - `EMBEDDING_DAILY_TOKEN_BUDGET=200000`
- **计数键**：`quota:embedding:requests:{day}`、`quota:embedding:tokens:{day}`。

### 2.9 Reservation（预占）

- **含义**：调用前先按估算值扣额度，而不是调用后按实际 usage 扣。
- **为什么**：并发下防止超支；不需要等 provider 返回 usage。
- **代价**：估多了会浪费额度；失败也会占用。
- **本项目估算**：
  - embedding：`len(query)//3 + 16`
  - chat：`sum(len(message))//3 + max_output`

### 2.10 Redis Lua 原子操作

- **含义**：把“检查 + 增加 + 设置 TTL”放在一段 Redis 脚本里执行。
- **为什么**：多个请求并发时，先读后写会超卖。
- **本项目**：Lua 检查 request/token 上限，满足才 `INCR/INCRBY`，否则返回拒绝。

### 2.11 TTL（Time To Live）

- **含义**：key 的过期时间。
- **本项目**：
  - 额度 key：2 天
  - 限流 key：90 秒
  - session：7 天
- **为什么**：自动清理，不需要定时任务。

---

## 3. SSE 与流式名词

### 3.1 SSE（Server-Sent Events）

- **含义**：服务器通过 HTTP 持续向浏览器推送文本事件。
- **协议格式**：

```text
event: token
data: {"text":"Hello"}

```

- **为什么用**：LLM 生成是渐进的，SSE 比 WebSocket 简单，比轮询自然。
- **本项目 endpoint**：`POST /api/v1/chat/stream`。

### 3.2 `text/event-stream`

- **含义**：SSE 的 MIME 类型。
- **为什么**：告诉浏览器/代理这是流式事件，不要按普通 JSON 缓存。

### 3.3 Event / Data

- **event**：事件名，如 `sources`、`token`、`usage`、`done`、`error`。
- **data**：JSON 字符串。
- **本项目顺序**：

```text
sources -> token* -> usage -> done
```

- **为什么 sources 先发**：用户可以先看到出处，再等答案。

### 3.4 Token event

- **含义**：一小段增量文本。
- **注意**：这里的 token 是“文本块”，不是计费 token；一个事件可能包含多个字/词。

### 3.5 Usage event

- **含义**：token 用量。
- **本项目现状**：`{"input_tokens": null, "output_tokens": null}`，因为 OpenAI 兼容流不保证返回 usage。
- **为什么保留**：SSE 契约固定；后续 provider 支持 usage 时可以填真实值。

### 3.6 Done / Error event

- **done**：正常结束，带 model/session_id。
- **error**：生成中途失败，显式通知前端，不能静默截断。
- **为什么**：流式连接断了，HTTP 状态码可能已经是 200，必须靠事件表达错误。

### 3.7 `fetch` + `ReadableStream`

- **含义**：浏览器用 `fetch` 拿响应体，再用 `reader.read()` 逐块读取。
- **为什么不用 `EventSource`**：`EventSource` 只支持 GET；我们的 chat 是 POST，还要带 JSON body。
- **解析难点**：网络分块不一定按 event 边界，需要缓冲区按 `\n\n` 切。

### 3.8 `proxy_buffering off`

- **含义**：Nginx 不把上游响应攒满再发。
- **为什么**：不关的话，SSE 会被整段缓冲，用户看不到逐字效果。
- **本项目**：`docker/nginx.conf` 的 `/api/` location 已关闭，并加 `X-Accel-Buffering: no`。

### 3.9 Chunk（流式分块）

- **含义**：HTTP 响应体被分成多个 chunk。
- **为什么**：前端不能假设“一次 read 就是一个完整 event”，必须自己缓冲。

---

## 4. 多轮会话名词

### 4.1 Session（会话）

- **含义**：一次连续对话的上下文容器。
- **本项目**：一个 `session_id` 对应 Redis 里的一组消息。

### 4.2 `session_id`

- **含义**：前端生成并保存的会话标识（UUID）。
- **本项目**：
  - 前端 `localStorage`
  - 请求体传给 `/chat` 或 `/chat/stream`
  - 不传就是单轮
- **限制**：它不是登录身份；知道 session_id 的人可以读历史。

### 4.3 History（历史）

- **含义**：之前的 user/assistant 消息。
- **本项目**：
  - Redis `chat:session:{id}:messages`
  - 最多 12 条消息（约 6 轮）
  - TTL 7 天
- **为什么限制条数**：控制 prompt token，避免历史淹没当前文档上下文。

### 4.4 Prompt history injection（历史注入）

- **含义**：把历史消息放在 system prompt 之后、当前 user prompt 之前。
- **本项目顺序**：

```text
system
history(user/assistant)*
current user（带本轮文档上下文）
```

- **关键约束**：历史只帮助理解指代；事实仍以本轮检索上下文为准。
- **为什么**：历史回答里的 `[n]` 编号属于旧上下文，混用会导致引用错。

### 4.5 Session TTL

- **含义**：会话自动过期时间。
- **本项目**：7 天（604800 秒）。
- **为什么**：避免 Redis 无限增长；过期后前端还能用新 session 继续。

### 4.6 Query rewrite（查询改写）

- **含义**：把“那它呢？”改写成完整问题，再做检索。
- **本项目现状**：未实现。
- **为什么需要**：多轮记忆解决了“模型知道上文”，但检索仍只拿当前短 query，容易跑偏。
- **候选**：用便宜小模型改写、规则拼接上一轮问题、多 query 检索。

---

## 5. 可复现构建名词

### 5.1 Multi-stage build（多阶段构建）

- **含义**：一个 Dockerfile 里多个 `FROM`，前一个阶段构建，后一个阶段只复制产物。
- **本项目**：
  - `build` 阶段：Node + npm ci + npm run build
  - runtime 阶段：Nginx + 复制 `dist`
- **为什么**：运行镜像不需要 Node/npm/源码，体积小、攻击面小。

### 5.2 Build stage / Runtime stage

- **build stage**：安装构建依赖、编译前端。
- **runtime stage**：只保留运行所需文件。
- **本项目 runtime**：`nginx:1.27-alpine` + `dist` + `nginx.conf`。

### 5.3 `npm ci`

- **含义**：按 `package-lock.json` 精确安装依赖。
- **为什么不用 `npm install`**：`npm ci` 更可复现，适合 CI/Docker。
- **要求**：必须有 `package-lock.json`。

### 5.4 `dist`

- **含义**：Vite 构建产物目录。
- **本项目**：`apps/web/dist`。
- **过去**：Dockerfile 直接复制本机 `dist`，新机器没有就失败。
- **现在**：Docker build stage 内部生成 `dist`，不依赖本机。

### 5.5 Clean build（干净构建）

- **含义**：不复用旧镜像层，`docker build --no-cache`。
- **为什么**：验证“从零能不能构建”，而不是“缓存刚好能过”。
- **本项目**：
  - API `--no-cache` 约 226 秒
  - Web `--no-cache` 成功

### 5.6 Layer cache（层缓存）

- **含义**：Docker 复用未变化的层。
- **为什么**：改源码不应重装全部依赖。
- **本项目 API**：
  1. 先复制 `pyproject.toml` + stub package 安装依赖
  2. 再复制 `src`，`pip install --no-deps`
- **效果**：改业务代码只重跑最后一层。

### 5.7 Image tag（镜像标签）

- **含义**：给镜像起名字/版本，如 `fastapi-docs-rag-api:clean`。
- **为什么**：区分“干净构建验证镜像”和日常 `latest`；生产应 pin 版本，不用 `latest`。

### 5.8 `.dockerignore`

- **含义**：构建上下文排除文件。
- **本项目**：排除 `.env`、`.git`、`node_modules`、`apps/web/dist`、tests 等。
- **为什么**：减小上下文、避免泄露密钥、避免把本机产物带进镜像。

### 5.9 Reproducible build（可复现构建）

- **含义**：换一台机器、清空缓存，也能按同一份代码构建出可运行镜像。
- **本项目现状**：API/Web 干净构建都跑通；仍依赖网络拉基础镜像和依赖。
- **真正空机器还需要**：`.env`、Qdrant 索引、域名/证书。

---

## 6. 观测名词

### 6.1 Request ID（请求 ID）

- **含义**：给每个请求分配唯一标识，贯穿日志/追踪。
- **本项目**：`X-Request-ID`，不传自动生成 `req-<uuid>`。
- **为什么**：线上排查时，可以把一条日志和一次请求对应起来。

### 6.2 Structured logging（结构化日志）

- **含义**：日志是 JSON，而不是不可解析的字符串。
- **本项目字段**：`ts`、`level`、`event`、`request_id`、`method`、`path`、`status`、`duration_ms`、`client`。
- **为什么**：方便 Loki/ELK/CloudWatch 检索和聚合。

### 6.3 Metrics（指标）

- **含义**：可聚合的数值，如请求数、延迟、错误数。
- **和日志的区别**：日志是单次事件；指标是统计量。
- **本项目**：`/metrics` 输出 Prometheus 文本格式。

### 6.4 Counter（计数器）

- **含义**：只增不减的累计值。
- **本项目**：
  - `rag_http_requests_total`
  - `rag_tokens_reserved_total`
  - `rag_quota_rejections_total`
- **为什么**：适合算 QPS、错误率、总量。

### 6.5 Summary / Quantile（分位数）

- **含义**：p50 是中位数，p95 是 95% 请求不超过的值。
- **为什么**：平均延迟会掩盖长尾；p95 更接近用户体验。
- **本项目**：内存窗口采样，输出 p50/p95。

### 6.6 Prometheus exposition format

- **含义**：Prometheus 能抓取的文本格式。
- **例子**：

```text
rag_http_requests_total{method="POST",path="/api/v1/chat/stream",status="200"} 1
rag_tokens_reserved_total{kind="llm"} 4324
```

- **为什么**：Grafana/Prometheus 可以直接消费。

### 6.7 Readiness / Liveness

- **liveness**：进程还活着吗？`/healthz`。
- **readiness**：现在能接流量吗？`/readyz`。
- **本项目 `/readyz`**：
  - config
  - Qdrant 真实连接
  - Redis 真实连接
  - LLM 只检查 key+model 配置

### 6.8 Config-only health（仅配置检查）

- **含义**：只验证 key/model 是否存在，不发真实 LLM 请求。
- **为什么**：readiness 可能每 10 秒跑一次；每次都发 completion 会烧钱、触发限流。
- **代价**：不能发现“key 对但余额不足/网络不通”的问题；需要单独的告警或低频探针。

### 6.9 Multi-process metrics（多进程指标）

- **含义**：多个 worker 各自维护指标，需要聚合。
- **问题**：当前 `/metrics` 是进程内内存；`uvicorn --workers 4` 时每个 worker 一份数据。
- **解决**：Prometheus client 的 multiprocess 模式，或每个 worker 单独 scrape 后在 Prometheus 聚合。

### 6.10 Grafana / Prometheus

- **Prometheus**：定时抓取 `/metrics`，存储时间序列。
- **Grafana**：查询 Prometheus，画面板和告警。
- **本项目现状**：还没有部署；只有自定义 `/metrics`。

---

## 7. 高并发名词

### 7.1 Concurrency vs Throughput

- **并发**：同时有多少请求在处理。
- **吞吐**：每秒完成多少请求。
- **100 并发**不等于 100 QPS；聊天请求持续几十秒，吞吐可能只有几个/秒。

### 7.2 Worker（工作进程）

- **含义**：uvicorn 启动的独立进程。
- **为什么**：绕过 Python GIL，利用多核。
- **代价**：每个 worker 独立内存；metrics/缓存/连接池不共享。

### 7.3 Thread pool（线程池）

- **含义**：FastAPI 对同步 `def` 端点用线程池执行。
- **本项目**：端点大多是同步函数；streaming generator 也走线程。
- **瓶颈**：默认线程池有限，100 个长连接会排队；需要多 worker 或 async 改造。

### 7.4 Semaphore / Queue（信号量/队列）

- **含义**：限制同时进行的操作数，超出的排队。
- **为什么**：LLM provider 有 RPM/TPM；100 个请求同时打过去只会一起 429。
- **本项目现状**：未实现全局 LLM 并发控制；只有每日额度。

### 7.5 RPM / TPM

- **RPM**：Requests Per Minute，每分钟请求数。
- **TPM**：Tokens Per Minute，每分钟 token 数。
- **为什么重要**：免费额度和付费额度都可能有限流；应用层 worker 再多也没用。

### 7.6 Backpressure（背压）

- **含义**：下游处理不过来时，上游要减速/排队/拒绝，而不是无限堆积。
- **本项目**：429 + `Retry-After`，或队列等待超时，是背压手段。

### 7.7 Load test（压测）

- **含义**：模拟并发请求，测延迟/错误率/吞吐。
- **本项目**：已有 8 路 `/search` 压测；chat 100 并发还没做。
- **建议**：用 fake LLM 压 API/SSE，不直接用真实模型烧钱。

### 7.8 Fake LLM

- **含义**：测试用假模型客户端，固定延迟/固定回答。
- **为什么**：压测 API、Nginx、Redis、Qdrant，不受 provider 限流和费用影响。
- **真实 LLM**：只做小流量验证。

---

## 8. LangSmith / Tracing 名词

### 8.1 Tracing（追踪）

- **含义**：记录一次请求内部的步骤：检索、prompt、LLM 调用、输出、耗时。
- **为什么**：比日志更细，能看“哪一步慢/错”。

### 8.2 Run / Span

- **Run**：LangSmith 里一次被追踪的调用。
- **Span**：一次调用内部的子步骤。
- **本项目**：可以给检索、chat completion 各建一个 run。

### 8.3 LangSmith

- **含义**：LangChain 的云端 tracing/评估平台。
- **接入方式**：
  - 用 LangChain/LangGraph 时，设 `LANGCHAIN_TRACING_V2=true` 自动上报
  - 裸 `httpx` 调用不会自动上报，需要 `@traceable` 或 `RunTree`
- **本项目现状**：chat 是裸 httpx，所以只设环境变量不够。
- **流式限制**：生成器追踪不完整；可以先 trace 检索和非流式 complete，或手工 RunTree。

### 8.4 `@traceable`

- **含义**：LangSmith 提供的装饰器，把函数输入/输出/耗时包成一个 run。
- **用法**：

```python
from langsmith import traceable

@traceable(run_type="llm", name="qwen-chat")
def complete(messages):
    ...
```

- **注意**：会向 LangSmith 发送输入/输出，注意隐私和费用。

---

## 9. 今天没有做的

- HTTPS / 域名 / 端口收紧
- 真实用户登录
- 付费 LLM 额度与全局并发队列
- Prometheus / Grafana 部署
- LangSmith 接入
- 生成质量评估（引用/拒答）
- 多轮 query rewrite
- rerank / hybrid
- 增量索引 / `source_ref` 修复

这些不是“忘了”，而是按优先级排在后面；今天先把评估、安全、流式、多轮、构建、观测做出来。

---

## 10. 怎么读这两份文档

- 看今天做了什么、命令和结果：`WORKLOG_2026-11-15.md`
- 看今天新名词是什么意思：本文
- 看昨天的历史和基础概念：`WORKLOG_2026-11-14.md` / `WORKLOG_2026-11-14_EXPLAINED.md`
- 看设计事实：`docs/PROJECT_DESIGN.md`

一句话：

> 今天新增的名词，核心只有三组：**怎么证明检索有效（Recall/MRR/golden set）**、**怎么在公网前控制成本和滥用（限流/额度/token/429）**、**怎么让系统可观测、可复现、可流式、多轮（SSE/session/Docker multi-stage/request_id/metrics）**。
