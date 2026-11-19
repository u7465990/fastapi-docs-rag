# 并发与延迟基线

本文件记录本地 Docker 环境下的真实压测，不把“改了 async”当成“支持高并发”。

## 环境

```text
Windows 10 / Docker Desktop
i7-11390H (4C/8T), 16GB
API / Qdrant / Redis 均在本机 Docker
测试脚本：scripts/load_chat.py
```

测试使用 `X-Real-IP` 为每个请求分配独立 IP，避免把限流 429 误判成吞吐瓶颈。  
真实公网环境必须把该头限制为可信代理设置，不能让客户端任意伪造。

## 改造前（同步 API）

检索：

```text
20 并发 / 20 请求
p50: 444.72ms
p95: 496.46ms
p99: 582.73ms
rps: 26.88
错误: 0
```

Chat：

```text
5 并发 / 5 请求
p50: 4596.61ms
p95: 6337.93ms
rps: 0.79
错误: 0
```

## 改造后（async API）

检索：

```text
20 并发 / 20 请求
p50: 372.73ms
p95: 532.58ms
p99: 766.72ms
rps: 24.54
错误: 0
```

检索 100 并发：

```text
100 并发 / 100 请求
p50: 2118.54ms
p95: 3700.18ms
p99: 3811.88ms
rps: 25.11
status: 100 x 200
错误: 0
```

100 并发检索没有报错，但 p95 从 20 并发的 533ms 升到 3.7s。  
说明当前 16 个 retrieval 并发槽 + embedding/Qdrant 上游开始排队，吞吐没有随并发线性增长。

尝试把 `RETRIEVAL_MAX_CONCURRENCY` 从 16 调到 32：

```text
100 并发 / 100 请求
p50: 2486ms
p95: 4539ms
rps: 21.0
错误: 0
```

p95 反而从 3.70s 升到 4.54s，吞吐从 25.1 降到 21.0。  
结论：当前环境下提高 retrieval 并发只会把更多压力推给 embedding/Qdrant，**保留 16**，不为了数字好看盲目加并发。

Chat：

```text
5 并发 / 5 请求
p50: 5198.97ms
p95: 5416.14ms
rps: 0.92
错误: 0
```

Chat 10 并发：

```text
10 并发 / 10 请求
p50: 4642.46ms
p95: 8616.13ms
rps: 1.16
错误: 0
```

Chat 20 并发（独立临时 API，额度临时提高，仅为压测）：

```text
20 并发 / 20 请求
p50: 24508.55ms
p95: 29098.49ms
rps: 0.68
status: 20 x 200
错误: 0
```

20 并发没有报错，但 p95 29s，已经不可用。  
瓶颈是 `LLM_MAX_CONCURRENCY=8` + provider 在高压下的尾延迟；不是 FastAPI 线程池。  
**不能把这个结果说成“支持 20 并发低延迟”。**

SSE：

```text
5 并发 / 5 请求
总延迟 p95: 4840.21ms
首 token p50: 730.37ms
首 token p95: 1097.31ms
错误: 0
```

## 阶段耗时

`/api/v1/search` 和 `/api/v1/chat` 的响应包含 `timings`，例如：

```json
{
  "setup_ms": 0.1,
  "embedding_ms": 233.4,
  "dense_ms": 4.8,
  "bm25_ms": 2.8,
  "rerank_ms": 0.4,
  "retrieval_ms": 1186.4
}
```

`retrieval_ms` 包含首次 retriever 初始化；预热后 `setup_ms` 会接近 0。  
SSE 的 `done` 事件也带 `timings`。

## 检索缓存

相同 query + 过滤条件 + active index identity：

```text
first:  retrieval_ms 1186ms
second: retrieval_ms 0.53ms, retrieval_cache_hit=1
```

缓存 key 包含 `source_ref/build_id`、embedding/rerank 模型、limit、doc_type、版本和 include 开关，索引切换后自然失效。

## 结论

- `async` 让 API 路由不再阻塞 event loop，慢 LLM 请求不会占满线程池。
- Chat 5 并发 p95 从 6.34s 降到 5.42s，约 15%；吞吐从 0.79 到 0.92 rps。
- 检索没有明显提升，因为 query embedding / Qdrant 仍是外部依赖，瓶颈不在 event loop。
- 提高 retrieval 并发到 32 反而使 100 并发 p95 从 3.70s 升到 4.54s，因此保留 16。
- 检索结果缓存命中后 `retrieval_ms` 从 1186ms 降到 0.53ms。
- 10 并发 chat 没有错误，但 p95 升高，说明上游模型延迟和并发限制开始成为主瓶颈。
- 不能说“支持 100 并发 chat”；当前只验证到 10 并发 chat、100 并发检索。

## 为什么没有跑 100 并发 chat

当前每日 LLM 额度只剩 18 次请求，`LLM_MAX_CONCURRENCY=8`。

如果直接发 100 个 chat：

- 前 8 个进入模型
- 后续请求在 semaphore 排队
- 额度耗尽后大量返回 429
- 结果不是“100 并发能力”，而是“配额 + 队列行为”

因此没有把它包装成 100 并发 chat 数据。  
要测 100 chat，需要独立测试环境、提高每日额度，并接受上游 provider 的 RPM/TPM 限制。

## 复现

```bash
# 检索
python scripts/load_chat.py \
  --path /api/v1/search \
  --concurrency 20 \
  --requests 20 \
  --unique-ip

# Chat
python scripts/load_chat.py \
  --path /api/v1/chat \
  --concurrency 5 \
  --requests 5 \
  --unique-ip

# SSE
python scripts/load_chat.py \
  --path /api/v1/chat/stream \
  --concurrency 5 \
  --requests 5 \
  --stream \
  --unique-ip
```

## 限制

- 单机 Docker，不是多副本集群
- 模型供应商的 RPM/TPM 与网络波动会影响结果
- 没有做 30 分钟稳定性测试
- 没有做故障注入/断网测试
- 报告中的 p95 不能外推为生产 SLA

## 下一步

1. 把 `X-Real-IP` 限制为可信代理
2. 为 chat/stream 增加 20 并发稳定性测试
3. 加请求队列满时的 429/503 指标
4. 把上游 429 和超时单独统计
