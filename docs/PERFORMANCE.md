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

SSE：

```text
5 并发 / 5 请求
总延迟 p95: 4840.21ms
首 token p50: 730.37ms
首 token p95: 1097.31ms
错误: 0
```

## 结论

- `async` 让 API 路由不再阻塞 event loop，慢 LLM 请求不会占满线程池。
- Chat 5 并发 p95 从 6.34s 降到 5.42s，约 15%；吞吐从 0.79 到 0.92 rps。
- 检索没有明显提升，因为 query embedding / Qdrant 仍是外部依赖，瓶颈不在 event loop。
- 10 并发 chat 没有错误，但 p95 升高，说明上游模型延迟和并发限制开始成为主瓶颈。
- 不能说“支持 100 并发”；当前只验证到 10 并发 chat、20 并发检索。

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
