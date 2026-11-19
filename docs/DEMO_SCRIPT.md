# 3 分钟 Demo 脚本

目标：让面试官在 3 分钟内看到“可打开、可追溯、可解释、可评估”，不要只念架构。

## 录制前

```powershell
docker compose up -d
# 可选：开公网
docker run -d --name cf-quick cloudflare/cloudflared:latest tunnel --url http://host.docker.internal:8080
```

打开：

```text
http://localhost:8080
```

准备问题：

1. `Depends 和 Annotated 怎么一起用？`
2. `那它和普通参数有什么区别？`（演示多轮 rewrite）
3. `FastAPI 怎么做视频转码和推流？`（演示拒答）
4. 打开“调试信息”

## 0:00–0:25 开场

讲：

> 这是一个面向 FastAPI 官方文档的 RAG，支持中文提问检索英文文档，回答必须带引用，检索不到会拒答。  
> 我想重点讲检索和评测，而不是只做一个聊天框。

## 0:25–1:00 正常问答

输入：

```text
Depends 和 Annotated 怎么一起用？
```

展示：

- 流式输出
- 来源 [1]–[6]
- 点击来源
- 最后引用

讲：

> 父子切分保留上下文，dense + BM25 做混合召回，RRF 融合，答案里的编号由后端映射为 URL。

## 1:00–1:30 多轮改写 + 调试面板

追问：

```text
那它和普通参数有什么区别？
```

打开“调试信息”，展示：

- 检索 query
- `embedding_ms`
- `dense_ms` / `bm25_ms`
- `rerank_ms`
- `retrieval_ms`
- `llm_ms`
- 每个来源的 dense / rerank 分数

讲：

> 短追问会用上一轮用户问题做确定性 query rewrite，不额外调用 LLM。  
> `timings` 用来定位延迟，而不是只看总时间。

## 1:30–2:00 拒答

问：

```text
FastAPI 怎么做视频转码和推流？
```

展示拒绝回答，而不是硬编。

讲：

> 检索 Top-1 低于阈值就直接拒答；答案生成后还会做 citation 校验，没有有效引用会重试一次。

## 2:00–2:40 评测与失败案例

打开：

```text
docs/EVALUATION.md
docs/BAD_CASES.md
docs/PERFORMANCE.md
```

讲：

- 30 条检索：MRR 0.953 → 0.967
- 小 cross-encoder 全量 rerank MRR 0.967；高置信度跳过后 0.983
- 24 条生成：21 pass
- citation validity 1.0
- 100 并发检索：p95 3.70s，0 error
- 失败案例：g03/g22 引用选择、g15 历史版本缺失

不要隐瞒失败。

## 2:40–3:00 收尾

讲：

> 这个项目最想证明的是：RAG 不是把文档丢给 LLM，而是检索、重排、引用、拒答、评估和延迟都要可测量。  
> 当前明确没做的是历史多 collection 和 RAGAS，这是下一步。

## 录制建议

- 1280×720 或 1920×1080
- 浏览器缩放 100%
- 麦克风先试音
- 不要录 API Key、`.env`、终端历史
- 视频不要加速；3 分钟足够

## 旁白稿

完整旁白见本文件各时间段。  
建议先录一遍不剪辑，第二遍再补特写。
