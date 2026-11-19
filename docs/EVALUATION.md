# 检索评估基线（2026-11-15，2026-11-17 更新 hybrid）

> 范围：dense baseline + dense/BM25 hybrid + lexical rerank。
> 不包含：LLM 生成质量、引用准确率、拒答质量、cross-encoder rerank。
> 数据：`data/eval/golden_set.jsonl`，30 条中文问题。
> 索引：`fastapi_docs_current`，155 篇文档 / 1363 个 child 向量。
> Embedding：DashScope `text-embedding-v4`，1024 维。
> 运行方式：宿主机 `scripts/evaluate.py` 直连本地 Qdrant 容器。

---

## 1. Golden set

文件：`data/eval/golden_set.jsonl`

每条字段：

```json
{
  "id": "t01",
  "question": "FastAPI 的第一个应用怎么创建并运行？",
  "expected_urls": ["https://fastapi.tiangolo.com/tutorial/first-steps/"],
  "category": "tutorial",
  "language": "zh"
}
```

覆盖：

| category | cases |
|---|---:|
| tutorial | 15 |
| advanced | 4 |
| multi-hop | 3 |
| deployment | 3 |
| how-to | 2 |
| security | 2 |
| version | 1 |

说明：

- 问题全部中文，文档全部英文。
- `expected_urls` 允许一个或多个正确 URL；多跳题给多个目标。
- 当前索引包含 reference 页面，但检索默认过滤 `doc_type=reference`，与设计一致；需要时可通过 `include_reference=true` 打开。

---

## 2. 最终基线

运行：

```powershell
uv run --no-sync python scripts/evaluate.py
```

结果：

```text
cases: 30
Recall@1: 0.933
Recall@5: 1.000
Recall@10: 1.000
MRR: 0.953
latency_ms: p50=165 p95=233 max=252
misses: 0
```

分类 Recall@5：

```text
advanced: 1.000 (4)
deployment: 1.000 (3)
how-to: 1.000 (2)
multi-hop: 1.000 (3)
security: 1.000 (2)
tutorial: 1.000 (15)
version: 1.000 (1)
```

延迟口径：

- 包含 query embedding + Qdrant 检索
- 不包含 LLM
- 受网络波动影响；不同时间跑会有差异

---

## 3. 迭代记录

### 3.1 初始 26 条

```text
Recall@1: 0.923
Recall@5: 1.000
MRR: 0.955
```

### 3.2 加入 4 条更难的多跳题后（30 条）

```text
Recall@1: 0.867
Recall@5: 1.000
MRR: 0.919
```

暴露了两个现象：

- `TestClient` 问题被 `reference/testclient` 抢到第 1
- OAuth2/JWT 相关问题被相邻的 `simple-oauth2` 抢到第 3/4

### 3.3 默认过滤 reference 页面

设计文档本来就规定 `reference/` 默认不参与正文检索；此前检索过滤器只排除了 `meta`，没有排除 `reference`。

修改后：

```text
Recall@1: 0.900
Recall@5: 1.000
MRR: 0.936
```

`TestClient` 问题从 rank 2 变为 rank 1。

### 3.4 修正一条 golden 标注

`m03`（返回 204 不返回 body）的答案同时存在于：

- `tutorial/response-status-code/`
- `advanced/response-change-status-code/`

最初只标注 tutorial，导致 advanced 命中被算错；把两个 URL 都加入 expected 后：

```text
Recall@1: 0.933
Recall@5: 1.000
MRR: 0.953
```

这不是检索提升，而是**评估标注修正**。

---

## 4. 还没解决的 near miss

| id | query | 期望 | 实际 top-1 | rank |
|---|---|---|---|---:|
| s01 | OAuth2 密码流加 JWT 怎么实现？ | `tutorial/security/oauth2-jwt/` | `tutorial/security/simple-oauth2/` | 3 |
| m02 | OAuth2PasswordBearer 怎么配合 JWT？ | `tutorial/security/first-steps/` 或 `oauth2-jwt/` | `tutorial/security/simple-oauth2/` | 4 |

共性：

- 两篇文档主题相邻、术语高度重叠
- dense 只按语义相似度排序，没有 rerank
- 单 query 很难区分“简单 OAuth2”和“OAuth2 + JWT”

候选干预：

1. rerank（最直接）
2. hybrid + 英文 query 扩展
3. 标题/section 加权
4. 多跳题拆成两个 query

在没做这些之前，不把 MRR 当成最终质量。

---

## 4.5 2026-11-17：hybrid + lexical rerank

### 实现

- `BM25Index`：从 Qdrant payload 读取 1363 个 child，在内存建 BM25。
- `HybridRetriever`：dense top 20 + BM25 top 20，用加权 RRF 融合。
- `LexicalReranker`：标题、section、URL token 重叠 + 短语命中；不是 cross-encoder。
- 默认：`RETRIEVAL_MODE=hybrid`、`RERANK_ENABLED=true`。
- 中文纯 query 不注入 glossary，BM25 返回空，自动退化为 dense；混合/英文 API 词才走 BM25。
- 为避免标题词把正确的 dense top-1 挤掉，rerank 固定 dense top-1，只重排尾部候选（dense-first guard）。

### 结果

```text
mode=hybrid, rerank=true
cases: 30
Recall@1: 0.933
Recall@5: 1.000
Recall@10: 1.000
MRR: 0.967
latency_ms: p50=184 p95=332 max=432
```

对比：

| 版本 | Recall@1 | Recall@5 | MRR |
|---|---:|---:|---:|
| dense baseline | 0.933 | 1.000 | 0.953 |
| hybrid + rerank | 0.933 | 1.000 | **0.967** |

- s01 / m02 从 rank 3/4 提升到 rank 2；top-1 仍是 `simple-oauth2`。
- Recall 没有下降，MRR 有小幅提升。
- 这是 **lexical rerank**，不是 cross-encoder；不能宣称“语义精排”。

### 运行

```powershell
uv run --no-sync python scripts/evaluate.py --mode dense
uv run --no-sync python scripts/evaluate.py --mode hybrid
uv run --no-sync python scripts/evaluate.py --mode hybrid --rerank
```

### 局限

- BM25 依赖英文 token；纯中文 query 实际只走 dense。
- 词典扩展只覆盖少量 FastAPI 术语。
- dense-first guard 是为了保护当前 30 条上的 top-1，不代表通用最优。
- 没有做参数搜索，也没有交叉验证；不能把 0.967 当作泛化结论。

---

## 4.6 2026-11-17：生成 / 引用 / 拒答评估

### 数据集

`data/eval/generation_set.jsonl`，24 条：

- 20 条可回答
- 4 条无答案（2 条强制无上下文，2 条真实检索后应拒答/否定）
- 加入 version、code、hard-negative、真实 OOD 题

字段：

```json
{
  "question": "...",
  "expected_urls": ["..."],
  "expected_keywords": ["lifespan", "on_event"],
  "should_refuse": false,
  "force_no_context": false
}
```

### 运行

```powershell
uv run --no-sync python scripts/evaluate_generation.py
```

### 结果

```text
cases: 24
answerable: 20
no_answer: 4
passed: 20
pass_rate: 0.833
source_recall: 0.950
citation_validity: 0.900
citation_hit: 0.800
keyword_coverage: 1.000
refusal_accuracy: 1.000
latency p50: 4791ms
latency p95: 5383ms
latency max: 6352ms
```

### 失败分布

- `g04` BackgroundTasks：内容正确，但没有引用
- `g08` HTTPS/TLS：内容正确，但没有引用
- `g15` 0.100.0 版本题：期望 release-notes，检索没召回
- `g16` regex→pattern：内容正确，但没有引用
- `g09` OAuth2/JWT 在 prompt 强化后已通过

所以主要问题从“答案错”变成了：

1. 模型有时不写 `[n]`
2. 版本题检索仍偏弱

无答案题 4/4 都正确拒答/否定，refusal_accuracy = 1.0。

### 口径说明

- `source_recall`：期望 URL 是否出现在检索 sources。
- `citation_validity`：答案里的 `[n]` 是否都在 sources 范围内。
- `citation_hit`：被引用的 source 是否包含期望 URL。
- `keyword_coverage`：期望关键词是否都出现在答案里。
- `refusal_accuracy`：无答案题是否明确拒答。
- 无答案题用 `force_no_context=true`，测的是拒答路径，不是“检索不到”的端到端行为。

### 局限

- 关键词是弱监督，不等于语义正确。
- 没有 LLM-as-Judge / RAGAS faithfulness。
- 没有人工逐条评分。
- 只有 14 条，偏小。
- 延迟包含完整 LLM 生成，p50 约 4.8 秒。

---

## 5. 结论与局限

### 结论

- 30 条中文问题、英文文档，dense baseline 的 Recall@5 = 1.0。
- hybrid + lexical rerank 把 MRR 从 0.953 提到 0.967，Recall 不变。
- 14 条生成评估 13 条通过；唯一失败是 g09 引用选错，不是检索没召回。
- reference 默认过滤和 golden 标注修正是两处真实改进。
- p50/p95 检索延迟在本地 8 并发测试中仍低于 500ms（见 `WORKLOG_2026-11-14.md`）。

### 局限

- 30 条太少，且问题是从文档主题手工构造的，偏“可答”
- 没有无答案题、没有硬负例、没有代码符号题、没有真正的版本差异题
- Recall@5 = 1.0 不代表生产质量，只代表这 30 条都能在前 5 找到
- 没有评估 LLM 答案、引用准确率、拒答
- 没有评估 reference 之外的 meta/release_notes 干扰

### 下一步

1. 扩到 60–100 条，加入无答案/硬负例/代码符号/版本差异题
2. 评估 `/chat`：引用准确率、拒答正确率、faithfulness（RAGAS 或人工）
3. 修 g09 的引用选择，并扩到 30+ 条生成题
4. 把 lexical rerank 升级为 cross-encoder，并复测 s01/m02
5. 把评估报告和 bad case 纳入 CI smoke

---

## 6. 复现

```powershell
# 先确认 Qdrant 有索引、.env 有 embedding key
uv run --no-sync python scripts/ingest.py --dry-run
uv run --no-sync python scripts/evaluate.py

# 报告写入 data/reports/retrieval_eval_<ts>.json（被 gitignore）
```

当前报告示例：

```text
data/reports/retrieval_eval_20261115T053057Z.json
```
