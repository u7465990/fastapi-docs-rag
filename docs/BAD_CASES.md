# Bad Cases（初始版）

> 当前是最小闭环记录，不是最终的 10–20 条完整交付物。
> 每条都是实测，不编造现象；未解决的 case 明确标为 TODO。
> 评估口径见 [EVALUATION.md](EVALUATION.md)。

---

## 概览

| id | 状态 | 现象 | 影响 |
|---|---|---|---|
| BC-01 | 已修复 | `TestClient` 问题被 `reference/testclient` 抢到 top-1 | Recall@1 |
| BC-02 | 已改善，未解决 | OAuth2 + JWT 被 `simple-oauth2` 抢到 rank 3（hybrid rerank 后 rank 2） | MRR |
| BC-03 | 已改善，未解决 | OAuth2PasswordBearer + JWT 多跳题 rank 4（hybrid rerank 后 rank 2） | MRR / 多跳 |
| BC-04 | 已修复（标注） | 204 问题只标注 tutorial，advanced 命中被算错 | 评估口径 |
| BC-05 | TODO | g09 答案内容正确，但只引用 `[1] simple-oauth2`，没引用 `oauth2-jwt` | 引用准确率 |
| BC-06 | 已修复（策略） | rerank 曾把版本题 v01 从 rank 1 挤到 rank 6 | Recall@5 |
| BC-07 | 已放弃（策略） | hybrid 不加 dense-first guard 时 MRR 低于 dense | 检索质量 |

当前整体基线：

- 检索：30 条，Recall@5 = 1.0，Recall@1 = 0.933，MRR = 0.967
- 生成：14 条，13 pass，citation_validity = 0.917，refusal_accuracy = 1.0

这些不是“完全没召回”，而是**排序、引用选择或标注问题**。

---

## BC-01：reference 页面抢走 tutorial 的 top-1

### 现象

```text
query: 怎么用 TestClient 给 FastAPI 写测试？
expected: https://fastapi.tiangolo.com/tutorial/testing/
observed top-1: https://fastapi.tiangolo.com/reference/testclient/
rank: 2
```

### 根因

- 索引里包含 `doc_type=reference` 的页面
- 检索过滤器默认只排除了 `meta`，没有排除 `reference`
- `reference/testclient` 与“TestClient 测试”语义高度接近，dense 排序把它排到了第一

### 干预

按设计文档“reference 默认不参与正文检索”，在 `build_filter` 中默认排除 `doc_type=reference`；需要时可通过 `include_reference=true` 显式打开。

### 前后指标

| 指标 | 修复前 | 修复后 |
|---|---:|---:|
| Recall@1 | 0.867 | 0.900 |
| MRR | 0.919 | 0.936 |

### 结果

`TestClient` 从 rank 2 变为 rank 1；reference 页面不再污染普通问答。

### 备注

这是检索过滤策略问题，不是 embedding 模型问题。

---

## BC-02：OAuth2 + JWT 被相邻教程抢到 rank 3

### 现象

```text
query: OAuth2 密码流加 JWT 怎么在 FastAPI 里实现？
expected: https://fastapi.tiangolo.com/tutorial/security/oauth2-jwt/
observed top-1: https://fastapi.tiangolo.com/tutorial/security/simple-oauth2/
rank: 3
```

### 当前状态

hybrid + lexical rerank 后：

```text
rank: 2
top-1: simple-oauth2（仍未解决）
MRR: 0.953 -> 0.967
```

### 根因

- `simple-oauth2` 和 `oauth2-jwt` 是相邻教程，术语高度重叠
- dense 只看语义相似度，无法区分“简单 OAuth2”和“OAuth2 + JWT”
- 没有 rerank，也没有对标题/路径做加权

### 候选干预

1. rerank（cross-encoder）
2. 标题/section/path 加权
3. 多查询：把“OAuth2”“JWT”“密码流”拆成多个 query
4. hybrid + 英文 query 扩展

### 状态

TODO。当前 MRR 0.953 已包含这个损失；后续做 rerank 后必须复测。

---

## BC-03：OAuth2PasswordBearer + JWT 多跳题 rank 4

### 现象

```text
query: OAuth2PasswordBearer 怎么配合 JWT 做登录认证？
expected: tutorial/security/first-steps/ 或 tutorial/security/oauth2-jwt/
observed top-1: tutorial/security/simple-oauth2/
rank: 4
```

### 当前状态

hybrid + lexical rerank 后：

```text
rank: 2
top-1: simple-oauth2（仍未解决）
```

### 根因

- 需要同时理解 `OAuth2PasswordBearer` 和 JWT 两篇文档
- 单 query dense 检索只能返回一个“最像”的片段
- `simple-oauth2` 的 token 返回片段恰好包含 OAuth2 关键词

### 候选干预

1. 多跳 query decomposition
2. rerank 后保留多个来源
3. 对 `oauth2-jwt` 的标题/路径加权

### 状态

TODO。属于 multi-hop 检索问题，不是简单调 prompt 能解决。

---

## BC-04：204 响应题的 golden 标注过窄

### 现象

```text
query: 怎么让接口返回 204 状态码并且不返回响应体？
expected(initial): tutorial/response-status-code/
observed top-1: advanced/response-change-status-code/
rank: 2
```

### 根因

这个问题的正确答案同时存在于：

- `tutorial/response-status-code/`
- `advanced/response-change-status-code/`

最初只标注 tutorial，导致 advanced 命中被算成错误。  
这是**评估标注问题**，不是检索错误。

### 干预

把 advanced URL 加入 `expected_urls`。

### 前后指标

| 指标 | 标注修正前 | 标注修正后 |
|---|---:|---:|
| Recall@1 | 0.900 | 0.933 |
| MRR | 0.936 | 0.953 |

### 结果

问题变成 rank 1；没有改检索代码。

### 教训

- 评估指标变化必须区分“检索变好”和“标注变准”；
- 一个 query 有多个正确 URL 时，golden set 要允许列表，不能只写一个。

---

## BC-05：答案正确但引用选错

### 现象

```text
query: OAuth2 密码流加 JWT 怎么在 FastAPI 里实现？
expected: tutorial/security/oauth2-jwt/
sources: 8 条，包含 oauth2-jwt/
answer: 内容包含 JWT / OAuth2 / OAuth2PasswordBearer
answer citation: [1]
source[1]: tutorial/security/simple-oauth2/
citation_hit: false
```

### 根因

- 模型只引用了第一条 source，没有引用真正支撑 JWT 的 `oauth2-jwt` 片段
- 上下文里同时有 `simple-oauth2` 和 `oauth2-jwt`，两者主题相邻
- 当前 prompt 只要求“标注来源编号”，没有强制“引用最直接支撑结论的片段”

### 候选干预

1. 生成后做 citation check：结论句和 source 的 token 重叠
2. 把 sources 按与答案的相似度重排后再映射编号
3. prompt 增加“引用最直接的 source，不要只引第一条”
4. LLM-as-Judge 检查引用是否支撑结论

### 状态

TODO。已加入生成评估 `g09`。

---

## BC-06：rerank 把版本题挤出前 5

### 现象

```text
query: FastAPI 0.162.0 对布尔表单字段做了什么变更？
expected: release-notes/
dense: rank 1
hybrid + lexical rerank（无 guard）: rank 6
Recall@5: 1.000 -> 0.967
```

### 根因

- lexical rerank 给 `request-forms` 的标题/URL token 加了分
- `release-notes` 标题和 query 的 token 重叠弱
- 版本题被普通表单文档挤掉

### 干预

- 固定 dense top-1，只重排尾部候选（dense-first guard）
- 不把 lexical rerank 当作通用语义精排

### 结果

```text
hybrid + rerank（有 guard）:
Recall@1 0.933
Recall@5 1.000
MRR 0.967
```

### 教训

启发式 rerank 会伤害“高置信 dense top-1”；需要 guard 或真正的 cross-encoder。

---

## BC-07：hybrid 无 guard 时 MRR 低于 dense

### 现象

```text
dense:
Recall@1 0.933
MRR 0.953

hybrid（无 rerank / 无 dense-first guard）:
Recall@1 0.867
MRR 0.933
```

### 根因

- BM25 和 dense 的排名冲突
- RRF 让 BM25 的 top-1 与 dense 的 top-1 同权
- 中文 query 的英文词典扩展带来噪声

### 干预

- 加 dense-first guard
- 纯中文 query 不注入 glossary
- BM25 权重低于 dense（默认 2.0 : 0.5）

### 结果

hybrid + guard 的 MRR 回到 0.967。

### 教训

hybrid 不是“加了就更好”；必须用同一套 golden set 做前后对比，并保留回退策略。

---

## 后续模板

新增 bad case 时按这个格式记录：

```markdown
## BC-XX：一句话现象

### 现象
query / expected / observed / rank

### 根因
是解析、切分、召回、排序、生成、版本还是标注问题？

### 干预
具体改了什么，或者准备改什么。

### 前后指标
| 指标 | 前 | 后 |
|---|---:|---:|

### 结果与教训
```

目标：10–20 条实测 case，覆盖：

- 无答案 / 应拒答
- 代码符号（`HTTPException`、`Annotated`）
- 版本差异 / 废弃 API
- 中文口语化 query
- 多跳
- 引用错误
- 切分错误（代码块被切断）
