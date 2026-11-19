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
| BC-05 | 已修复（重试） | g09 答案内容正确，但只引用 `[1] simple-oauth2`，没引用 `oauth2-jwt` | 引用准确率 |
| BC-06 | 已修复（策略） | rerank 曾把版本题 v01 从 rank 1 挤到 rank 6 | Recall@5 |
| BC-07 | 已放弃（策略） | hybrid 不加 dense-first guard 时 MRR 低于 dense | 检索质量 |
| BC-08 | 已修复（重试） | g04 BackgroundTasks 答案正确但没有引用 | 引用完整率 |
| BC-09 | 已修复（重试） | g08 HTTPS/TLS 答案正确但没有引用 | 引用完整率 |
| BC-10 | 部分修复 | g15 0.100.0 版本题路由到 release-notes，但索引只保留最近 20 版 | 版本检索 |
| BC-11 | 已修复（重试） | g16 regex→pattern 答案正确但没有引用 | 引用完整率 |
| BC-12 | TODO | g03 路径/查询参数答案正确但引用没命中期望 URL | 引用选择 |
| BC-13 | TODO | g22 装饰器/参数区别答案正确但引用指向相邻文档 | 引用选择 |

当前整体基线：

- 检索：30 条，Recall@5 = 1.0，Recall@1 = 0.933，MRR = 0.967
- 生成：24 条，21 pass，citation_validity = 1.0，citation_hit = 0.9，refusal_accuracy = 1.0

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

已修复：prompt 强化 + 无引用重试后，g09 在 24 条评估中通过。

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

## BC-08：BackgroundTasks 答案正确但无引用

```text
query: BackgroundTasks 怎么用？
source_recall: true
keyword_coverage: 1.0
citation_hit: false
```

### 根因

- 模型输出了完整步骤和代码，但关键结论没有 `[n]`
- prompt 已要求引用，但模型仍可能漏掉
- 当前 `check_citations` 能发现“无引用”，但没有强制重试

### 干预候选

1. 生成后检查 `citation_check.citations == 0`，带纠正 prompt 重试一次
2. 对无引用的答案在 UI 明确标记“未通过引用校验”
3. 生成结构化答案：结论 + citations 分开输出

### 状态

已修复：`generate_answer` / `/chat` 在无有效引用时带纠正 prompt 重试一次；g04 已通过。

---

## BC-09：HTTPS/TLS 答案正确但无引用

```text
query: FastAPI 怎么配置 HTTPS 和 TLS 证书？
source_recall: true
keyword_coverage: 1.0
citation_hit: false
```

### 根因

- 答案解释了 TLS termination proxy，内容正确
- 但没有给出 `[n]`
- 说明引用生成仍不稳定

### 干预候选

已修复：同 BC-08 的引用重试；g08 已通过。

---

## BC-10：0.100.0 版本题没召回 release-notes

```text
query: FastAPI 0.100.0 对 on_startup 和 on_shutdown 做了什么变更？
expected: release-notes/
source_recall: false
```

### 根因

- 版本问题需要优先路由到 release notes
- 当前只做普通 hybrid 检索，没有版本意图识别
- 模型转而回答 0.162.0 的 deprecated 信息，答案偏离问题

### 干预候选

1. 识别版本号 query，强制 `doc_type=release_notes`
2. 召回 release notes 后再做普通文档补充
3. 版本类问题加入路由/多 query

### 状态

部分修复：`route_doc_type` 已把版本号问题强制路由到 `release_notes`；但当前索引只保留最近 20 个版本，0.100.0 不在其中，模型正确拒答而不是编造。完整修复需要历史 collection。

---

## BC-11：regex→pattern 答案正确但无引用

```text
query: Pydantic v2 迁移后，FastAPI 的 regex 参数应该改成什么？
source_recall: true
keyword_coverage: 1.0
citation_hit: false
```

### 根因

- 答案内容正确：`regex` -> `pattern`
- 没有引用
- 说明“无引用”是当前生成的主要失败模式，而不是知识错误

### 干预候选

已修复：同 BC-08 的引用重试；g16 已通过。

---

## BC-12：g03 引用没命中期望 URL

```text
query: 怎么定义带路径参数和查询参数的 GET 接口？
source_recall: true
citation_hit: false
answer: 内容正确
```

### 根因

- 检索到了 tutorial/path-params 和 query-params
- 生成引用了相邻/泛化片段，citation_hit 只看期望 URL
- 不是知识错误，是引用选择问题

### 干预候选

同 BC-05：更严格的 citation 选择 prompt，或按句子做来源匹配。

---

## BC-13：g22 装饰器 vs 参数引用偏移

```text
query: Depends 在路径操作装饰器里和函数参数里有什么区别？
source_recall: true
citation_hit: false
answer: 内容正确
```

### 根因

- 期望 `dependencies-in-path-operation-decorators/`
- 回答引用了 `tutorial/dependencies/` 的相邻片段
- 语义上相邻，但引用不是最精确页面

### 干预候选

- cross-encoder 重排后，引用选择仍有偏移
- 需要句子级 citation attribution，而不是只做文档级引用

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
