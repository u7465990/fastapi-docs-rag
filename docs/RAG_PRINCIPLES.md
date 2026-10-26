# 原理文档：版本感知的 FastAPI 技术文档 RAG

> 版本：v0.1（2026-10-18）  
> 定位：解释本项目每个设计**为什么成立**，以及底层机制与失败模式。  
> 与 `PROJECT_DESIGN.md` 的分工：设计文档回答“做什么、用什么、怎么验收”；本文档回答“原理是什么、为什么这样选、哪里会坏”。

---

## 0. 先建立全局图景

RAG（Retrieval-Augmented Generation）不是“搜索 + 把结果拼进 prompt”这么简单。它是一条**信息逐级压缩、逐级损耗**的链路：

```text
原始文档
  → 解析清洗        损失：结构噪声、格式信息
  → 切分 chunk      损失：跨 chunk 的上下文、指代关系
  → 向量 / 稀疏表示  损失：细节、精确符号、否定/数值语义
  → top-k 召回      损失：未进入候选的相关内容
  → 融合 + rerank   损失：排序误差
  → 上下文拼装      损失：token 预算、去重、顺序
  → LLM 生成        损失：忠实度、引用错误
```

两个核心结论：

1. **检索决定上限，生成只决定答案离上限有多近。** 上下文里没有正确信息，再强的模型也只能编或拒答。
2. **优化必须定位到具体损耗环节。** 同一个“答错了”，可能是切分切断了代码、BM25 没召回精确符号、rerank 排错、还是模型没遵守引用约束。没有分层评估，就只能盲目调 prompt。

本项目的全部设计，本质上是在这条链路的每个环节做一个**有依据的折中**。

---

## 1. RAG 的第一性原理

### 1.1 LLM 的知识边界

LLM 的知识来自参数（parametric memory），它有三个硬约束：

| 约束 | 含义 | 后果 |
|---|---|---|
| 知识截止 | 训练数据有截止时间 | 回答不了新版本 API |
| 不可审计 | 参数里的事实没有出处 | 无法给出可点击引用 |
| 概率生成 | 输出是“最可能的续写” | 会流畅地编造（幻觉） |

RAG 把知识从“参数”中解耦到“外部索引”，让模型退化为**阅读器 + 推理器**：它只需要理解当前上下文，并在约束下组织答案。由此获得三项能力：

- **新鲜**：索引更新即可回答新内容，无需重训模型。
- **私有**：可接入模型没见过的内部文档。
- **可归因**：每个结论能映射到 chunk → section → URL。

### 1.2 为什么不用长上下文或微调替代

| 方案 | 为什么不够 |
|---|---|
| 把整份文档塞进长上下文 | 成本随 token 线性增长；注意力在长上下文中间会衰减（lost in the middle）；文档更新后仍需重新加载；难以给出稳定引用 |
| 微调模型 | 微调擅长“风格 / 格式 / 任务模式”，不适合注入可验证、频变的事实；无法保证引用；文档更新就要重训；仍会幻觉 |
| 直接联网搜索 | 结果不可控、不可复现，技术文档场景需要稳定且版本明确的来源 |

长上下文可以作为 RAG 的补充（例如把命中 section 的相邻内容一并给模型），但不能替代“先检索、再生成”的结构。原因是：**上下文窗口变大，不等于相关信息密度变高。**

### 1.3 RAG 的信息瓶颈

RAG 的召回率上限由“索引表示 + 召回策略”决定。设某个问题真正需要的证据集合为 $E$，最终进入上下文的集合为 $C$：

- 若 $E \cap C = \varnothing$，生成阶段必败（只能拒答或幻觉）。
- 若 $E \cap C \neq \varnothing$ 但大量噪声也在 $C$ 中，模型可能被误导、引用错段落，或超出 token 预算。
- 若 $E \subseteq C$ 但顺序/格式差，模型可能漏读关键句。

所以本项目的原则是：**先保证证据进得来（Recall），再保证噪声排得掉（Precision），最后才是让模型好好写（Faithfulness）。** 这也是迭代顺序“dense baseline → hybrid + rerank → Agentic → 评估”的原因。

---

## 2. 离线索引：把文档变成可检索的表示

### 2.1 采集：版本感知必须从源头开始

版本感知不是索引阶段临时加字段，而是采集阶段就要拿到“这份内容对应哪个源版本”。三种数据入口的能力差异：

| 数据源 | 能拿到什么 | 版本感知能力 |
|---|---|---|
| 官方文档 Git 仓库（Markdown/MDX） | commit SHA、文件级 diff、重命名/删除 | **最强**：可以精确知道哪些页面变了 |
| sitemap.xml | URL、`lastmod` | 中等：`lastmod` 经常不准或全站一起变 |
| 渲染后的 HTML | URL、内容、ETag/Last-Modified | 最弱：只能靠内容哈希和页面文本里的版本标记 |

本项目只有一种入口：**FastAPI 官方 Git 仓库**（`fastapi/fastapi`）。英文文档在 `docs/en/docs/**/*.md`，代码示例在 `docs_src/**/*.py`，后者通过 `{* ../../docs_src/... *}` 被文档引用。FastAPI 文档只维护“最新版”页面，版本号来自 `release-notes.md` 顶部版本或 git tag，因此 `source_ref`（commit SHA）和 `docs_version`（0.162.0）必须同时保存。

没有 `source_ref`，“文档变更 → 可检索延迟”就无法准确度量，只能退化成“我们多久抓一次”；没有 `docs_version`，回答就无法标注适用版本。

### 2.2 解析与清洗：保住结构就是保住语义

技术文档的关键信息往往不在正文段落里，而在结构里：

- **标题层级**决定 section 边界和上下文归属；
- **代码块**包含 API 名称、参数、可运行示例；
- **提示框 / 警告块**（note、warning、deprecated）包含版本与迁移信息；
- **表格**包含参数对比、版本支持矩阵。

如果清洗时只保留“可见文本”，这些结构全部丢失，后续切分和生成都会退化。因此解析目标是 **保留结构的 Markdown**，而不是纯文本。

FastAPI 文档还有三种特有结构，必须在清洗阶段处理：

- `{* ../../docs_src/... *}` 代码 include 要先展开，并去掉 `hl[...]` 行高亮参数；
- `/// tip`、`/// warning` 等提示块要保留类型与正文，其中常含废弃/迁移信息；
- 标题里的 `{ #anchor }` 要分离出来，用于生成可点击的 URL 锚点。

否则代码示例、废弃警告和精确引用都会丢失。`reference/**` 页面是 mkdocstrings 指令（`::: fastapi.FastAPI`），没有实际 API 文本，v1 默认排除。

同时要处理噪声：导航栏、页脚、cookie 提示、重复的“Edit this page”等。这些内容会污染 embedding，让不相关页面在向量空间里彼此靠近。

### 2.3 切分：在“可检索”与“可理解”之间折中

#### 为什么必须切

- embedding 模型有输入长度限制，整页编码会把多个主题压进一个向量，语义被稀释；
- 检索粒度太粗，命中后只能整页塞给 LLM，浪费 token；
- 引用需要定位到 section，整页无法精确归因。

#### 为什么不能切太碎

- chunk 太短会丢失指代对象（“它”“上面的配置”），embedding 也无法判断主题；
- 代码块被切断后不可运行，引用也失去意义；
- 一个完整概念被拆到多个 chunk，单条召回都不足以回答。

#### 父子切分（Small-to-Big）的原理

```text
Parent（约 1200–1600 tokens，完整小节）
├─ Child 1（约 300–500 tokens）← 用于 embedding / 召回
├─ Child 2
└─ Child 3
```

- **Child 负责“被找到”**：粒度小，语义集中，向量区分度高。
- **Parent 负责“被理解”**：命中 child 后返回 parent，给 LLM 完整的论证、代码和上下文。
- 二者是同一份内容的两种视图，不是两次独立切分。

这解决了 RAG 的经典矛盾：**检索希望粒度小，生成希望上下文全。**

#### overlap 的作用与代价

相邻 child 保留 50–80 tokens 重叠，防止关键句正好落在边界。代价是索引体积和 embedding 成本增加。overlap 不是越大越好：过大时相邻 chunk 高度相似，会挤占 top-k 名额。

#### 代码块保护

代码块是不可分割的语义单元。切分器需要：

1. 先展开 `{* ../../docs_src/... *}` include，再识别 fenced code block 的起止；
2. 短代码块整体保留；
3. 超长代码块按空行 / 函数 / 类边界切，但每段都保留语言标记和所属标题；
4. 在 metadata 中写 `content_type="code"`、`code_language`、`source_path`。

技术文档 RAG 里，很多“答非所问”不是模型问题，而是代码示例被切成了两半。

#### 标题上下文注入

embedding 前给 child 拼接：

```text
[FastAPI > Tutorial > Dependencies > Classes as Dependencies]
<chunk 正文>
```

原理：同样的正文（例如“配置如下……”）在不同 section 下含义完全不同。标题路径相当于给向量一个**免费的上下文先验**，成本几乎为零，通常是最划算的优化之一。

注意：存入 payload 的正文应保留原文；标题前缀可以只用于 embedding，或同时保留在检索文本中。具体策略应在评估中对比。

#### Contextual Retrieval（可选实验）

用 LLM 为每个 chunk 生成 1–2 句“它在全文中的位置与作用”，再拼接后 embedding。效果可能优于标题路径，但有额外 LLM 成本。本项目的原则是：**先评估，再决定是否全量启用**，不默认开启。

### 2.4 Embedding：把语义压进向量

#### 基本机制

Embedding 模型是一个 **bi-encoder**：把 query 和 document 分别独立编码成向量，用相似度衡量语义接近程度。常用余弦相似度：

```text
cos(q, d) = (q · d) / (||q|| · ||d||)
```

大多数模型输出已归一化，此时点积等于余弦相似度。工程上要统一：索引和查询必须使用**同一个模型、同一版本、同一维度、同一种归一化方式**。

#### 维度的含义与锁死原则

- 维度是模型输出空间的坐标数，不是“维度越高越准”这么简单；
- 不同模型的向量空间**不可比较**，换模型等于换坐标系；
- 因此 `embedding_model` + `dimension` 必须写入 manifest；换模型 = 重建 collection。

这是“向量维度一旦建库不可随意改”的真正原因：不是 Qdrant 不能改维度，而是旧向量在新模型下没有意义。

#### 多语言与跨语言检索

中文问题检索英文文档，依赖 embedding 模型的跨语言对齐能力。Qwen `text-embedding-v4`、BGE-M3 这类多语言模型会把“如何注入数据库会话”和 “how to inject a database session”映射到相近区域。但这不是无限的：

- 专有名词、缩写、代码符号的跨语言对齐较弱；
- 中文提问里的英文 API 名（如 `Depends`、`Annotated`）反而更容易被保留；
- 不同语言的语料分布差异会带来偏移。

#### Embedding 的失败模式

向量检索擅长“语义相近、用词不同”，但不擅长：

- 精确符号：`Depends`、`Annotated`、`HTTPException`、`lifespan`、错误码；
- 罕见词 / 新 API 名：训练语料里没有，向量没有稳定位置；
- 否定与数值：`不是 A` 与 `是 A` 可能很近；
- 长文本细节：一个 chunk 里 80% 是背景，关键句只占一行。

这些正是 BM25 / 稀疏检索存在的理由。

### 2.5 BM25：为什么关键词检索仍不可替代

BM25 是经典的词袋检索模型，核心由三部分组成：

```text
score(D, Q) = Σ_{q_i ∈ Q} IDF(q_i)
              · [ f(q_i, D) · (k1 + 1) ]
              / [ f(q_i, D) + k1 · (1 − b + b · |D| / avgdl) ]

IDF(q_i) = ln(1 + (N − n(q_i) + 0.5) / (n(q_i) + 0.5))
```

- **TF 饱和**：词出现次数增加的边际收益递减，避免长文堆词；
- **IDF**：越罕见的词权重越高，`lifespan` 比 `the` 重要；
- **长度归一化**：`b` 控制长文档的惩罚，避免长文天然占优；
- 典型参数：`k1 ∈ [1.2, 2.0]`，`b = 0.75`。

为什么技术文档必须用它：

- 类名、函数名、参数名、错误信息是**精确匹配问题**，不是语义相似问题；
- 用户经常直接贴报错或 API 名，dense 可能召回“语义相关但符号不同”的页面；
- BM25 对未登录词、新 API 名不依赖训练语料。

#### Qdrant 中的 sparse vector

Qdrant 本身只存储和检索稀疏向量（`indices` + `values`），BM25 的权重通常由客户端生成（如 FastEmbed 的 BM25），并可在 collection 上配置 `modifier: "idf"` 让 Qdrant 在查询时应用 IDF。**索引端和查询端必须使用同一套分词 / 加权逻辑**，否则 term 对不上。

#### 一个必须正视的原理性缺口：中文 query → 英文文档

- dense 分支：多语言 embedding 可以跨语言匹配；
- sparse 分支：中文 query 分词后是中文 term，英文文档里是英文 term，**BM25 几乎无法命中**。

如果查询不翻译，hybrid 实际上退化成“dense + 少量英文字号 token 的稀疏召回”。可行的处理方式（需评估）：

1. 查询理解阶段生成英文检索式，只用于 sparse 分支；
2. dense 用原中文 query，sparse 用英文扩展 query，两路各取所长；
3. 或改用多语言稀疏模型（如 BGE-M3 的 sparse 输出 / 多语言 SPLADE）。

这是本项目实现时需要专门验证的点，不能默认“hybrid 一定比 dense 好”。

### 2.6 存储：named vectors + payload

Qdrant 的 collection 可以为一个点存多路向量（`dense`、`sparse`），同时用 payload 存元数据：

```text
chunk_id, parent_id, doc_id
url, title, section_path
product, language, doc_type
docs_version, source_ref, fetched_at, updated_at, is_deprecated
content_type, code_language
text, parent_text
content_sha256, index_build_id
```

两类字段用途不同：

- **向量**用于相似度计算；
- **payload** 用于过滤、展示、引用映射、版本归因。过滤字段需要建 payload index，否则过滤会退化为扫描。

### 2.7 增量索引的本质：复用“没变的部分”

增量索引不是“少写几个点”，而是把整条流水线拆成**可判定是否变化**的阶段：

1. **页面级**：commit diff / `lastmod` / 内容哈希判断页面是否变化；
2. **chunk 级**：稳定 ID（`doc_id + section_hash + chunk_index`）+ 内容哈希判断 chunk 是否变化；
3. **向量级**：以 `content_sha256` 为 key 的 embedding 缓存，未变则直接复用向量。

这里的关键原理：

- **ID 稳定**才有“同一个 chunk”的概念；
- **内容哈希稳定**才能区分“真的变了”和“重新跑了一遍”；
- **embedding 是纯函数**（同一模型 + 同一文本 → 同一向量），所以缓存是安全的。

#### 为什么“全量写点 + 增量算向量”

设计文档选择写入新 collection，而不是原地更新。原因是：

- 小数据集下全量写点的成本远低于其带来的确定性；
- 原地更新容易出现“删了一半、写了一半”的中间状态；
- 新 collection + alias 切换可以做到**要么全旧、要么全新**，且可回滚。

### 2.8 alias 原子切换

Qdrant 的 alias 是一个指向 collection 的别名。写入流程：

```text
1. 创建 fastapi_docs_build_<ts>
2. 全量写入点（只对变更 chunk 调用 embedding）
3. 校验：点数、版本覆盖率、抽样检索、关键问题回归
4. 原子地把 fastapi_docs_current 指向新 collection
5. 保留上一个 build，用于回滚
```

原理：检索端只认 alias，不认具体 collection。切换瞬间完成，没有“部分新、部分旧”的窗口。旧版本保留一个刷新周期，是**可回滚性**的保证。

---

## 3. 在线检索：召回 → 融合 → 精排

### 3.1 查询理解：先把“用户的话”变成“检索的话”

| 步骤 | 解决的问题 | 原理 |
|---|---|---|
| 多轮改写 | 追问里的“它”“上面那个” | 检索需要独立、自包含的 query |
| 版本意图识别 | “旧版怎么写”“v0.3 还能用吗” | 提取版本约束，转化为 metadata filter |
| 意图路由 | 闲聊 / 超范围 / 文档问答 | 避免对所有输入都走 RAG，省成本、减少误答 |
| Multi-Query | 一种说法召不全 | 生成多个同义 query，扩大召回面 |
| HyDE | query 太短、与文档表述差异大 | 先生成假设答案，再用假设答案检索；**必须评估后再开**，否则可能引入假设中的错误语义 |

原则：查询理解是**可选的增益层**，不是必需层。每一项都要有评估数据证明它带来的 Recall / 答案质量提升大于延迟和成本。

### 3.2 Dense 召回：ANN 的近似性

向量检索通常不是精确扫描全部点，而是用 **HNSW**（分层可导航小世界图）等近似最近邻算法：

- 图分多层，上层稀疏用于快速跳转，下层密集用于精细搜索；
- 查询时从上层贪心下行，在候选邻居中扩展；
- `ef_search` 越大，召回越高、延迟越大。

这意味着 **dense top-k 是近似的**：同一个 query 在不同参数、不同索引状态下可能召回不同结果。评估时要固定参数，否则指标会漂移。

### 3.3 Sparse 召回：精确词项匹配

稀疏向量只存非零 term 及权重，检索时按 term 倒排/稀疏索引找候选。它的优势是精确、可解释；劣势是：

- 不做语义泛化（同义词、改写不命中）；
- 依赖分词质量；
- 跨语言几乎失效（见 §2.5）。

### 3.4 为什么必须 hybrid

dense 和 sparse 的失败模式高度互补：

| 场景 | Dense | Sparse |
|---|---|---|
| “怎么注入数据库会话” vs “how to inject a database session” | ✅ 语义对齐 | ❌ 词不重合 |
| `Depends` / `Annotated` / `lifespan` | ⚠️ 符号弱 | ✅ 精确命中 |
| 报错信息片段 | ❌ 易被语义近似干扰 | ✅ 精确 |
| 同义改写、口语化提问 | ✅ | ❌ |
| 新 API 名（未在训练语料） | ⚠️ 向量不稳定 | ✅ 只要文档里有这个词 |

技术文档的特殊性在于：**精确符号和自然语言解释同时大量存在**。只看 dense 会丢符号，只看 BM25 会丢语义，所以 hybrid 是这类语料上的必备基线，而不是“堆技术”。

### 3.5 RRF：为什么用排名融合而不是分数融合

Reciprocal Rank Fusion：

```text
RRF(d) = Σ_{r ∈ R} 1 / (k + rank_r(d))     k = 60
```

原理与优点：

- 只用**排名**，不用原始分数，避免 dense cosine（0~1）与 BM25（无界）量纲不一致；
- 对某一路的分数分布不敏感，鲁棒；
- `k` 起平滑作用：k 越大，靠前与靠后的差距越小；k=60 是常用默认值；
- 一个文档被多路同时召回时，RRF 会自然加分。

代价：丢弃了分数的“置信度”信息。若两路分数都经过良好校准，加权融合可能更好；但那需要额外校准工作，RRF 是更稳的默认。

### 3.6 Rerank：Cross-Encoder 的原理与代价

#### Bi-encoder vs Cross-encoder

```text
Bi-encoder:    score(q, d) = cos(E(q), E(d))        ← 可预计算，快
Cross-encoder: score(q, d) = Transformer([q; d])     ← 联合编码，准，但无法预计算
```

Cross-encoder 让 query 和文档在注意力层中**逐 token 交互**，能判断“这段是否真的回答了这个问题”，而不只是“主题是否接近”。代价是每个候选都要跑一次模型，无法对全库预计算。

因此流程必须是：

```text
召回候选（dense 40 + sparse 40 → RRF → top 20）
  → cross-encoder 精排
  → top 6–8 进入上下文
```

**不要对全库 rerank**：成本与延迟不可接受。也不要指望 rerank 救回没召回的文档：它只能重排候选。

#### 阈值与拒答

Cross-encoder 的输出分数**不是校准过的概率**。用固定阈值（如 0.5）判断“是否相关”通常是错的。正确做法：

1. 在 golden set 上观察正/负样本的分数分布；
2. 选一个能平衡“误拒答”和“误回答”的分位点；
3. 记录阈值对应的拒答率与误答率，并随模型/语料更新重新校准。

#### 多语言 reranker

中文 query + 英文 passage 要求 reranker 本身支持跨语言。`bge-reranker-base` 主要覆盖中英；更稳的选择是 `bge-reranker-v2-m3` 这类多语言模型，或者用 API rerank。实现时应把 reranker 模型写进配置与评估报告，避免“换了模型但指标不可比”。

### 3.7 过滤与版本感知检索

先过滤还是后过滤，是一个原理性问题：

- **后过滤**（先 ANN 再按 payload 删）：过滤条件严格时，top-k 可能被删光，召回率骤降；
- **预过滤**（Qdrant 在 ANN 搜索时带 filter）：在图上只走满足条件的节点，结果更可靠。

所以 `product / language / doc_type / docs_version / is_deprecated` 都应作为 Qdrant filter 传入，而不是取回后再筛。过滤字段必须建 payload index。

版本策略：

- 未指定版本：默认当前版本；旧版本可召回但降权，避免完全丢失历史答案；
- 指定版本：按 `docs_version` 过滤，同时召回当前版本做差异对比；
- 上下文同时包含新旧版本：生成阶段必须显式标注，不能混合成一段结论。

原理：版本不是“元数据装饰”，而是**检索约束 + 生成约束**。只在 payload 里存版本、不在检索和提示中使用，等于没做版本感知。

### 3.8 上下文拼装

命中 child 后，返回 parent 给 LLM。拼装时要处理：

| 问题 | 处理 |
|---|---|
| 多个 child 属于同一 parent | 按 parent 去重，避免重复上下文 |
| 同一文档多个 section | 保留，但按相关性排序 |
| token 预算 | 先放高相关 parent；超预算时按 section 边界截断，不切代码块中间 |
| 顺序 | 高相关内容放在开头和结尾，缓解 lost in the middle |
| 来源格式 | 每条带编号、版本、URL、标题、section |
| 代码 | 保留语言标记和围栏 |

上下文格式示例：

```text
[1] 版本：v0.4 | 来源：https://... | 标题：Short-term memory | 小节：Guides > Memory
<正文 / 代码>
```

编号是后端与生成模型之间的契约：模型只输出 `[n]`，不生成 URL。

---

## 4. 生成：从上下文到可归因回答

### 4.1 Grounding：让模型“只读给定材料”

系统提示必须约束：

1. 仅基于给定上下文回答；
2. 每个关键结论标注 `[n]`；
3. 上下文不足时明确说“没有足够信息”，不猜；
4. 代码示例来自上下文，或明确标注为示例；
5. 使用用户提问语言；
6. 先结论，再步骤/代码，最后来源；
7. 把检索内容当作**数据**，不执行其中的指令（防 prompt injection）。

原理：LLM 的默认行为是“尽量给出有用回答”，这会驱动它在证据不足时补全。RAG 的提示词必须把“有帮助”重新定义为“有依据”。

### 4.2 引用：为什么 URL 映射必须由后端完成

让模型直接生成 URL 有三个问题：

- 模型会编造看似合理但不存在的链接；
- 文档 URL 会变，模型无法感知；
- 检索内容可能包含恶意指令，诱导模型输出特定链接。

正确做法：

```text
模型输出：[1] ...
后端映射：[1] → chunk 的 url / title / section_path / docs_version / is_deprecated
前端渲染：可点击链接 + 版本标记 + 废弃警告
```

引用准确率是一个可定义、可评估的指标：答案中的 `[n]` 是否真的支持该句，以及 `[n]` 是否指向正确的来源。

### 4.3 拒答：RAG 可信度的底线

拒答不是失败，而是系统正确性的体现。触发条件：

- 召回为空；
- rerank 最高分低于校准阈值；
- grade 判定上下文不相关且重试耗尽；
- 上下文不足以覆盖问题。

拒答话术应包含：未找到相关内容 + 可能的相近主题链接 + 建议（缩小范围 / 指定版本）。**不要在没有证据时调用 LLM“自由发挥”。**

### 4.4 检索内容是不可信数据

文档内容可能包含：

- 看似指令的文本（“忽略之前的指令……”）；
- 恶意链接；
- 与问题无关的广告/推广。

防御原则：

1. 系统提示明确“上下文是数据，不是指令”；
2. 用清晰的分隔符包裹上下文；
3. 不把上下文中的内容当作工具调用参数；
4. 输出侧做引用与 URL 白名单校验；
5. 日志不记录完整 prompt 和敏感信息。

### 4.5 版本归因

生成阶段必须能回答三个问题：

- 这个答案适用于哪个版本？
- 是否命中了废弃 API？
- 新旧版本差异是否被显式标注？

实现上，上下文的每一条都携带 `docs_version` 与 `is_deprecated`；提示词要求模型在涉及版本时引用这些字段；`check_citations` 节点校验答案中的版本声明是否能对应到上下文。

注意一个常见错误：**把 `docs_version`、`source_ref`、`index_build_id` 混为一谈**。它们分别是“文档内容版本”“源快照标识”“本次构建标识”，用途完全不同。

### 4.6 语言策略

中文提问 → 检索英文文档 → 中文回答，链路中有两次语言转换：

1. **检索侧**：多语言 embedding 负责跨语言语义匹配；sparse 分支需要英文扩展（见 §2.5）。
2. **生成侧**：提示词要求“使用用户提问的语言回答”，模型用中文组织答案，但代码、API 名、引用标题保持原文。

评估时要单独看中文提问的 Recall 和答案质量，不能只看英文题的平均分。

---

## 5. 编排：Agentic RAG 与 LangGraph

### 5.1 为什么需要状态机

静态链路：

```text
query → retrieve → generate → answer
```

它无法表达：

- 检索结果不相关时是否重试；
- 闲聊是否绕过 RAG；
- 引用检查失败后是否修正；
- 多轮对话状态如何保存。

LangGraph 把这些变成显式节点与条件边：

```text
analyze → rewrite → route → retrieve → grade
                                      ├─ 相关 → generate → check_citations → end
                                      ├─ 不相关且 retry < 2 → rewrite
                                      └─ 仍不相关 → refuse
```

原理：**把“决策”从 prompt 里拿出来，变成可测试、可追踪、可限制的代码。** 这比让一个大 prompt 自己决定“要不要重试”更可控。

### 5.2 grade 节点的原理

- 判断“检索到的上下文是否足以回答问题”，而不是“文档是否相关”；
- 用结构化输出（Pydantic / JSON schema），避免解析自然语言；
- 评分模型与生成模型尽量分开，降低自评偏差；
- 重试必须有上限（建议 2 次），否则可能无限循环、烧钱。

grade 的价值：在生成之前拦截“用不相关上下文硬答”的情况。代价是多一次 LLM 调用；可以用小模型降低成本。

### 5.3 降级路径

| 故障 | 降级 |
|---|---|
| rerank 模型不可用 / 超时 | 退回 RRF 排序 |
| 首次召回为空 | 查询改写后重试一次 |
| 仍为空 | 不调用 LLM，直接拒答 |
| LLM 超时 | 返回缓存答案（若命中）或明确错误 |
| 引用检查失败 | 修正一次；仍失败则附警告或降级为无引用答案 |
| 版本冲突 | 显式列出两版结论，不合并 |

原则：**每一层都要有“坏掉时怎么办”，而不是假设依赖永远可用。**

### 5.4 流式的原理

SSE（Server-Sent Events）是单向、基于文本的事件流。本项目的顺序：

```text
sources → token*N → usage → done
```

- 检索在生成前完成，所以 `sources` 可以先发；
- `token` 逐段发送，降低首屏等待；
- `usage` 发送 token / 成本；
- 异常必须显式发 `error`，否则前端会一直等待。

工程细节：`/chat/stream` 是 POST，原生 `EventSource` 只支持 GET，必须用 `fetch` + `ReadableStream` 解析分帧。SSE 的“事件边界”由空行分隔，解析器必须处理跨 chunk 的半条消息。

---

## 6. 项目签名：版本感知的活知识库

### 6.1 三个版本字段的语义

| 字段 | 回答的问题 | 示例 |
|---|---|---|
| `docs_version` | 这份内容对应 FastAPI 的哪个版本？ | `0.162.0`、`unknown` |
| `source_ref` | 这份内容来自哪个源快照？ | `git:7a0a3f0...` |
| `index_build_id` | 它是哪次索引构建写入的？ | `20261026T100000Z` |

为什么要分开：

- 同一个 `docs_version` 可能被多次重建（`index_build_id` 不同）；
- 同一个 `source_ref` 可能对应多个页面；
- 无法判断版本时，`docs_version` 必须是 `unknown`，而不是拿构建时间冒充。

### 6.2 变更检测的层级

```text
Git commit diff  >  content_sha256
```

- **Git diff**：最精确，能区分新增/修改/删除/重命名；
- **内容哈希**：兜底，能发现内容变化，但受空白、导航噪声影响。

FastAPI 文档不需要 sitemap 分支：数据源就是 Git 仓库，commit diff 已足够；版本号另外从 `release-notes.md` 或 git tag 读取。注意 `docs_version` 与 `source_ref` 不是一回事：同一个 FastAPI 版本可能对应多个文档提交，同一个 commit 也可能只是修正错别字而版本号未变。

哈希前必须做规范化（统一换行、去掉导航/页脚、去掉动态时间戳等），否则会产生大量“假变更”，导致不必要的重嵌入。

### 6.3 增量重建的数学

设总 chunk 数为 $N$，变更 chunk 数为 $M$，则：

```text
重嵌入比例 = M / N
```

目标不是“增量一定更快”，而是：

- 小幅变更时，$M \ll N$，embedding 成本和时间显著下降；
- 变更集判定正确，不漏更、不误更；
- 未变 chunk 的向量从缓存复用，而不是重新计算。

embedding 缓存 key 用 `content_sha256`（或 `model + content_sha256`），确保换模型时不会误用旧向量。

### 6.4 陈旧度与更新延迟

两个容易混淆的指标：

```text
陈旧度(staleness) = now − fetched_at
更新延迟(lag)     = indexed_at − source_changed_at
```

- 陈旧度衡量“我们多久没抓了”；
- 更新延迟衡量“从源变更到可检索花了多久”。

设计目标“文档变更 ≤ 24h 可检索”对应的是 lag，不是 staleness。只有 `source_ref` / commit 时间可用时，lag 才能准确计算。

### 6.5 废弃页面的处理

不要立即物理删除：

- 历史问题仍可能需要旧答案；
- 用户可能仍在用旧 API；
- 直接删除会让“迁移类问题”失去上下文。

正确做法：

1. 标记 `is_deprecated=true`；
2. 默认检索降权或过滤；
3. 若用户明确问旧版本/旧 API，仍可召回；
4. 生成时提示“该 API 已废弃”并给出上下文中的迁移写法；
5. 确认删除后再清理。

FastAPI 场景要把两种标记分开：

- **section 级废弃**：标题明确写 `(deprecated)`，或 warning 块说明整个 API 已废弃 → `is_deprecated=true`；
- **正文提及废弃**：只是顺带说“某参数已废弃” → `deprecation_hint=true`，不能把整个 chunk 判为废弃。

典型例子：`advanced/events.md` 的 “Alternative Events (deprecated)” 对应 `@app.on_event`，迁移目标是 `lifespan`；生成时必须提示这一点，而不是继续推荐旧写法。

---

## 7. 评估原理

### 7.1 分层评估

| 层 | 指标 | 回答的问题 |
|---|---|---|
| 检索 | Recall@k、MRR、nDCG、延迟 | 证据找得到吗？排得前吗？ |
| 生成 | Faithfulness、Answer Relevancy、引用准确率、拒答正确率 | 答案有依据吗？答对了吗？ |
| 版本 | 版本归因准确率、废弃 API 识别率、陈旧度、更新延迟 | 版本信息可信吗？ |
| 系统 | 首 token 延迟、端到端延迟、成本、错误率 | 能用吗？贵吗？ |

### 7.2 检索指标

```text
Recall@k = |相关文档 ∩ top-k| / |相关文档|
MRR      = (1/|Q|) · Σ 1 / rank_first_relevant
DCG@k    = Σ_{i=1..k} rel_i / log2(i + 1)
NDCG@k   = DCG@k / IDCG@k
```

- **Recall@k** 衡量“有没有召回到”，是 RAG 最关键的上限指标；
- **MRR** 关注第一个相关结果的位置；
- **nDCG** 同时考虑相关性和排序位置。

技术文档场景要特别注意：`golden_urls` 可能对应多个 section，评估时要按 chunk / section 粒度对齐，而不是只按页面 URL。

### 7.3 生成指标

- **Faithfulness**：答案的每个 claim 是否被上下文支持；
- **Answer Relevancy**：答案是否真正回答了问题；
- **Context Precision / Recall**：上下文本身的相关性与覆盖度；
- **引用准确率**：`[n]` 是否指向支持该句的来源；
- **拒答正确率**：无答案题是否拒答，有答案题是否误拒；
- **代码可运行率**：抽检代码示例是否能跑。

### 7.4 LLM-as-Judge 的偏差

用 LLM 打分方便，但有系统性偏差：

- 偏好长答案、格式漂亮的答案；
- 与生成模型同源时，会高估“自己风格”的答案；
- 对事实细节的核查不稳定。

对策：

- judge 与生成模型尽量不同；
- 人工校准 20–30 条；
- 多指标交叉（faithfulness + 引用准确率 + 人工抽检）；
- 把 judge 的 prompt 和模型版本记录下来，保证可比。

### 7.5 bad case 闭环

```text
跑评估 → 聚类 bad case → 归因到环节
  → 切分 / 检索 / 融合 / rerank / 生成 / 版本
  → 单点干预 → 回归评估 → 记录前后指标
```

原则：**每次只改一个变量**，否则无法知道收益来自哪里。失败 query 要进入 bad case 池，反哺 golden set。

---

## 8. 延迟与成本预算

设计目标：检索 < 500ms，首 token < 2s。典型耗时构成（数量级，不是实测值）：

| 环节 | 典型耗时 | 说明 |
|---|---|---|
| 查询改写 / 版本识别 | 100–500ms | 小模型；可与检索并行优化 |
| Query embedding | 50–200ms | 取决于 API / 本地模型 |
| dense + sparse 检索 | 10–50ms | Qdrant 本地 |
| RRF | < 5ms | 纯内存 |
| rerank top-20 | 100–400ms | 本地 CPU cross-encoder |
| 上下文拼装 | < 10ms | 去重 / 截断 |
| LLM 首 token | 500–1500ms | 云端 API 波动大 |

优化原则：

- 能并行的并行（dense 与 sparse、改写与检索的某些部分）；
- 能缓存的不重复算（答案缓存、检索缓存、embedding 缓存）；
- 用小模型做路由 / 改写 / 评分；
- 上下文只放 top 6–8，不塞满窗口；
- 缓存 key 必须包含 `index_build_id` 或版本，避免用旧索引的答案回答新问题。

---

## 9. 失败模式地图

| 现象 | 最可能的环节 | 干预 |
|---|---|---|
| 明明文档里有，却答“没找到” | 切分 / 召回 / 阈值 | 检查 chunk 边界、Recall@k、阈值校准 |
| 召回了相关页，但答案引用了错段落 | rerank / 引用映射 | 看 rerank 分数、引用检查 |
| 代码示例不完整 / 跑不了 | 切分 | 代码块保护、按函数边界切 |
| 中文问题召回差 | embedding / sparse 分支 | 多语言 embedding、英文扩展 query |
| 新 API 名搜不到 | dense / 分词 | 加 BM25、检查索引是否更新 |
| 答案是旧版本写法 | 版本过滤 / 增量索引 | 检查 `docs_version`、`is_deprecated`、alias 指向 |
| 文档更新了但系统不知道 | 变更检测 / 调度 | 检查 `source_ref`、diff 报告、lag |
| 答得很流畅但内容是编的 | 生成 / 上下文不足 | 拒答阈值、faithfulness 评估、引用检查 |
| 引用链接点不开 | URL 规范化 / 映射 | canonical URL、锚点校验 |
| 首 token 很慢 | 检索 + LLM 串行 | 并行化、sources 先行、缓存 |

这张表的价值：**遇到 bad case 时先定位环节，再改代码**，而不是直接调 prompt。

---

## 10. 走一遍：一个中文问题的一生

问题：**“FastAPI 里 `on_event` 已废弃，怎么迁移到 `lifespan`？”**

```text
1. analyze
   - 意图：how-to + 版本/废弃意图
   - 产品：fastapi
   - 语言：python
   - 版本意图：latest
   - 语言：中文提问
   - 过滤：include_deprecated=true

2. rewrite / query expansion
   - 独立问题（若有多轮）：保留原意
   - 英文检索式（供 sparse 分支）："FastAPI on_event deprecated migrate to lifespan"
   - 可选 multi-query：app.on_event / lifespan events / startup shutdown

3. retrieve
   - dense：中文 query → 多语言 embedding → 语义召回
   - sparse：英文扩展 → BM25 → 精确命中 on_event / lifespan
   - filter：product=fastapi, language=python, include_deprecated=true
   - dense top40 + sparse top40 → RRF(k=60) → top20

4. rerank
   - cross-encoder 对 (query, child) 精排
   - 取 top 6–8；最高分低于阈值则走拒答

5. context assembly
   - child → parent 展开
   - 按 parent 去重，控制 token 预算
   - 每条带编号、docs_version、URL、section
   - 命中 deprecated section 时在 context 头标注 DEPRECATED

6. generate
   - 中文回答，代码/API 名保留英文
   - 关键结论标注 [1][2]
   - 显式说明 on_event 已废弃，给出 lifespan 的迁移写法

7. check_citations
   - [n] 是否存在、是否支持该句
   - 版本/废弃声明是否能对应到上下文
   - 失败则修正或降级

8. stream
   - sources → token*N → usage → done
```

这个例子体现了整条链路的分工：**中文语义靠 dense，英文符号靠 sparse，排序靠 rerank，完整性靠 parent，可信度靠引用检查，版本靠 metadata + 提示约束。**

---

## 11. 术语表

| 术语 | 含义 |
|---|---|
| Bi-encoder | query 与文档分别编码，可预计算，适合召回 |
| Cross-encoder | query 与文档联合编码，精度高但无法预计算，适合精排 |
| ANN / HNSW | 近似最近邻检索及其图索引算法 |
| BM25 | 基于词频、逆文档频率、长度归一化的稀疏检索模型 |
| RRF | 基于排名的多路召回融合方法 |
| Recall@k | top-k 中召回到的相关内容比例 |
| MRR | 第一个相关结果排名的倒数均值 |
| nDCG | 考虑排序位置的归一化折损累计增益 |
| Parent-Child | 小 chunk 召回、大 chunk 生成的两级切分 |
| Contextual Retrieval | 为 chunk 添加上下文说明后再 embedding |
| HyDE | 先生成假设答案，再用假设答案检索 |
| Faithfulness | 答案是否被给定上下文支持 |
| Staleness | 页面内容距上次抓取的时间 |
| Index lag | 源变更到可检索的延迟 |
| Alias | Qdrant 中指向 collection 的别名，用于原子切换 |
| Named vectors | 一个点携带多路命名向量（如 dense + sparse） |
| `docs_version` | 文档内容对应的版本 |
| `source_ref` | 源快照标识（commit SHA / ETag） |
| `index_build_id` | 本次索引构建的标识 |

---

## 12. 待实现时验证的原理性风险

以下不是设计错误，而是需要在实现/评估阶段用数据确认的点：

1. **中文 query 的 sparse 分支**：不翻译时 BM25 基本失效；英文扩展是否有效、是否引入噪声，需要 A/B。
2. **reranker 的跨语言能力**：`bge-reranker-base` 对中英混合的效果要实测；必要时换 `bge-reranker-v2-m3` 这类多语言 reranker。
3. **拒答阈值**：cross-encoder 分数不是概率，必须用 golden set 校准，不能拍一个 0.5。
4. **标题前缀对 BM25 的影响**：标题注入对 dense 通常有益，但会改变 BM25 词频分布；建议 dense 用带标题文本、sparse 用正文 + 标题字段分别验证。
5. **版本只覆盖最新版**：FastAPI 文档仓库不保存按版本分的页面；历史版本要靠 git tag checkout。若不做历史快照，旧版问题只能从 release notes 回答，必须明确告知覆盖范围，不能假装读过旧文档。
6. **增量索引的正确性**：必须有“变更集 → 点数 → 抽样检索 → 回滚”的校验，否则增量逻辑的 bug 会静默污染知识库。
7. **评估口径一致性**：每次实验固定 embedding 模型、reranker、top-k、prompt 和 judge 模型，否则指标不可比。
8. **include 展开的正确性**：`{* ... *}` 可能有嵌套目录、`hl[...]` 参数、缺失文件；必须有单测覆盖，否则代码示例会静默丢失或错位。
9. **reference 页面为空**：`::: fastapi.FastAPI` 是构建指令，不是正文；若误把它当作 API 文档索引，会得到“有页面、无内容”的假召回。
10. **release notes / meta 页面污染**：`release-notes.md` 版本号密集，容易抢占普通问题的 top-k；`about/`、`resources/` 等元信息页应打 `doc_type=meta` 并默认过滤。
11. **废弃标记的粒度**：只在 section 明确废弃时置 `is_deprecated=true`；正文顺带提废弃只置 `deprecation_hint=true`，否则会把正常内容整体降权。

---

> 本文档是原理层说明，不替代 `PROJECT_DESIGN.md` 的实现决策。若后续实验推翻了某个假设（例如 hybrid 在中文 query 上没有提升），应以评估数据为准更新本文档与设计文档。
