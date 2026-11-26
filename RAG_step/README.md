# 从 0 到 1 设计一个 RAG（最小可用实现）

一个**零第三方依赖**（只用 Python 标准库，不需要 pip install）的 RAG 工程，约 1200 行（含大量中文注释），七个步骤各一个文件，
每一步都能单独运行、看到中间结果。

示例语料是一套虚构的员工手册，配套 12 条评估问题。
跑通后你可以直接把自己的文档丢进 `data/docs/`，或者把向量化模块换成真正的 embedding 模型。

---

## 0. 60 秒跑起来

```bash
cd /Users/f/Desktop/scholar/01step/RAG_step

uv run python build_index.py                                  # 离线：建索引
uv run python ask.py "满 5 年司龄的员工年假有多少天？"          # 在线：问答
uv run python eval.py                                         # 评估：12 条问题
```

`uv run` 用的是 `01step` 仓库根目录的共享环境（首次先在根目录执行 `uv sync --all-packages`）。
因为只用标准库，直接 `python3 build_index.py` 也能跑，两种写法任选。

预期看到：

```
[build] 文档 5 篇 → 块 15 个 → 词表 1957 维，平均块长 281.7 字符，耗时 0.007s
...
样本 12 条 | hit@3 100.0% | MRR 1.000 | 答案命中率 100.0%
```

没有 API Key 也能跑：生成环节会退化成"检索原文摘录"，正好用来验证检索准不准。
配了模型后同一条命令就会输出带 `[1]` 引用的自然语言答案（见第 7 节）。

---

## 1. RAG 解决的是哪三个问题

直接问大模型"我们公司年假几天"，它答不出来，或者一本正经地编。

RAG（Retrieval-Augmented Generation）的思路很简单：**别让模型背，先查资料再回答**。

| 问题 | 大模型裸答 | RAG |
| --- | --- | --- |
| 不知道你的私有资料 | 编一个 | 检索到原文再回答 |
| 上下文塞不下整本书 | 截断 / 丢失 | 只取最相关的几段 |
| 回答无法核实 | 没法验证 | 每条结论带 `[编号]` 可回溯 |

一句话：**RAG 把"记忆"从模型参数里搬到外部知识库，模型只负责理解和表达。**

---

## 2. 设计：两条链路，一个边界

```
离线（跑一次，分钟级）                                 在线（每次提问，毫秒级）
─────────────────────────────                       ─────────────────────────────
docs/*.md                                            question
   │                                                    │
   ├─ 1. load      读取 + 元数据（doc_id / title）        ├─ 3. embed   同一个 embedder
   ├─ 2. chunk     按结构切块 + 重叠                       │
   ├─ 3. embed     文本 → 向量                            ├─ 4. retrieve 余弦 top-k
   └─ 4. index     块 + 向量 → rag_index.json ──────────► │              阈值过滤
                                                       │              相邻块扩展
                                                       ├─ 5. prompt   资料编号 + 问题 + 约束
                                                       └─ 6. answer   LLM / 抽取式兜底
                                                                     │
                                                            7. evaluate  hit@k / MRR / 答案命中
```

**关键设计决策（这六条决定了后面所有代码的形状）：**

1. **离线建索引、在线只读索引**。问答时不碰原始文档，所以 QPS 再高也不影响索引。
2. **embedder 必须和索引一起落盘**。同一个问题在离线阶段和在线阶段必须用同一套 idf / 同一套模型，否则向量不在一个空间，检索全错。这是 RAG 最常见的低级事故。
3. **检索单元小、上下文大**。块按 300~500 字符切（精度高），送给模型时再拼上相邻块（完整性好）。
4. **结构优先于长度**。表格、标题、代码块先保住，再考虑不超过 chunk_size。
5. **生成层可插拔**。有 LLM 调 API，没有就走抽取式，链路永远能跑通、能评估。
6. **一切可量化**。每个参数改动都要能用 hit@k / MRR 说清楚是变好还是变坏。

---

## 3. 目录

```
RAG_step/
├── data/
│   ├── docs/                     # 知识库：5 篇员工手册
│   └── eval/questions.jsonl      # 评估集：question / expect / source
├── rag/                          # 七个模块，一个模块一个概念
│   ├── config.py                 # 所有参数集中在这里
│   ├── loader.py                 # 1. 加载
│   ├── chunker.py                # 2. 切块
│   ├── embedder.py               # 3. 向量化（TF-IDF 稀疏向量）
│   ├── vector_store.py           # 4. 索引 + 余弦检索
│   ├── retriever.py              # 4+. 阈值过滤 / 相邻块扩展
│   ├── generator.py              # 5~6. prompt + 生成
│   ├── pipeline.py               # 串起来：RAG.build() / RAG.ask()
│   └── evaluate.py               # 7. 评估
├── steps/                        # 每一步一个可运行脚本
│   ├── step1_load.py ... step7_eval.py
├── build_index.py                # 建索引入口
├── ask.py                        # 问答入口（支持交互模式）
├── eval.py                       # 评估入口
└── index/rag_index.json          # 产物（build 后生成）
```

---

## 4. 七个步骤，逐步拆

每一步都按「做什么 → 看哪里 → 跑什么 → 这一课的坑」来讲。

### Step 1 · 加载

```bash
uv run python steps/step1_load.py
```

读 `data/docs` 下所有 `.md/.txt`，产出 `Document(doc_id, title, text, path)`。

**坑**：`doc_id` 必须是稳定唯一的标识（这里用文件名）。后面引用、评估、增量更新全靠它回溯。
PDF / Word 只是多一层解析器，解析完仍然收敛到"字符串 + 元数据"，结构不用改。

### Step 2 · 切块（最容易翻车的一步）

```bash
uv run python steps/step2_chunk.py
```

`chunk_size=400` 字符、`overlap=50` 字符。先按行/段落切，再拼到上限，并做三件结构保护：

- **整张 markdown 表格不可分割** —— 表格拆开后只剩竖线，表头和数据行分家；
- **标题不留在块尾** —— 块尾是 `## 差旅住宿标准` 就把它挪到下一块开头；
- **重叠从标点之后开始** —— 不会把半句话复制到下一块。

```
块太大 → 检索精度掉，噪声多，浪费 context
块太小 → 语义不完整，答案被切断
有重叠 → 答案正好跨边界时不会丢
```

**这一课的核心**：切块不是"按字数硬切"，是**保住结构**。PDF 的分页、代码块、列表同理。

### Step 3 · 向量化

```bash
uv run python steps/step3_embed.py
```

中文不分词，直接 **单字 + 相邻双字**，再算 TF-IDF：`tf = 1+log(词频)`，`idf = log((1+N)/(1+df))+1`，最后 L2 归一化。

```python
"年假有多少天？" → ['年','假','多','少','天','年假','少天']
```

两个真实的中文坑，代码里都处理了：

1. **跨词 bigram 是噪声**：`"假有"`、`"有多"` 这种组合会被当成罕见词，权重反而最高。
   规则：bigram 任一端是停用词就不生成。
2. **疑问词 idf 最高**：`"多少"` 在语料里几乎不出现，idf 拉满，会把检索带偏。直接把疑问词组加进停用词。

**为什么要知道这些**：换成语义模型（bge-m3 等）后这些问题会自动消失，但你要能判断"检索不准"到底是切块、分词还是模型的问题。

**局限**（也是下一步的动机）：问「休假」不会命中「年假」，问英文不会命中中文，问「住宿费」可能命中「住宿」但不理解「费用」。TF-IDF 只认字面。

### Step 4 · 索引与检索

```bash
uv run python steps/step4_index.py
```

索引就是「块数组 + 向量数组」，检索是暴力余弦排序。10 万块以内都够快，也是所有向量库的底层。

真正让检索像样的是三个后处理（`rag/retriever.py`）：

| 处理 | 参数 | 作用 |
| --- | --- | --- |
| 相对阈值 | `min_score_ratio=0.3` | 低于最高分 30% 的文档丢掉，清掉同批噪声 |
| 绝对阈值 | `min_score=0.10` | 最高分都不到 0.10，判定语料里没有，直接拒答 |
| 相邻块扩展 | `expand_neighbors=True` | 命中块前后各带一块，保证上下文连续 |

跑一下能看到真实分数：

```
满 5 年司龄的年假有多少天？   top1 = 0.283   → 进入 prompt
公司去年的营收是多少？        top1 = 0.073   → 被 0.10 挡住，返回"无法回答"
```

**坑**：相邻块扩展会"传染"——一次命中多个块时整篇文档都会被拉进来。所以有 `max_chunks_per_doc=2`，
并且扩展是**围绕最强命中块取连续窗口**，不能出现 0 号和 2 号块拼一起、中间缺一块的情况。

### Step 5 · 组装 prompt

```bash
uv run python steps/step5_prompt.py
```

同样一份资料，prompt 写得松就会编。system 里五条规则各防一件事：

1. 只用资料 → 防止拿训练时的旧知识答公司制度
2. 没有就说不知道 → 防止"合理猜测"
3. 数字原样给出 → 防止把「15 天」写成「两周」
4. 标注 `[编号]` → 用户能点回原文核对
5. 禁止编造编号 → 防止引用一个不存在的 `[6]`

资料用 `[1] 来源：xxx` 编号，和最终引用一一对应。上下文控制在 `max_context_chars=2000` 字符左右——
**不是越多越好**，塞 10 个块模型更容易被无关内容带偏。

### Step 6 · 生成

```bash
uv run python steps/step6_answer.py
```

`rag/generator.py` 有两种模式：

- **llm**：配了 `LLM_API_KEY` 就调 OpenAI 兼容接口（`/chat/completions`），DeepSeek、豆包方舟、vLLM、Ollama 全部可用；
- **extractive**：没配就贴最相关的原文块，明确标注"这是摘录，不是回答"。

为什么要有兜底：**先确认检索找对了地方，再谈模型答得好不好**。检索错，再强的模型也救不回来。
调用失败（网络、限额、401）也会自动降级到 extractive，并在答案末尾说明原因。

### Step 7 · 评估

```bash
uv run python steps/step7_eval.py
uv run python steps/step7_eval.py -k 1          # 只看第一名
uv run python steps/step7_eval.py --no-expand   # 关掉相邻块扩展做对照
```

评估集三列就够：`question` / `expect`（答案里应出现的关键词）/ `source`（正确文档）。

| 指标 | 含义 | 本次结果 |
| --- | --- | --- |
| hit@3 | 正确文档是否在前 3 | 100% |
| MRR | 正确文档排第几（1/rank 均值） | 1.000 |
| 答案命中率 | 答案是否含预期关键词 | 100% |

**调参的正确姿势**：

```
数字涨了但答案变差  → 问题在生成层（prompt / 模型）
数字不涨            → 问题在检索层（切块 / embedding / query 表述）
```

改 `chunk_size`、`top_k`、`min_score` 就重跑 `eval.py`，用数字说话。
这套 12 条样本偏简单，只用于验证链路通不通，不足以证明 A/B 优劣——**你自己的业务评估集才是准星**。

---

## 5. 参数表（都在 `rag/config.py`）

| 参数 | 默认 | 说明 | 什么时候调 |
| --- | --- | --- | --- |
| `chunk_size` | 400 | 每块最大字符数 | 制度条文 300~500；合同/论文 600~1000 |
| `chunk_overlap` | 50 | 相邻块重叠字符数 | 答案容易跨边界时调到 80~120 |
| `top_k` | 3 | 送模型的文档数 | 噪声多就降，召回不够就升（先看 hit@k） |
| `expand_neighbors` | True | 是否带相邻块 | 上下文被截断时开，噪声多时关 |
| `max_chunks_per_doc` | 2 | 每文档最多几块 | 长文档可以 3~4 |
| `min_score_ratio` | 0.3 | 相对分数下限 | 单主题语料可 0.5，多主题 0.2 |
| `min_score` | 0.10 | 绝对分数下限 | **换 embedding 模型后必须重调**（cosine 分布不同） |
| `max_context_chars` | 2000 | prompt 里资料上限 | 按模型上下文和成本调 |
| `temperature` | 0.0 | 生成温度 | 事实问答保持 0 |

`min_score` 是 TF-IDF 量纲下的经验值。换成 dense embedding 后 cosine 一般在 0.3~0.9，
这个阈值要重新用评估集标定，不能照搬。

---

## 6. 换成你自己的文档

```bash
# 1. 把 .md / .txt 丢进去（支持子目录）
cp -r ~/你的资料/* data/docs/

# 2. 改评估集（这一步最重要，别跳过）
#    data/eval/questions.jsonl 每行一条：
#    {"id":"q01","question":"...","expect":["关键词"],"source":"文件名.md"}

# 3. 重建 + 评估
uv run python build_index.py
uv run python eval.py
uv run python ask.py "你的问题"
```

文档格式建议：**一篇文档一个主题**，标题用 `##` 分节。切块器靠标题和空行判断边界，
一坨没有结构的纯文本，切出来的块质量会差很多。

---

## 7. 接自己的模型（不改代码）

```bash
export LLM_API_KEY=sk-xxx
export LLM_BASE_URL=https://api.deepseek.com/v1
export LLM_MODEL=deepseek-chat
uv run python ask.py "满 5 年司龄年假多少天？"
```

各厂商的 base_url（都是 OpenAI 兼容）：

| 服务 | LLM_BASE_URL | 常用 model |
| --- | --- | --- |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` |
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat` |
| 豆包 / 方舟 | `https://ark.cn-beijing.volces.com/api/v3` | 你的推理接入点 ID |
| Ollama 本地 | `http://localhost:11434/v1` | `qwen2.5:7b` |

接上后 `mode` 会变成 `llm`，答案变成一句话 + `[1]` 引用。想看真正发出去的 prompt，加 `--show-prompt`。

---

## 8. 换成稠密向量 / 向量库

`embedder.py` 的接口只有两个方法：`fit(texts)` 和 `transform(text) -> dict`。
换成 embedding API，就是把"token → 权重"换成"维度下标 → 值"：

```python
class OpenAIEmbedder:
    def fit(self, texts): self.dim = 1536          # 稠密模型不需要拟合
    def transform(self, text):
        vec = call_embedding_api(text)              # 返回 list[float]
        vec = vec / norm(vec)                       # 同样 L2 归一化
        return {str(i): v for i, v in enumerate(vec)}
```

其余代码一行都不用改：`vector_store` 存的是 dict，JSON 照样能序列化；`cosine` 对稠密/稀疏都成立。

换向量库（FAISS / Milvus / pgvector）只动 `vector_store.py` 的 `add` / `search`：

```python
# store.search(query, top_k) 的语义不变：返回 [(score, chunk), ...]
```

**一个必须注意的点**：换 embedding 模型 = 重新建索引 + 重新标定 `min_score`。
旧索引和新 query 不在一个向量空间，检索会全错。

---

## 9. 生产化清单（按优先级）

现在的实现是"能跑通、能评估"的最小闭环，离生产还差这些。按投入产出排序：

1. **业务评估集**（50~200 条真实问题，含"答不了"的）。没有它，后面所有优化都是盲调。
2. **稠密向量 + 混合检索**：dense（语义）+ BM25（字面）各取 top20，RRF 融合。专有名词、编号、型号靠 BM25 兜底。
3. **rerank**：cross-encoder（bge-reranker）对 top20 重排，取 top5。通常是最划算的一步，能明显提精度。
4. **query 改写**：多轮对话里把"那病假呢"补全成"病假需要什么证明"；或让模型生成 2~3 个不同角度的查询再召回。
5. **引用校验**：答案里的每个 `[n]` 必须能在资料里找到支撑句，否则删掉或重答。
6. **增量索引**：按 `doc_id + 内容哈希` 做 upsert / delete，别每次全量重建。
7. **权限过滤**：检索前按用户部门过滤 metadata，别让实习生搜到薪酬文档。
8. **可观测**：记录每个问题的 top-k 分数、命中块、回答、用户反馈，用来找 badcase。

---

## 10. 十个最常见的坑

1. **建索引和查询用了不同 embedder** → 向量不在一个空间，分数全是噪声。
2. **表格被切散** → 表头在一块、数据在另一块，模型看到的是竖线。
3. **chunk 太大** → top-1 命中一篇两万字的文档，模型被淹。
4. **只做向量检索** → 查"工号 XJ-2024-037"这种精确串，dense 完全没用，必须配 BM25。
5. **top_k 拍脑袋** → 应该由评估集的 hit@k 曲线决定，不是越大越好。
6. **没有分数阈值** → 语料里没有的问题也会硬塞三篇文档，模型只能编。
7. **prompt 不约束"不知道就说不知道"** → 幻觉率直接翻倍。
8. **引用编号是模型自己编的** → 必须把编号和资料块绑定，事后校验。
9. **换了 embedding 模型没重建索引、没重标阈值** → 上线当天效果断崖。
10. **没有评估集** → 每次"优化"都靠感觉，越调越差。

---

## 11. 下一步

- 想接真模型：按第 7 节设三个环境变量，跑 `uv run python ask.py "..." --show-prompt`，对照资料和答案，看引用对不对。
- 想接稠密向量：按第 8 节替换 `embedder`，先只做 dense 跑评估，再加 BM25 做混合，看数字变化。
- 想扩语料：先扩到 50~100 篇，重点观察 `hit@3` 什么时候开始掉——那就是该上 rerank 的信号。

代码都能改，参数都在 `rag/config.py` 一个文件里。改完跑一次 `eval.py`，比读十篇文章有用。
