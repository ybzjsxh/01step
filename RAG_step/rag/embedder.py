"""第 3 步：把文本变成向量（embedding）。

真实项目这里会调用 bge-m3 / text-embedding-3 这类模型，得到 1024 维稠密向量。
最小实现为了零依赖、可解释，用的是 TF-IDF 稀疏向量：

    文本 → 分词 → 词频(tf) → 逆文档频率(idf) → 加权 → L2 归一化 → 向量

中文不分词也能用：把每个汉字当 unigram，再把相邻两个字当 bigram。
"年假"这个词会变成 token「年」「假」「年假」，既保留字面又带上词组信息。

接口刻意设计成 fit(texts) / transform(text) -> dict[token, weight]，
换成稠密模型时只要把 token 换成维度下标、dict 换成 list[float]，其余代码不用动。
"""

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]+|[a-zA-Z0-9_]+")

# 只保留单字停用词（多字词不会成为 token，写了也不生效）
# 注意：不把"不/上/下/中/内/外/前/后"放进来 —— 它们能组成有意义的词（不可、内部、上线）
STOPWORDS = set(
    "的 了 是 在 有 和 与 及 或 我 你 他 她 它 们 这 那 个 吗 呢 啊 吧 就 都 也 很 要 会 呀 哦 么 为 对 以 等 被 把 从 到 请".split()
    # 疑问词在语料里几乎不出现，idf 反而最高，会把检索带偏，必须当停用词
+ "多少 什么 怎么 如何 哪些 哪个 是否".split()
) | {"the", "a", "an", "is", "are", "of", "to", "and", "in", "on", "for", "with", "what", "how"}


def tokenize(text: str) -> list[str]:
    """中文：单字 + 相邻双字（含停用词的跨词双字直接丢弃）；英文/数字：整词小写。"""
    text = text.lower()
    tokens: list[str] = []
    for m in _TOKEN_RE.finditer(text):
        s = m.group()
        if "\u4e00" <= s[0] <= "\u9fff":
            tokens.extend(s)                                          # unigram
            # "假有""有多"这种跨词 bigram 是噪声：任一端是停用词就不生成
            tokens.extend(
                s[i:i + 2] for i in range(len(s) - 1)
                if s[i] not in STOPWORDS and s[i + 1] not in STOPWORDS
            )
        else:
            tokens.append(s)
    return [t for t in tokens if t not in STOPWORDS]


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    """两个已 L2 归一化的稀疏向量，余弦相似度就是点积。"""
    if len(a) > len(b):
        a, b = b, a
    return sum(w * b.get(t, 0.0) for t, w in a.items())


class TfidfEmbedder:
    def __init__(self, idf: dict[str, float] | None = None, n_docs: int = 0):
        self.idf: dict[str, float] = idf or {}
        self.n_docs = n_docs
        self.default_idf = 1.0

    def fit(self, texts: list[str]) -> "TfidfEmbedder":
        df: Counter[str] = Counter()
        for text in texts:
            df.update(set(tokenize(text)))
        n = len(texts)
        self.n_docs = n
        self.idf = {tok: math.log((1 + n) / (1 + c)) + 1.0 for tok, c in df.items()}
        # 查询里出现但语料里没见过的词，按最高 idf 处理，避免整条 query 变空
        self.default_idf = math.log(1 + n) + 1.0
        return self

    def transform(self, text: str) -> dict[str, float]:
        counts = Counter(tokenize(text))
        vec: dict[str, float] = {}
        for tok, c in counts.items():
            idf = self.idf.get(tok, self.default_idf)
            vec[tok] = (1.0 + math.log(c)) * idf
        norm = math.sqrt(sum(w * w for w in vec.values()))
        if not norm:
            return {}
        return {t: w / norm for t, w in vec.items()}

    # ---- 持久化 ----
    def to_dict(self) -> dict:
        return {"idf": self.idf, "n_docs": self.n_docs, "default_idf": self.default_idf}

    @classmethod
    def from_dict(cls, d: dict) -> "TfidfEmbedder":
        obj = cls(idf=d["idf"], n_docs=d["n_docs"])
        obj.default_idf = d.get("default_idf", 1.0)
        return obj
