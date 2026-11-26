"""Step 3：向量化 —— 看清"一句话被变成了什么"。

运行：
    python3 steps/step3_embed.py

这一步是理解检索分数的关键：为什么"年假"能匹配上"休假"，又为什么会误匹配。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag import Config, TfidfEmbedder, chunk_documents, cosine, load_documents, tokenize

cfg = Config()
chunks = chunk_documents(load_documents(cfg.docs_dir), size=cfg.chunk_size, overlap=cfg.chunk_overlap)
embedder = TfidfEmbedder().fit([c.text for c in chunks])

query = "年假有多少天？"
print(f"问题：{query}")
print(f"分词结果（中文单字 + 相邻双字）：{tokenize(query)}\n")

vec = embedder.transform(query)
print(f"词表维度：{len(embedder.idf)}")
print(f"该问题命中 {len(vec)} 个 token，权重最高的 8 个：")
for tok, w in sorted(vec.items(), key=lambda x: -x[1])[:8]:
    print(f"  {tok:<6} tf-idf={w:.3f}  idf={embedder.idf.get(tok, embedder.default_idf):.2f}")

print("\nidf 的含义：越少见的词权重越高。'年假' 比 '公司' 更能区分文档。")

# ---- 用余弦相似度比较三句话 ----
candidates = [
    "员工年假天数按司龄计算：",
    "年假可以拆分成最小 0.5 天使用。",
    "发票抬头必须为公司名称。",
]
print("\n与问题的相似度：")
for text in candidates:
    sim = cosine(vec, embedder.transform(text))
    print(f"  {sim:.3f}  {text}")

print("\n注意：TF-IDF 只认字面。问「休假」不会命中「年假」，问英文不会命中中文。")
print("这就是为什么真实项目要换成 bge-m3 这类稠密向量模型 —— 它懂语义，也跨语言。")
print("换法：把 TfidfEmbedder 换成调 embedding API 的实现，fit/transform 两个方法保持同样的输入输出即可。")
