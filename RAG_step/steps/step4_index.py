"""Step 4：建索引与检索 —— 看见分数、阈值和相邻块扩展的效果。

运行：
    python3 steps/step4_index.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag import Config, RAG

cfg = Config()
rag = RAG(cfg)
rag.build(verbose=False)

print(f"索引：{len(rag.store)} 块，{len(rag.store.embedder.idf)} 维词表\n")

queries = [
    "满 5 年司龄的年假有多少天？",
    "报销发票抬头写什么？",
    "公司去年的营收是多少？",      # 语料里没有
]

for q in queries:
    print("=" * 66)
    print(f"问题：{q}")
    print("-" * 66)
    print("原始召回（不过滤阈值）：")
    for score, chunk in rag.store.search(q, 5):
        print(f"  {score:.3f}  {chunk.chunk_id:<28} {chunk.text[:26].replace(chr(10), ' ')}…")

    hits = rag.retrieve(q)
    print(f"过滤 + 扩展后进入 prompt 的文档：{len(hits)} 篇")
    for i, hit in enumerate(hits, 1):
        print(f"  [{i}] {hit.doc_id}  score={hit.score:.3f}  块={hit.chunk_ids}")
    print()

print("三个观察：")
print("1. 前两个问题的最高分在 0.2 以上，第三个只有 0.05 左右 —— 绝对阈值 0.10 把它挡住了；")
print("2. 相对阈值（最高分的 30%）会顺带清掉同一次检索里的低分噪声；")
print("3. 命中的块会被扩展成「命中块 + 相邻块」，所以块 id 是多个。")
print("\n想亲手调：改 rag/config.py 里的 top_k / min_score / expand_neighbors，重跑本脚本。")
