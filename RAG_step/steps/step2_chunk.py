"""Step 2：切块 —— 看见"结构保护"到底保护了什么。

运行：
    python3 steps/step2_chunk.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag import Config, chunk_documents, load_documents

cfg = Config()
docs = load_documents(cfg.docs_dir)
chunks = chunk_documents(docs, size=cfg.chunk_size, overlap=cfg.chunk_overlap)

print(f"参数：chunk_size={cfg.chunk_size} 字符，overlap={cfg.chunk_overlap} 字符")
print(f"{len(docs)} 篇文档 → {len(chunks)} 个块，平均 {sum(len(c.text) for c in chunks)/len(chunks):.0f} 字符\n")

# 按文档统计块数
per_doc: dict[str, int] = {}
for c in chunks:
    per_doc[c.doc_id] = per_doc.get(c.doc_id, 0) + 1
for doc, n in per_doc.items():
    print(f"  {doc}: {n} 块")

# 挑一个含表格的块，看看表格有没有被拆散
target = next(c for c in chunks if c.doc_id.startswith("02") and "|" in c.text)
print("\n" + "=" * 60)
print(f"示例块 {target.chunk_id}（{len(target.text)} 字符）")
print("=" * 60)
print(target.text)

print("\n观察三点：")
print("1. 表格整张在同一块里，表头和数据行没有分家；")
print("2. 块以「## 差旅住宿标准」开头，不会出现标题孤零零留在上一块结尾；")
print("3. 相邻块有一段重叠（下一个块的开头会重复本块最后一两句），防止答案正好卡在边界。")

nxt = next((c for c in chunks if c.doc_id == target.doc_id and c.index == target.index + 1), None)
if nxt:
    print(f"\n下一块 {nxt.chunk_id} 的开头 60 字符：{nxt.text[:60]!r}")
