"""Step 1：加载文档 —— 先把"资料"变成结构化的 Document 列表。

运行：
    python3 steps/step1_load.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag import Config, load_documents

cfg = Config()
docs = load_documents(cfg.docs_dir)

print(f"目录：{cfg.docs_dir}")
print(f"共加载 {len(docs)} 篇文档\n")
for d in docs:
    print(f"- doc_id = {d.doc_id}")
    print(f"  title  = {d.title}")
    print(f"  字数   = {len(d.text)}")
    print(f"  开头   = {d.text[:60].replace(chr(10), ' / ')}…")
print("\n要点：doc_id 必须是稳定唯一标识（这里用文件名），后面所有引用都靠它回溯。")
