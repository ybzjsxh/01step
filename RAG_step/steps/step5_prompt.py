"""Step 5：组装 prompt —— RAG 里最"便宜"也最有效的一环。

运行：
    python3 steps/step5_prompt.py

同样的检索结果，prompt 写得好不好，决定了模型是"照资料答"还是"自己编"。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag import Config, RAG, build_messages

cfg = Config()
rag = RAG.load()
question = "满 5 年司龄的员工年假有多少天？"
hits = rag.retrieve(question)
messages = build_messages(question, hits, cfg.max_context_chars)

print("=" * 66)
print("system 消息（行为约束）")
print("=" * 66)
print(messages[0]["content"])
print("\n" + "=" * 66)
print("user 消息（资料 + 问题）")
print("=" * 66)
print(messages[1]["content"])

total = sum(len(m["content"]) for m in messages)
print("\n" + "-" * 66)
print(f"整个 prompt 约 {total} 字符，其中资料部分 {len(messages[1]['content']) - len(question) - 40} 字符左右。")
print("上下文不是越多越好：塞满 10 个块，模型反而更容易被无关内容带偏。")

print("""
system 里五条规则分别在防：
1. 只用资料 → 防止模型拿训练时的旧知识答公司制度；
2. 没有就说不知道 → 防止"合理猜测"，这是幻觉的主要来源；
3. 数字原样给出 → 防止把「15 天」改写成「两周」；
4. 标注 [编号] → 让用户能点回原文核对；
5. 禁止编造编号 → 防止引用一个根本不存在的 [6]。
""")
