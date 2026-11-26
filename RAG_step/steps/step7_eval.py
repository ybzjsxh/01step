"""Step 7：评估 —— 用数字判断"改得好不好"。

运行：
    python3 steps/step7_eval.py
    python3 steps/step7_eval.py --k 1        # 只看第一名对不对
    python3 steps/step7_eval.py --no-expand  # 关掉相邻块扩展做对照
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag import Config, RAG, evaluate

ap = argparse.ArgumentParser()
ap.add_argument("-k", "--k", type=int, default=None, help="只看前 k 个结果")
ap.add_argument("--no-expand", action="store_true", help="关闭相邻块扩展")
args = ap.parse_args()

cfg = Config()
if args.no_expand:
    cfg.expand_neighbors = False
rag = RAG.load(cfg)

print("指标说明：")
print("  hit@k  正确文档有没有出现在前 k 个结果里（召回）")
print("  MRR    正确文档排第几位，1/rank 取平均（排序质量）")
print("  answer 生成结果是否包含预期关键词（端到端）\n")

evaluate(rag, cfg.eval_path, top_k=args.k)

print("""
怎么用这套评估：
- 改 chunk_size / top_k / 阈值，重跑，看三个数字有没有涨；
- 数字涨了但答案变差 → 问题在生成层（prompt 或模型）；
- 数字不涨 → 问题在检索层（切块、embedding、query 表述）；
- 评估集要覆盖「能答的」和「答不了的」两类，否则调出来的阈值会偏。
""")
