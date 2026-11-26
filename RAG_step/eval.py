#!/usr/bin/env python3
"""跑评估集，输出 hit@k / MRR / 答案命中率。

    python eval.py
    python eval.py --k 3 --chunk-size 300
"""

import argparse
import json
from pathlib import Path

from rag import Config, RAG, evaluate
from rag.config import ROOT


def main() -> None:
    ap = argparse.ArgumentParser(description="RAG 检索与答案评估")
    ap.add_argument("-k", "--top-k", type=int, default=None)
    ap.add_argument("--chunk-size", type=int, default=None, help="临时用新的切块参数重建索引")
    ap.add_argument("--no-expand", action="store_true", help="关闭相邻块扩展，对比效果")
    ap.add_argument("--json", action="store_true", help="只输出 JSON 结果")
    args = ap.parse_args()

    cfg = Config()
    if args.chunk_size:
        cfg.chunk_size = args.chunk_size
        cfg.index_path = ROOT / "index" / f"rag_index_{args.chunk_size}.json"
    if args.no_expand:
        cfg.expand_neighbors = False

    rag = RAG(cfg)
    if not Path(cfg.index_path).exists():
        rag.build()
    else:
        rag = RAG.load(cfg)

    result = evaluate(rag, cfg.eval_path, top_k=args.top_k, verbose=not args.json)
    if args.json:
        print(json.dumps({k: v for k, v in result.items() if k != "details"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
