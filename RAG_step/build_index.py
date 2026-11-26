#!/usr/bin/env python3
"""离线建索引：一条命令把 docs 变成 index/rag_index.json。

    python build_index.py
    python build_index.py --chunk-size 300 --overlap 60
"""

import argparse
from pathlib import Path

from rag import Config, RAG


def main() -> None:
    ap = argparse.ArgumentParser(description="构建 RAG 索引")
    ap.add_argument("--docs", default=None, help="文档目录，默认 data/docs")
    ap.add_argument("--chunk-size", type=int, default=None, help="每块最大字符数")
    ap.add_argument("--overlap", type=int, default=None, help="相邻块重叠字符数")
    ap.add_argument("--index", default=None, help="索引输出路径")
    args = ap.parse_args()

    cfg = Config()
    if args.docs:
        cfg.docs_dir = Path(args.docs).expanduser().resolve()
    if args.chunk_size:
        cfg.chunk_size = args.chunk_size
    if args.overlap is not None:
        cfg.chunk_overlap = args.overlap
    if args.index:
        cfg.index_path = Path(args.index).expanduser().resolve()

    RAG(cfg).build()


if __name__ == "__main__":
    main()
