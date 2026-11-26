#!/usr/bin/env python3
"""在线问答。

    python ask.py "满 5 年司龄的年假有多少天？"
    python ask.py                       # 进入交互模式
    python ask.py "..." --show-prompt   # 打印真正发给模型的 prompt
"""

import argparse

from rag import RAG


def show(answer, show_prompt: bool = False) -> None:
    print("\n" + "=" * 60)
    print(f"问题：{answer.question}")
    print("=" * 60)
    print(f"模式：{answer.mode}   命中 {len(answer.hits)} 个文档")
    for i, hit in enumerate(answer.hits, 1):
        print(f"  [{i}] score={hit.score:.3f}  {hit.doc_id}  块={','.join(hit.chunk_ids)}")
    print("-" * 60)
    if show_prompt and answer.messages:
        print("【发给模型的 prompt】")
        print(answer.messages[1]["content"])
        print("-" * 60)
    print("【回答】")
    print(answer.text)
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description="RAG 问答")
    ap.add_argument("question", nargs="*", help="问题；留空进入交互模式")
    ap.add_argument("-k", "--top-k", type=int, default=None)
    ap.add_argument("--show-prompt", action="store_true")
    args = ap.parse_args()

    rag = RAG.load()

    if args.question:
        show(rag.ask(" ".join(args.question), top_k=args.top_k), args.show_prompt)
        return

    print("RAG 已就绪，输入问题回车；输入 exit 退出。")
    while True:
        try:
            q = input("\n问题> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if q.lower() in {"exit", "quit", "q"}:
            break
        if not q:
            continue
        show(rag.ask(q, top_k=args.top_k), args.show_prompt)


if __name__ == "__main__":
    main()
