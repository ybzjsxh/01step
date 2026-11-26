"""第 7 步：评估。

RAG 必须能"量化"才好调参。最小评估集只需要三列：
    question / expect（答案里应出现的关键词）/ source（正确文档）
然后分别算：
- 检索命中率 hit@k：正确文档有没有出现在前 k 个结果里
- MRR：正确文档排在第几位（1/rank，越靠前越好）
- 答案命中率：生成结果是否包含预期关键词

改 chunk_size、top_k、是否扩展相邻块，就重跑一次这个脚本，用数字说话。
"""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EvalRow:
    qid: str
    question: str
    source: str
    rank: int | None       # 正确文档首次出现的名次，None = 未命中
    answer_ok: bool
    answer: str


def load_eval_set(path: Path) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def evaluate(rag, eval_path: Path, top_k: int | None = None, verbose: bool = True) -> dict:
    cases = load_eval_set(eval_path)
    k = top_k or rag.config.top_k
    details: list[EvalRow] = []

    for case in cases:
        hits = rag.retrieve(case["question"], top_k=k)
        rank = None
        for i, hit in enumerate(hits, 1):
            if hit.doc_id == case["source"]:
                rank = i
                break
        answer = rag.ask(case["question"], top_k=k)
        ok = any(e in answer.text for e in case.get("expect", []))
        details.append(EvalRow(case["id"], case["question"], case["source"], rank, ok, answer.text))

    n = len(details)
    hit_at_k = sum(1 for d in details if d.rank is not None) / n if n else 0.0
    mrr = sum(1 / d.rank for d in details if d.rank) / n if n else 0.0
    ans_acc = sum(1 for d in details if d.answer_ok) / n if n else 0.0

    if verbose:
        print(f"{'ID':<5}{'检索名次':<10}{'答案命中':<10}问题")
        print("-" * 78)
        for d in details:
            print(f"{d.qid:<5}{str(d.rank or '-'):<10}{'✓' if d.answer_ok else '✗':<10}{d.question}")
        print("-" * 78)
        print(f"样本 {n} 条 | hit@{k} {hit_at_k:.1%} | MRR {mrr:.3f} | 答案命中率 {ans_acc:.1%}")

    return {
        "n": n,
        "k": k,
        "hit_at_k": hit_at_k,
        "mrr": mrr,
        "answer_accuracy": ans_acc,
        "details": [d.__dict__ for d in details],
    }
