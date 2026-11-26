"""把七步串成一条完整链路。

离线（建索引，跑一次）：
    docs → load → chunk → embed → index → rag_index.json

在线（每次问答，毫秒级）：
    question → embed → search → hit + 相邻块 → prompt → LLM → answer

只要 index 里有 embedder 和 vectors，在线阶段完全不需要原始文档。
"""

import time
from pathlib import Path

from .chunker import chunk_documents
from .config import Config
from .embedder import TfidfEmbedder
from .generator import Answer, generate_answer
from .loader import load_documents
from .retriever import Hit, Retriever
from .vector_store import VectorStore


class RAG:
    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.store: VectorStore | None = None

    # ---------- 离线：建索引 ----------
    def build(self, verbose: bool = True) -> dict:
        cfg = self.config
        t0 = time.time()

        docs = load_documents(cfg.docs_dir)
        chunks = chunk_documents(docs, size=cfg.chunk_size, overlap=cfg.chunk_overlap)
        if not chunks:
            raise RuntimeError("没有切出任何块，检查 docs_dir")

        embedder = TfidfEmbedder().fit([c.text for c in chunks])
        vectors = [embedder.transform(c.text) for c in chunks]

        store = VectorStore(embedder=embedder)
        store.add(chunks, vectors)
        store.save(cfg.index_path)
        self.store = store

        stats = {
            "docs": len(docs),
            "chunks": len(chunks),
            "vocab": len(embedder.idf),
            "avg_chunk_chars": round(sum(len(c.text) for c in chunks) / len(chunks), 1),
            "seconds": round(time.time() - t0, 3),
            "index_path": str(cfg.index_path),
        }
        if verbose:
            print(f"[build] 文档 {stats['docs']} 篇 → 块 {stats['chunks']} 个 "
                  f"→ 词表 {stats['vocab']} 维，平均块长 {stats['avg_chunk_chars']} 字符，耗时 {stats['seconds']}s")
            print(f"[build] 索引已写入 {stats['index_path']}")
        return stats

    # ---------- 在线：加载索引 ----------
    @classmethod
    def load(cls, config: Config | None = None, index_path: Path | None = None) -> "RAG":
        cfg = config or Config()
        path = index_path or cfg.index_path
        if not Path(path).exists():
            raise FileNotFoundError(f"索引不存在：{path}\n先运行：python build_index.py")
        obj = cls(cfg)
        obj.store = VectorStore.load(path)
        return obj

    # ---------- 在线：检索 / 问答 ----------
    def retrieve(self, question: str, top_k: int | None = None) -> list[Hit]:
        assert self.store is not None, "请先 build() 或 RAG.load()"
        cfg = self.config
        retriever = Retriever(
            self.store,
            top_k=top_k or cfg.top_k,
            expand_neighbors=cfg.expand_neighbors,
            max_chunks_per_doc=cfg.max_chunks_per_doc,
            min_score_ratio=cfg.min_score_ratio,
            min_score=cfg.min_score,
        )
        return retriever.retrieve(question)

    def ask(self, question: str, top_k: int | None = None) -> Answer:
        hits = self.retrieve(question, top_k=top_k)
        return generate_answer(question, hits, self.config)
