"""第 4 步：建索引 + 检索（vector store）。

最小索引就是「块列表 + 块向量列表」，检索时逐个算余弦相似度。
这叫暴力检索（flat / brute force）：10 万块以内都够快，也是向量库的底层实现。
真实项目换成 FAISS / Milvus / pgvector 时，把 add / search 这两个方法换掉即可。

索引落盘成一个 JSON：
    {"embedder": ..., "chunks": [...], "vectors": [{token: weight}, ...]}
这样"离线建索引"和"在线问答"就彻底解耦了。
"""

import json
import time
from pathlib import Path

from .chunker import Chunk
from .embedder import TfidfEmbedder, cosine


class VectorStore:
    def __init__(self, embedder: TfidfEmbedder | None = None):
        self.embedder = embedder
        self.chunks: list[Chunk] = []
        self.vectors: list[dict[str, float]] = []

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, chunks: list[Chunk], vectors: list[dict[str, float]]) -> None:
        assert len(chunks) == len(vectors), "块和向量数量必须一致"
        self.chunks.extend(chunks)
        self.vectors.extend(vectors)

    def search(self, query: str, top_k: int = 4) -> list[tuple[float, Chunk]]:
        assert self.embedder is not None, "索引里没有 embedder，请先 build 索引"
        qvec = self.embedder.transform(query)
        if not qvec:
            return []
        scored = [(cosine(qvec, vec), chunk) for vec, chunk in zip(self.vectors, self.chunks)]
        scored = [x for x in scored if x[0] > 0]
        scored.sort(key=lambda x: (-x[0], x[1].chunk_id))
        return scored[:top_k]

    # ---- 持久化 ----
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "embedder": self.embedder.to_dict() if self.embedder else None,
            "chunks": [c.to_dict() for c in self.chunks],
            "vectors": self.vectors,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "VectorStore":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        store = cls(embedder=TfidfEmbedder.from_dict(payload["embedder"]))
        store.chunks = [Chunk.from_dict(d) for d in payload["chunks"]]
        store.vectors = payload["vectors"]
        return store
