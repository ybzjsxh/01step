"""第 4 步（续）：检索器。

在向量检索之上做了三件小事，这三件事在实际项目里非常关键：

1. **相对分数过滤**：只保留得分不低于最高分 35% 的文档，
   避免 0.05 分的无关文档混进 context 干扰模型。
2. **相邻块扩展**：命中的块前后各带一块，拼成完整上下文。
   答案常常紧挨着命中块却被切块边界切走（「住宿上限」在块尾、具体金额在下一块开头）。
3. **每文档块数上限**：扩展会"传染"（命中多块时整篇文档都被带进来），
   必须限制每个文档最多贡献几块，否则 context 很快被一篇长文档占满。

通用原则：**检索单元要小（保精度），送模型的上下文要够（保完整），但总量要可控。**
"""

from dataclasses import dataclass, field

from .chunker import Chunk
from .vector_store import VectorStore


@dataclass
class Hit:
    score: float
    chunk: Chunk                  # 该文档中得分最高的块
    text: str                     # 实际送给模型的文本（可能由多块拼成）
    chunk_ids: list[str] = field(default_factory=list)

    @property
    def doc_id(self) -> str:
        return self.chunk.doc_id


class Retriever:
    def __init__(
        self,
        store: VectorStore,
        top_k: int = 3,
        expand_neighbors: bool = True,
        neighbor_window: int = 1,
        max_chunks_per_doc: int = 2,
        min_score_ratio: float = 0.3,
        min_score: float = 0.10,
    ):
        self.store = store
        self.top_k = top_k
        self.expand_neighbors = expand_neighbors
        self.neighbor_window = neighbor_window
        self.max_chunks_per_doc = max_chunks_per_doc
        self.min_score_ratio = min_score_ratio
        self.min_score = min_score
        self._index: dict[str, dict[int, Chunk]] | None = None

    def _doc_index(self) -> dict[str, dict[int, Chunk]]:
        if self._index is None:
            self._index = {}
            for c in self.store.chunks:
                self._index.setdefault(c.doc_id, {})[c.index] = c
        return self._index

    def retrieve(self, query: str, top_k: int | None = None) -> list[Hit]:
        k = top_k or self.top_k
        # 多取一些候选，再做过滤和裁剪
        raw = self.store.search(query, max(k * 3, k))
        if not raw:
            return []

        floor = max(raw[0][0] * self.min_score_ratio, self.min_score)
        raw = [(s, c) for s, c in raw if s >= floor]
        if not raw:
            return []

        if not self.expand_neighbors:
            return [Hit(score=s, chunk=c, text=c.text, chunk_ids=[c.chunk_id]) for s, c in raw][:k]

        doc_index = self._doc_index()
        groups: dict[str, dict] = {}
        order: list[str] = []
        for score, chunk in raw:
            if chunk.doc_id not in groups:
                groups[chunk.doc_id] = {"best": score, "best_chunk": chunk, "hit_idx": set()}
                order.append(chunk.doc_id)
            g = groups[chunk.doc_id]
            if score > g["best"]:
                g["best"] = score
                g["best_chunk"] = chunk
            g["hit_idx"].add(chunk.index)

        hits: list[Hit] = []
        for doc_id in order:
            g = groups[doc_id]
            hit_idx = set(g["hit_idx"])
            center = g["best_chunk"].index

            # 以最强命中块为中心向两侧扩，保证上下文是连续的一段，
            # 而不是 0 号和 2 号块拼在一起、中间缺一块。
            keep = [center]
            radius = 1
            max_n = max(self.max_chunks_per_doc, 1)
            while len(keep) < max_n and radius <= self.neighbor_window:
                for i in (center - radius, center + radius):
                    if len(keep) >= max_n:
                        break
                    if i in doc_index[doc_id] and i not in keep:
                        keep.append(i)
                radius += 1

            ordered = [doc_index[doc_id][i] for i in sorted(keep)]
            hits.append(
                Hit(
                    score=g["best"],
                    chunk=g["best_chunk"],
                    text="\n".join(c.text for c in ordered),
                    chunk_ids=[c.chunk_id for c in ordered],
                )
            )

        hits.sort(key=lambda h: -h.score)
        return hits[:k]
