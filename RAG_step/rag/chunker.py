"""第 2 步：切块（chunking）。

切块是 RAG 里最容易被低估的一步：
- 块太大：检索精度下降，噪声多，浪费 context；
- 块太小：语义不完整，答案被切断；
- 没有重叠：关键句正好跨在两块边界时会丢。

这里的策略是「按行/段落切 + 定长上限 + 句边界回退重叠」，并额外做三件保护：
1. **整张 markdown 表格不可分割**：表格拆开后就只剩一堆竖线，表头和数据行分家；
2. **标题不孤立在块尾**：块尾如果是「## 差旅住宿标准」这类标题，把它挪到下一块开头；
3. **重叠从标点后开始**：不会把半句话复制到下一块。

真实项目还要处理代码块、列表、PDF 分页、页码元数据。原则一样：
**先保住结构，再谈长度。**
"""

import re
from dataclasses import dataclass

from .loader import Document


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    index: int
    text: str

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "title": self.title,
            "index": self.index,
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(**d)


def split_units(text: str) -> list[str]:
    """把文档拆成不可再分的最小单元：一行文本，或一整张 markdown 表格。"""
    units: list[str] = []
    table: list[str] = []

    def flush_table():
        if table:
            units.append("\n".join(table) + "\n")
            table.clear()

    for line in text.splitlines():
        if not line.strip():
            continue
        if line.lstrip().startswith("|"):
            table.append(line.strip())
            continue
        flush_table()
        units.append(line.strip() + "\n")
    flush_table()
    return units


def _overlap_tail(text: str, n: int) -> str:
    """取块尾不超过 n 个字符作为下一块的开头，从标点之后开始；遇到标题行直接截掉。"""
    if n <= 0 or not text:
        return ""
    tail = text[-n:]
    m = re.search(r"[。！？；\n]", tail)
    tail = tail[m.end():].strip() if m else tail.strip()
    # 不要把标题带进重叠区
    cut = tail.find("\n#")
    if cut != -1:
        tail = tail[:cut].strip()
    if not tail or tail.startswith("#"):
        return ""
    return tail


def chunk_document(doc: Document, size: int = 400, overlap: int = 50) -> list[Chunk]:
    chunks: list[Chunk] = []
    buf = ""
    carry = ""        # 从上一块尾部挪过来的标题
    last_text = ""    # 上一次真正落盘的内容，用来算重叠

    def emit(text: str):
        idx = len(chunks)
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}::{idx:03d}",
                doc_id=doc.doc_id,
                title=doc.title,
                index=idx,
                text=text,
            )
        )

    def flush():
        nonlocal buf, carry, last_text
        text = buf.strip()
        if not text:
            buf = ""
            return
        parts = text.split("\n")
        # 块尾如果是标题，挪到下一块开头，避免"标题孤零零结尾"
        if len(parts) > 1 and parts[-1].lstrip().startswith("#"):
            carry = parts.pop().strip() + "\n"
            text = "\n".join(parts).strip()
        if text:
            emit(text)
            last_text = text
        buf = ""

    for unit in split_units(doc.text):
        # 极端情况：一整段没有换行且超长，硬切
        pieces = [unit[i:i + size] for i in range(0, len(unit), size)] if len(unit) > size else [unit]
        for piece in pieces:
            if buf.strip() and len(buf) + len(piece) > size:
                flush()
                buf = carry if carry else _overlap_tail(last_text, overlap)
                carry = ""
                if len(buf) + len(piece) > size:   # 重叠/标题反而挤爆，放弃
                    buf = ""
            buf += piece
        if len(buf) > size * 2:                   # 安全阀
            flush()
            buf = _overlap_tail(last_text, overlap)
    flush()

    if carry.strip():                             # 文档以标题结尾的极端情况
        emit(carry.strip())
    return chunks


def chunk_documents(docs: list[Document], size: int = 400, overlap: int = 50) -> list[Chunk]:
    out: list[Chunk] = []
    for doc in docs:
        out.extend(chunk_document(doc, size=size, overlap=overlap))
    return out
