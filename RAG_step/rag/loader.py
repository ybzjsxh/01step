"""第 1 步：加载文档。

最小实现只处理 .md / .txt 纯文本。PDF、Word 在真实项目里需要额外解析器，
但解析结果最终都要变成"字符串 + 元数据"，所以这一步的结构不用改。
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Document:
    doc_id: str            # 唯一标识，这里直接用文件名
    title: str             # 展示用标题
    text: str              # 全文
    path: str              # 来源路径
    metadata: dict = field(default_factory=dict)


def _extract_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def load_documents(docs_dir: Path) -> list[Document]:
    """按文件名排序读取目录下所有 .md/.txt。"""
    docs_dir = Path(docs_dir)
    if not docs_dir.exists():
        raise FileNotFoundError(f"文档目录不存在：{docs_dir}")

    files = sorted(
        [p for p in docs_dir.rglob("*") if p.suffix.lower() in {".md", ".txt"} and p.is_file()],
        key=lambda p: p.name,
    )
    docs = []
    for path in files:
        text = path.read_text(encoding="utf-8").replace("\r\n", "\n").strip()
        if not text:
            continue
        docs.append(
            Document(
                doc_id=path.name,
                title=_extract_title(text, path.stem),
                text=text,
                path=str(path),
            )
        )
    return docs
