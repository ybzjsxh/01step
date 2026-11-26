"""最小 RAG 实现：七个模块，一条链路。

    from rag import RAG
    rag = RAG.load()
    ans = rag.ask("满 5 年年假多少天？")
"""

from .chunker import Chunk, chunk_document, chunk_documents
from .config import Config
from .embedder import TfidfEmbedder, cosine, tokenize
from .evaluate import evaluate, load_eval_set
from .generator import Answer, build_messages, generate_answer
from .loader import Document, load_documents
from .pipeline import RAG
from .retriever import Hit, Retriever
from .vector_store import VectorStore

__all__ = [
    "Config", "RAG", "Document", "load_documents",
    "Chunk", "chunk_documents", "chunk_document",
    "TfidfEmbedder", "tokenize", "cosine",
    "VectorStore", "Retriever", "Hit",
    "Answer", "build_messages", "generate_answer",
    "evaluate", "load_eval_set",
]
