"""全局配置：所有可调参数集中在这里，方便对照教程调整。"""

import os
from dataclasses import dataclass, field
from pathlib import Path

# 项目根目录（rag/ 的上一级）
ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    # ---- 数据 ----
    docs_dir: Path = ROOT / "data" / "docs"
    index_path: Path = ROOT / "index" / "rag_index.json"
    eval_path: Path = ROOT / "data" / "eval" / "questions.jsonl"

    # ---- 切块 ----
    chunk_size: int = 400      # 每块最大字符数（中文约 250~300 token）
    chunk_overlap: int = 50    # 相邻块重叠字符数，防止答案被切断

    # ---- 检索 ----
    top_k: int = 3             # 最终送给模型的文档数量
    expand_neighbors: bool = True   # 命中块是否带上同文档相邻块
    max_chunks_per_doc: int = 2     # 每个文档最多贡献几块（防止整篇文档挤占 context）
    min_score_ratio: float = 0.3    # 得分低于最高分 30% 的文档直接丢弃
    min_score: float = 0.10         # 绝对分数下限：最高分都低于它就判定"语料里没有"

    # ---- 生成 ----
    max_context_chars: int = 2000    # 拼进 prompt 的资料上限
    temperature: float = 0.0

    # ---- 运行时 ----
    # 只支持 OpenAI 兼容接口：DeepSeek、豆包(方舟)、OpenAI、vLLM、Ollama 都兼容
    llm_base_url: str = field(default_factory=lambda: os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"))
    llm_model: str = field(default_factory=lambda: os.environ.get("LLM_MODEL", "gpt-4o-mini"))
    llm_api_key: str = field(default_factory=lambda: os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY", ""))
    timeout: int = 60
