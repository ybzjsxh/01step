"""第 5~6 步：组装 prompt + 生成回答。

两个关键点：
1. **prompt 是 RAG 质量的一半**。必须约束模型「只依据资料、不足就说不知道、标注引用」，
   否则模型会拿自己的先验知识补全，产生"看起来对但和公司制度不符"的幻觉。
2. **生成层可插拔**。配了 LLM_API_KEY 就走 OpenAI 兼容接口（DeepSeek / 豆包 / vLLM / Ollama 都行），
   没配就用抽取式兜底 —— 直接把最相关的原文句子摘出来，方便你先验证检索是否准确。

先保证"检索对"，再谈"生成好"。这是 RAG 工程的正确顺序。
"""

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from .config import Config
from .retriever import Hit

SYSTEM_PROMPT = """你是企业制度问答助手，只能依据用户提供的【资料】回答。
规则：
1. 只使用【资料】中的信息，不得使用资料之外的知识；
2. 资料中没有答案时，直接回答「根据现有资料无法回答该问题」，不要猜测；
3. 回答要简洁具体，涉及数字、天数、金额时原样给出；
4. 每个结论后用 [编号] 标注来源，如 [1]、[1][3]；
5. 不得编造资料中不存在的编号，不得把资料外的内容写进回答。"""


@dataclass
class Answer:
    question: str
    text: str
    hits: list[Hit]
    messages: list[dict] = field(default_factory=list)
    mode: str = "extractive"     # llm / extractive


def build_context(hits: list[Hit], max_chars: int) -> str:
    """把命中的块拼成带编号的资料区，并控制总长度。"""
    blocks: list[str] = []
    used = 0
    for i, hit in enumerate(hits, 1):
        text = hit.text.strip()
        remain = max_chars - used
        if remain < 100:
            break
        if len(text) > remain:
            text = text[:remain].rstrip() + "…"
        blocks.append(f"[{i}] 来源：{hit.doc_id}（{hit.chunk.title}）\n{text}")
        used += len(text)
    return "\n\n".join(blocks)


def build_messages(question: str, hits: list[Hit], max_context_chars: int = 2000) -> list[dict]:
    context = build_context(hits, max_context_chars)
    user = f"【资料】\n{context}\n\n【问题】\n{question}\n\n请依据资料作答，并在结论后标注来源编号。"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


# ---------------- LLM 调用（OpenAI 兼容） -----------------------------
def call_llm(messages: list[dict], config: Config) -> str:
    url = config.llm_base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": config.llm_model,
        "messages": messages,
        "temperature": config.temperature,
        "stream": False,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.llm_api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=config.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"LLM 接口返回 {e.code}：{e.read().decode('utf-8', 'ignore')[:300]}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"LLM 接口连接失败：{e.reason}") from e
    return data["choices"][0]["message"]["content"].strip()


# ---------------- 抽取式兜底 -----------------------------
def extractive_answer(question: str, hits: list[Hit], max_docs: int = 2, max_chars: int = 500) -> str:
    """没配 LLM 时：直接给出得分最高那块原文，用来验证"检索是否真的找到了答案所在的段落"。

    注意这里只取 hit.chunk（单块），不取扩展后的 hit.text：
    扩展块是给 LLM 补上下文用的，做检索校验时看单块更容易判断精度。
    """
    parts = []
    for i, hit in enumerate(hits[:max_docs], 1):
        text = hit.chunk.text.strip()
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "…"
        parts.append(f"[{i}] {hit.doc_id}（{hit.chunk.chunk_id}）：\n{text}")

    if not parts:
        return "根据现有资料无法回答该问题。（未检索到相关内容）"
    return "（未配置大模型，以下为检索原文摘录，用于验证检索质量）\n" + "\n\n".join(parts)


def generate_answer(question: str, hits: list[Hit], config: Config | None = None) -> Answer:
    config = config or Config()
    messages = build_messages(question, hits, config.max_context_chars)

    if not hits:
        return Answer(question=question, text="根据现有资料无法回答该问题。（未检索到相关内容）", hits=hits, messages=messages)

    if config.llm_api_key:
        try:
            text = call_llm(messages, config)
            return Answer(question=question, text=text, hits=hits, messages=messages, mode="llm")
        except RuntimeError as e:
            fallback = extractive_answer(question, hits)
            return Answer(
                question=question,
                text=f"{fallback}\n\n（调用大模型失败：{e}）",
                hits=hits,
                messages=messages,
            )

    return Answer(question=question, text=extractive_answer(question, hits), hits=hits, messages=messages)
