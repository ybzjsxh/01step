"""Step 6：生成回答 —— 有 LLM 走 LLM，没 LLM 走抽取式兜底。

运行：
    python3 steps/step6_answer.py
    # 接自己的模型（OpenAI 兼容接口都行）：
    # LLM_API_KEY=sk-xxx LLM_BASE_URL=https://api.deepseek.com/v1 LLM_MODEL=deepseek-chat \
    #   python3 steps/step6_answer.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag import Config, RAG

cfg = Config()
rag = RAG.load()

print(f"LLM 配置：{'已配置' if cfg.llm_api_key else '未配置'}"
      f"（model={cfg.llm_model}, base_url={cfg.llm_base_url}）\n")

answer = rag.ask("满 5 年司龄的员工年假有多少天？")
print(f"mode = {answer.mode}")
print(f"命中文档 = {[h.doc_id for h in answer.hits]}")
print("-" * 60)
print(answer.text)
print("-" * 60)

if answer.mode == "extractive":
    print("""
当前是抽取式模式：它不会"回答问题"，只会把最相关的原文块贴出来。
这样设计是为了让你先确认「检索找对了地方」—— 检索错，再强的模型也答不对。

换成真模型只需要三个环境变量（不用改代码）：
    export LLM_API_KEY=sk-xxx
    export LLM_BASE_URL=https://api.deepseek.com/v1   # 或豆包方舟、vLLM、Ollama
    export LLM_MODEL=deepseek-chat
再跑一次，mode 会变成 llm，答案会变成一句带 [1] 引用的话。
""")
else:
    print("模型返回的 [编号] 对应上面的命中文档，可直接点回原文核对。")
