"""
agent.py — GAIA ReAct агент
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from tools import build_all_tools

# ──────────────────────────────────────────────
# SYSTEM PROMPT
# ──────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are a precise research agent solving factual questions.
Your answers will be compared via EXACT STRING MATCH — so format matters critically.

═══════════════════════════════════════════
ANSWER FORMAT RULES (read carefully)
═══════════════════════════════════════════
- Reply with ONLY the final answer — no explanation, no preamble, no punctuation around it.
- Match exactly what the question asks for:
    • "comma-separated list" → use ", " between items
    • "alphabetical order"   → sort A→Z
    • "ascending order"      → sort low→high
    • "plural form"          → use plurals
    • "first name only"      → one word
    • "IOC country code"     → two/three uppercase letters
    • "in USD with two decimal places" → e.g. 9876.54
- Numbers: no thousands separators unless asked; match units asked for.
- If the question is in a non-standard form (e.g. reversed text, encoded),
  decode it first, then answer normally.

═══════════════════════════════════════════
TOOL USAGE STRATEGY
═══════════════════════════════════════════
1. Read the question and any attached file carefully.
2. If a file is attached:
   - .png/.jpg/.jpeg → use describe_image
   - .mp3/.wav/.m4a  → use transcribe_audio_file
   - .py             → use read_python_file, then execute_python
   - .xlsx/.csv      → use read_excel
3. If the question contains a YouTube URL:
   - ALWAYS start with youtube_info (free, no download)
   - If subtitles answer the question → done
   - If question is about speech/narration → youtube_transcribe
   - If question is about visuals/counting → youtube_describe_frames
   - If both needed → youtube_full
4. For factual questions → web_search, then visit_url for full content.
5. For Wikipedia questions → web_search with "site:en.wikipedia.org".
6. Think step by step before answering.
   For multi-hop questions, resolve each dependency in order.

═══════════════════════════════════════════
COMMON MISTAKES TO AVOID
═══════════════════════════════════════════
- Do NOT write "The answer is ...", "Final answer:", or any wrapper.
- Do NOT include units unless the question asks for them.
- Do NOT guess — if unsure, search again.
- Do NOT add trailing punctuation.
- Reversed/encoded text: decode first, then answer.

Reasoning: high
""").strip()


# ──────────────────────────────────────────────
# LLM
# ──────────────────────────────────────────────

def build_llm(model_id: str = "openai/gpt-oss-120b:free") -> ChatOpenAI:
    return ChatOpenAI(
        model=model_id,
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
        max_tokens=4096,
        temperature=0,
    )


# ──────────────────────────────────────────────
# AGENT
# ──────────────────────────────────────────────

class GAIAAgent:
    def __init__(self, model_id: str = "openai/gpt-oss-120b:free") -> None:
        tools = build_all_tools()
        llm = build_llm(model_id)

        self._agent = create_react_agent(
            model=llm,
            tools=tools,
            checkpointer=MemorySaver(),
            prompt=SYSTEM_PROMPT,
        )
        print(f"✅ GAIA Agent ready  (model: {model_id})\n")

    def solve(
        self,
        question: str,
        file_path: str | None = None,
        verbose: bool = False,
    ) -> str:
        """
        Решает один вопрос GAIA.

        Args:
            question:  текст вопроса
            file_path: путь к вложению (опционально)
            verbose:   печатать вызовы инструментов

        Returns:
            Строка-ответ для сабмита.
        """
        # формируем сообщение пользователя
        user_content = question
        if file_path and Path(file_path).exists():
            user_content += f"\n\nAttached file: {file_path}"

        config = {"configurable": {"thread_id": "gaia-single"}}

        result = self._agent.invoke(
            {"messages": [HumanMessage(content=user_content)]},
            config=config,
        )

        if verbose:
            self._print_trace(result["messages"])

        answer: str = result["messages"][-1].content
        return answer.strip()

    @staticmethod
    def _print_trace(messages: list) -> None:
        for msg in messages:
            kind = type(msg).__name__
            if kind == "HumanMessage":
                continue
            elif kind == "AIMessage":
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        print(f"  🔧 {tc['name']}({_fmt_args(tc['args'])})")
                elif msg.content:
                    print(f"  💬 {msg.content[:200]}")
            elif kind == "ToolMessage":
                preview = str(msg.content)[:300]
                print(f"  📥 [{msg.name}] {preview}{'…' if len(str(msg.content)) > 300 else ''}")
        print()


def _fmt_args(args: dict) -> str:
    parts = []
    for k, v in args.items():
        v_str = str(v)
        if len(v_str) > 60:
            v_str = v_str[:57] + "..."
        parts.append(f"{k}={v_str!r}")
    return ", ".join(parts)
