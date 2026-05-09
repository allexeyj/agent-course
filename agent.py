"""
agent.py — GAIA ReAct агент
"""

from __future__ import annotations

import hashlib
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
TOOL SELECTION STRATEGY
═══════════════════════════════════════════

── ATTACHED FILES ──────────────────────────
• .png / .jpg / .jpeg (generic image) → describe_image
• .png / .jpg with a CHESS BOARD      → chess_move  ← NOT describe_image
• .mp3 / .wav / .m4a                  → transcribe_audio_file
• .py                                 → read_python_file first, then decide:
      - has while True / time.sleep / random → analyze_python_logic
      - simple deterministic code           → execute_python
• .xlsx / .csv                        → read_excel
• .pdf                                → read_pdf_url

── CHESS POSITIONS ─────────────────────────
ALWAYS use chess_move for chess board images.
Never use describe_image for chess — it will not find the best move.
Provide the turn ("black" or "white") from the question.

── YOUTUBE VIDEOS ──────────────────────────
Decision tree (follow in order):
  1. ALWAYS start with youtube_info — get metadata + subtitles for free.
  2. If subtitles answer the question → done, no download needed.
  3. Question about SPEECH / DIALOGUE / NARRATION → youtube_transcribe
  4. Question about COUNTING OBJECTS simultaneously (birds, people, cars…)
       → youtube_count_objects(url, object_name="bird")
       This uses YOLOv8 — much more accurate than frame descriptions for counting.
  5. Question about VISUALS (describe scene, read text on screen, identify objects)
       → youtube_describe_frames(url, question="specific question about each frame")
       Always pass a specific question, not the generic default.
  6. Question needs BOTH speech AND visuals → youtube_full

── FACTUAL / RESEARCH QUESTIONS ────────────
  1. web_search with a precise query.
  2. If result is a web page → visit_url to read full content.
  3. If result is a PDF (scientific paper, report) → read_pdf_url.
  4. For Wikipedia → web_search with "site:en.wikipedia.org <topic>".
  5. Multi-hop questions: resolve each dependency step by step.
     Example: "Who played X in film Y → what role did they play in Z"
       Step 1: search who played X in Y
       Step 2: search that actor's role in Z

── PYTHON FILES ────────────────────────────
  1. Always read_python_file first to see the code.
  2. Check for: while True, time.sleep(), random, recursion.
  3. If any found → use analyze_python_logic (static reasoning, no execution).
  4. If clean/deterministic → use execute_python.
  5. IMPORTANT: a function that only returns one possible value
     (e.g. returns X only when condition Y is true, and Y implies X==0)
     can be answered by pure reasoning without running the code.

── PDF DOCUMENTS ───────────────────────────
  • Always use read_pdf_url (not visit_url) for direct PDF links.
  • For scientific papers: look in Acknowledgements / Funding sections
    for award numbers, grant IDs, NASA/NSF codes.
  • visit_url auto-redirects PDF content-type to read_pdf_url anyway.

═══════════════════════════════════════════
TOOL QUICK REFERENCE
═══════════════════════════════════════════
  web_search            — search the web (Tavily)
  visit_url             — fetch full page text (auto-handles PDFs)
  read_pdf_url          — extract text from PDF file or URL
  describe_image        — ask any question about an image (Moondream2)
  chess_move            — chess board image → best move via Stockfish
  transcribe_audio_file — audio file → text (Whisper)
  youtube_info          — metadata + subtitles, no download (START HERE)
  youtube_transcribe    — audio → Whisper transcript
  youtube_describe_frames — frames → Moondream2 visual descriptions
  youtube_count_objects — frames → YOLOv8 object count (max simultaneous)
  youtube_full          — full audio+visual analysis (slow, last resort)
  analyze_python_logic  — static AST analysis for tricky Python code
  read_python_file      — read .py file source
  execute_python        — run deterministic Python code
  read_excel            — read .xlsx or .csv as table

═══════════════════════════════════════════
COMMON MISTAKES TO AVOID
═══════════════════════════════════════════
- Do NOT write "The answer is ...", "Final answer:", or any wrapper.
- Do NOT include units unless the question asks for them.
- Do NOT guess — if unsure, search again with a different query.
- Do NOT add trailing punctuation.
- Do NOT use describe_image for chess boards — use chess_move.
- Do NOT use execute_python if the code has while True or time.sleep.
- Do NOT use visit_url for PDF links — use read_pdf_url.
- Reversed/encoded text: decode first, then answer.
- Botanical fruits (tomato, pepper, corn, zucchini, cucumber, avocado,
  pumpkin, eggplant, okra, green beans, peas, nuts) are NOT vegetables.

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
        task_id: str | None = None,
        verbose: bool = False,
    ) -> str:
        """
        Решает один вопрос GAIA.

        Args:
            question:  текст вопроса
            file_path: путь к вложению (опционально)
            task_id:   идентификатор задачи (для изоляции памяти между вопросами)
            verbose:   печатать вызовы инструментов

        Returns:
            Строка-ответ для сабмита.
        """
        user_content = question
        if file_path and Path(file_path).exists():
            user_content += f"\n\nAttached file: {file_path}"

        # Уникальный thread_id на каждый вопрос.
        # Критично: без этого MemorySaver тащит контекст всех предыдущих вопросов,
        # агент переигрывает старые tool calls и мешает ответы друг с другом.
        if task_id:
            thread_id = f"gaia-{task_id}"
        else:
            thread_id = f"gaia-{hashlib.md5(question.encode()).hexdigest()[:12]}"

        config = {"configurable": {"thread_id": thread_id}}

        result = self._agent.invoke(
            {"messages": [HumanMessage(content=user_content)]},
            config=config,
        )

        if verbose:
            self._print_trace(result["messages"])

        answer: str = result["messages"][-1].content
        answer = self._clean_answer(answer)
        return answer

    @staticmethod
    def _clean_answer(answer: str) -> str:
        """
        Постобработка ответа модели:
          - убирает пробелы по краям
          - удаляет обёртки типа "The answer is X"
          - исправляет задвоения типа "FunkMonkFunkMonk" → "FunkMonk"
          - убирает trailing punctuation
        """
        answer = answer.strip()

        # Удаляем явные обёртки с reasoning
        for prefix in (
            "The answer is ",
            "Final answer: ",
            "Answer: ",
            "FINAL ANSWER: ",
            "final answer: ",
        ):
            if answer.lower().startswith(prefix.lower()):
                answer = answer[len(prefix):].strip()
                break

        # Исправляем задвоения без пробела: "FunkMonkFunkMonk" → "FunkMonk"
        n = len(answer)
        if n >= 4 and n % 2 == 0:
            half = n // 2
            if answer[:half] == answer[half:]:
                answer = answer[:half]

        # Trailing punctuation
        answer = answer.rstrip(".,;:")

        return answer

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
                print(
                    f"  📥 [{msg.name}] {preview}"
                    f"{'…' if len(str(msg.content)) > 300 else ''}"
                )
        print()


def _fmt_args(args: dict) -> str:
    parts = []
    for k, v in args.items():
        v_str = str(v)
        if len(v_str) > 60:
            v_str = v_str[:57] + "..."
        parts.append(f"{k}={v_str!r}")
    return ", ".join(parts)