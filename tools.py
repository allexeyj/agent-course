"""
tools.py — все инструменты для GAIA агента
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool

from media import (
    caption_image,
    get_best_chess_move,
    read_pdf,
    transcribe_audio,
    youtube_audio_only,
    youtube_count_objects,
    youtube_frames_only,
    youtube_full_analysis,
    youtube_get_metadata,
    youtube_get_subtitles,
)

# ──────────────────────────────────────────────
# WEB SEARCH  (Tavily)
# ──────────────────────────────────────────────

web_search = TavilySearchResults(
    max_results=5,
    name="web_search",
    description=(
        "Search the web for factual information, Wikipedia articles, news, etc. "
        "Use for any question that requires external knowledge or current facts."
    ),
)

# ──────────────────────────────────────────────
# VISIT URL
# ──────────────────────────────────────────────

@tool
def visit_url(url: str) -> str:
    """
    Fetches and returns the text content of a web page.
    Use when web_search returns a URL you want to read in full.
    For PDF links use read_pdf_url instead.

    Args:
        url: Full URL including https://
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()

        # если это PDF — перенаправляем
        ct = resp.headers.get("content-type", "")
        if "pdf" in ct or url.lower().endswith(".pdf"):
            return read_pdf(url)

        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        lines = [l for l in text.splitlines() if l.strip()]
        return "\n".join(lines[:500])
    except Exception as e:
        return f"Error fetching {url}: {e}"


# ──────────────────────────────────────────────
# READ PDF
# ──────────────────────────────────────────────

@tool
def read_pdf_url(source: str) -> str:
    """
    Extracts text from a PDF file (local path or URL).
    Use when a question references a scientific paper, report, or any PDF document.
    Always prefer this over visit_url for .pdf links.

    Args:
        source: Local file path or full URL to a PDF file.
    """
    try:
        return read_pdf(source, max_chars=8000)
    except Exception as e:
        return f"Error reading PDF: {e}"


# ──────────────────────────────────────────────
# DESCRIBE IMAGE  (Moondream2)
# ──────────────────────────────────────────────

@tool
def describe_image(image_path: str, question: str = "Describe this image in detail.") -> str:
    """
    Answers a question about an image using Moondream2 vision model.
    Unlike a simple captioner, this accepts ANY question about the image.

    Args:
        image_path: Local file path or URL to the image.
        question:   Specific question to ask about the image.
                    Examples:
                      - "Describe this image in detail."
                      - "How many people are in this image?"
                      - "What text is written on the sign?"
                      - "What is the breed of this dog?"
    """
    try:
        return caption_image(image_path, question=question)
    except Exception as e:
        return f"Error analyzing image: {e}"


# ──────────────────────────────────────────────
# CHESS MOVE  (Moondream2 → FEN → Stockfish)
# ──────────────────────────────────────────────

@tool
def chess_move(image_path: str, turn: str = "black") -> str:
    """
    Analyzes a chess board image and returns the best move.

    Pipeline:
      1. Moondream2 reads the board and extracts FEN notation
      2. Stockfish calculates the best move
      3. Returns move in standard algebraic notation (e.g. "Qh4#")

    Use this tool for ANY question involving a chess board image.
    Do NOT use describe_image for chess — use this tool instead.

    Args:
        image_path: Local path to the chess board image.
        turn:       Whose turn it is: "black" or "white".
    """
    try:
        return get_best_chess_move(image_path, turn=turn)
    except Exception as e:
        return f"Error analyzing chess position: {e}"


# ──────────────────────────────────────────────
# TRANSCRIBE AUDIO
# ──────────────────────────────────────────────

@tool
def transcribe_audio_file(audio_path: str) -> str:
    """
    Transcribes an audio file (mp3, wav, m4a, etc.) to text using Whisper.
    Use when the question involves a voice recording or audio attachment.

    Args:
        audio_path: Local path to the audio file.
    """
    try:
        result = transcribe_audio(audio_path)
        return result["text"]
    except Exception as e:
        return f"Error transcribing audio: {e}"


# ──────────────────────────────────────────────
# YOUTUBE
# ──────────────────────────────────────────────

@tool
def youtube_info(url: str) -> str:
    """
    Retrieves YouTube video metadata WITHOUT downloading: title, description,
    duration, chapters, tags, and whether subtitles are available.
    Also attempts to fetch subtitles immediately if available.
    ALWAYS call this first before any other YouTube tool.

    Args:
        url: YouTube video URL.
    """
    try:
        meta = youtube_get_metadata(url)
        subtitles = youtube_get_subtitles(url)

        parts = [
            f"Title: {meta['title']}",
            f"Duration: {meta['duration_sec']}s",
            f"Description:\n{meta['description'][:1000]}",
            f"Tags: {', '.join(meta['tags'][:20])}",
            f"Subtitles available: {meta['subtitles_available']}",
            f"Auto-captions available: {meta['automatic_captions_available']}",
        ]
        if subtitles:
            parts.append(f"\nSubtitle text:\n{subtitles[:3000]}")

        return "\n".join(parts)
    except Exception as e:
        return f"Error fetching YouTube info: {e}"


@tool
def youtube_transcribe(url: str) -> str:
    """
    Downloads audio from a YouTube video and transcribes it with Whisper.
    Use when the question is about what is SAID in the video (speech, narration, dialogue).

    Args:
        url: YouTube video URL.
    """
    try:
        result = youtube_audio_only(url)
        return result["transcript"]["text"]
    except Exception as e:
        return f"Error transcribing YouTube audio: {e}"


@tool
def youtube_describe_frames(url: str, question: str = "Describe this image in detail.") -> str:
    """
    Downloads a YouTube video, extracts frames, and asks Moondream2
    a specific question about each frame.

    Use when the question is about what is SEEN in the video.
    For counting specific objects (birds, people, cars) use youtube_count_objects instead.

    Args:
        url:      YouTube video URL.
        question: What to ask about each frame.
                  Examples:
                    - "Describe this image in detail."
                    - "What animals are visible in this frame?"
                    - "What is written on the screen?"
    """
    try:
        result = youtube_frames_only(url, question=question)
        lines = []
        for f in result["frames"]:
            lines.append(f"[{f['timestamp_sec']:.1f}s] {f['caption']}")
        return "\n".join(lines)
    except Exception as e:
        return f"Error analyzing YouTube frames: {e}"


@tool
def youtube_count_objects(url: str, object_name: str = "bird") -> str:
    """
    Downloads a YouTube video, extracts frames, and counts a specific
    type of object in each frame using YOLOv8 object detection.

    Use when the question asks "how many X are visible simultaneously"
    or "what is the maximum number of X on screen at once".

    Supported objects: bird, person, car, cat, dog, airplane, boat,
                       horse, sheep, cow (and any COCO dataset class).

    Args:
        url:         YouTube video URL.
        object_name: What to count (e.g. "bird", "person", "car").
    """
    try:
        from media import youtube_count_objects as _count
        result = _count(url, object_name=object_name)

        lines = [
            f"Maximum {object_name}s simultaneously on screen: {result['max_count']}",
            f"Timestamp of maximum: {result['max_timestamp_sec']:.1f}s",
            "",
            "Per-frame breakdown:",
        ]
        for item in result["per_frame"]:
            if item["count"] > 0:
                lines.append(f"  [{item['timestamp_sec']:.1f}s] {item['count']} {object_name}(s)")

        return "\n".join(lines)
    except Exception as e:
        return f"Error counting objects in YouTube video: {e}"


@tool
def youtube_full(url: str) -> str:
    """
    Full YouTube analysis: audio transcription + frame captioning aligned by timestamp.
    Use when both speech AND visuals are needed to answer the question.
    WARNING: slow — only use if youtube_transcribe and youtube_describe_frames are not enough.

    Args:
        url: YouTube video URL.
    """
    try:
        result = youtube_full_analysis(url)
        lines = [
            f"Title: {result['metadata']['title']}",
            f"Duration: {result['metadata']['duration_sec']}s",
            "",
            "=== ALIGNED TRANSCRIPT + FRAMES ===",
        ]
        for a in result["aligned"]:
            lines.append(
                f"[{a['timestamp_sec']:.1f}s] "
                f"Visual: {a['caption']} | "
                f"Audio: {a['transcript_segment']}"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"Error in full YouTube analysis: {e}"


# ──────────────────────────────────────────────
# EXECUTE PYTHON
# ──────────────────────────────────────────────

@tool
def analyze_python_logic(file_path: str) -> str:
    """
    Reads a Python file and performs STATIC analysis to reason about
    its output WITHOUT executing it.

    Use this BEFORE execute_python when the code contains:
      - while True / infinite loops
      - time.sleep() calls
      - random / non-deterministic elements
      - recursion without clear termination

    Returns the source code annotated with analysis notes so the LLM
    can reason about the final output.

    Args:
        file_path: Local path to the .py file.
    """
    try:
        source = Path(file_path).read_text(encoding="utf-8")
        tree = ast.parse(source)

        notes = []

        # Ищем while True
        for node in ast.walk(tree):
            if isinstance(node, ast.While):
                if isinstance(node.test, ast.Constant) and node.test.value is True:
                    notes.append(f"⚠ Line {node.lineno}: 'while True' loop detected")

        # Ищем time.sleep
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "sleep"
            ):
                notes.append(f"⚠ Line {node.lineno}: time.sleep() call — execution will be slow")

        # Ищем random
        has_random = any(
            isinstance(node, ast.Import) and any(a.name == "random" for a in node.names)
            or isinstance(node, ast.ImportFrom) and node.module == "random"
            for node in ast.walk(tree)
        )
        if has_random:
            notes.append("⚠ Uses random module — output may vary unless constrained")

        # Ищем print statements
        prints = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "print"
            ):
                prints.append(f"  print() at line {node.lineno}")
        if prints:
            notes.append("📤 Print statements found:\n" + "\n".join(prints))

        analysis = "\n".join(notes) if notes else "No issues detected."

        return textwrap.dedent(f"""
=== SOURCE CODE ===
{source}

=== STATIC ANALYSIS ===
{analysis}

=== TASK ===
Based on the source code and analysis above, reason step by step about
what value this program will ALWAYS print, regardless of random variation.
What is the only possible final numeric output?
        """).strip()

    except Exception as e:
        return f"Error analyzing Python file: {e}"


@tool
def execute_python(code: str) -> str:
    """
    Executes a Python code snippet and returns stdout + stderr.
    Use for deterministic code without infinite loops.
    If the code has while True, time.sleep, or randomness —
    use analyze_python_logic first.
    Timeout: 30 seconds.

    Args:
        code: Valid Python source code as a string.
    """
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            tmp_path = f.name

        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = ""
        if result.stdout:
            output += f"STDOUT:\n{result.stdout.strip()}"
        if result.stderr:
            output += f"\nSTDERR:\n{result.stderr.strip()}"
        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return (
            "Error: execution timed out (30s). "
            "This code likely has an infinite loop or sleep. "
            "Use analyze_python_logic tool instead to reason about the output statically."
        )
    except Exception as e:
        return f"Error executing code: {e}"
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


@tool
def read_python_file(file_path: str) -> str:
    """
    Reads a Python file and returns its source code.
    Use to inspect the code before deciding whether to execute or analyze it.

    Args:
        file_path: Local path to the .py file.
    """
    try:
        return Path(file_path).read_text(encoding="utf-8")
    except Exception as e:
        return f"Error reading file: {e}"


# ──────────────────────────────────────────────
# READ EXCEL / CSV
# ──────────────────────────────────────────────

@tool
def read_excel(file_path: str) -> str:
    """
    Reads an Excel (.xlsx) or CSV file and returns its content as a markdown table.
    Use when the question involves a spreadsheet or tabular data file.

    Args:
        file_path: Local path to the .xlsx or .csv file.
    """
    try:
        ext = Path(file_path).suffix.lower()
        if ext == ".csv":
            df = pd.read_csv(file_path)
            return df.to_markdown(index=False)

        xl = pd.ExcelFile(file_path)
        sheets = {sheet: xl.parse(sheet) for sheet in xl.sheet_names}

        if len(sheets) == 1:
            return list(sheets.values())[0].to_markdown(index=False)

        parts = []
        for name, sdf in sheets.items():
            parts.append(f"### Sheet: {name}\n{sdf.to_markdown(index=False)}")
        return "\n\n".join(parts)
    except Exception as e:
        return f"Error reading file: {e}"


# ──────────────────────────────────────────────
# СБОРКА
# ──────────────────────────────────────────────

def build_all_tools() -> list:
    return [
        # поиск и веб
        web_search,
        visit_url,
        read_pdf_url,
        # изображения
        describe_image,
        chess_move,
        # аудио
        transcribe_audio_file,
        # youtube
        youtube_info,
        youtube_transcribe,
        youtube_describe_frames,
        youtube_count_objects,
        youtube_full,
        # код
        analyze_python_logic,
        read_python_file,
        execute_python,
        # данные
        read_excel,
    ]