"""
tools.py — все инструменты для GAIA агента
"""

from __future__ import annotations

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
    transcribe_audio,
    youtube_audio_only,
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

    Args:
        url: Full URL including https://
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # убираем скрипты/стили
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        # обрезаем до разумного размера
        lines = [l for l in text.splitlines() if l.strip()]
        return "\n".join(lines[:400])
    except Exception as e:
        return f"Error fetching {url}: {e}"


# ──────────────────────────────────────────────
# CAPTION IMAGE
# ──────────────────────────────────────────────

@tool
def describe_image(image_path: str) -> str:
    """
    Generates a text description of an image using BLIP.
    Use for any question that involves an image file.

    Args:
        image_path: Local file path or URL to the image.
    """
    try:
        return caption_image(image_path)
    except Exception as e:
        return f"Error captioning image: {e}"


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
    ALWAYS call this first before any other YouTube tool to decide what's needed.

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
    Use when the question is about what is SAID in the video (speech, narration).

    Args:
        url: YouTube video URL.
    """
    try:
        result = youtube_audio_only(url)
        return result["transcript"]["text"]
    except Exception as e:
        return f"Error transcribing YouTube audio: {e}"


@tool
def youtube_describe_frames(url: str) -> str:
    """
    Downloads a YouTube video, extracts frames adaptively, and generates
    a visual description of each frame using BLIP.
    Use when the question is about what is SEEN in the video (objects, people, count).

    Args:
        url: YouTube video URL.
    """
    try:
        result = youtube_frames_only(url)
        lines = []
        for f in result["frames"]:
            lines.append(f"[{f['timestamp_sec']:.1f}s] {f['caption']}")
        return "\n".join(lines)
    except Exception as e:
        return f"Error analyzing YouTube frames: {e}"


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
def execute_python(code: str) -> str:
    """
    Executes a Python code snippet and returns stdout + stderr.
    Use when the question involves a Python file or requires computation.
    Timeout: 30 seconds.

    Args:
        code: Valid Python source code as a string.
    """
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
        return "Error: code execution timed out (30s)"
    except Exception as e:
        return f"Error executing code: {e}"
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


@tool
def read_python_file(file_path: str) -> str:
    """
    Reads a Python file and returns its source code.
    Use before execute_python to inspect the code first.

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
        else:
            # читаем все листы
            xl = pd.ExcelFile(file_path)
            sheets = {}
            for sheet in xl.sheet_names:
                sheets[sheet] = xl.parse(sheet)

            if len(sheets) == 1:
                df = list(sheets.values())[0]
            else:
                # несколько листов — возвращаем все
                parts = []
                for name, sdf in sheets.items():
                    parts.append(f"### Sheet: {name}\n{sdf.to_markdown(index=False)}")
                return "\n\n".join(parts)

        return df.to_markdown(index=False)
    except Exception as e:
        return f"Error reading file: {e}"


# ──────────────────────────────────────────────
# СБОРКА
# ──────────────────────────────────────────────

def build_all_tools() -> list:
    return [
        web_search,
        visit_url,
        describe_image,
        transcribe_audio_file,
        youtube_info,
        youtube_transcribe,
        youtube_describe_frames,
        youtube_full,
        execute_python,
        read_python_file,
        read_excel,
    ]
