"""
media.py — локальные медиа-инструменты для GAIA агента
  - Whisper large-v3 : аудио → текст
  - Moondream2       : картинка → ответ на произвольный вопрос
  - YOLOv8x          : детекция и подсчёт объектов на фреймах
  - Stockfish        : анализ шахматных позиций (FEN → лучший ход)
  - pdfplumber       : извлечение текста из PDF (файл или URL)
  - yt-dlp           : YouTube → аудио / фреймы / оба
"""

from __future__ import annotations

import base64
import io
import os
import subprocess
import tempfile
from pathlib import Path

# ──────────────────────────────────────────────
# Ленивая загрузка моделей
# ──────────────────────────────────────────────

_whisper_model   = None
_moondream_model = None
_yolo_model      = None


def _get_whisper(model_size: str = "large-v3"):
    global _whisper_model
    if _whisper_model is None:
        import whisper
        print(f"[media] Загружаю Whisper ({model_size})…")
        _whisper_model = whisper.load_model(model_size)
    return _whisper_model


def _get_moondream():
    """
    Загружает Moondream2 (revision 2025-06-21).
    Новый API: model.query(image, question) / model.caption(image)
    Токенайзер больше не нужен.
    Требует: transformers==4.56.1
    """
    global _moondream_model
    if _moondream_model is None:
        from transformers import AutoModelForCausalLM
        print("[media] Загружаю Moondream2 (2025-06-21)…")
        _moondream_model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2",
            revision="2025-06-21",
            trust_remote_code=True,
            device_map={"": "cuda"},
        )
        print("[media] Moondream2 загружен")
    return _moondream_model


def _get_yolo():
    """Загружает YOLOv8x — самая точная версия для детекции объектов."""
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        print("[media] Загружаю YOLOv8x…")
        _yolo_model = YOLO("yolov8x.pt")
    return _yolo_model


# ──────────────────────────────────────────────
# WHISPER: аудио → текст
# ──────────────────────────────────────────────

def transcribe_audio(audio_path: str, word_timestamps: bool = False) -> dict:
    model = _get_whisper()
    result = model.transcribe(audio_path, word_timestamps=word_timestamps, verbose=False)
    segments = [
        {"start": s["start"], "end": s["end"], "text": s["text"].strip()}
        for s in result.get("segments", [])
    ]
    return {"text": result["text"].strip(), "segments": segments}


# ──────────────────────────────────────────────
# MOONDREAM2: картинка + вопрос → ответ
# ──────────────────────────────────────────────

def _load_image(image_path: str):
    """Загружает PIL Image из пути к файлу или URL."""
    from PIL import Image
    import requests as req

    if image_path.startswith("http"):
        raw = req.get(image_path, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        return Image.open(io.BytesIO(raw.content)).convert("RGB")
    return Image.open(image_path).convert("RGB")


def caption_image(image_path: str, question: str = "Describe this image in detail.") -> str:
    """
    Отвечает на произвольный вопрос об изображении через Moondream2.

    Args:
        image_path: путь к файлу или URL
        question:   что именно спросить об изображении
    """
    model = _get_moondream()
    image = _load_image(image_path)
    return model.query(image, question)["answer"]


def caption_image_base64(b64_str: str, question: str = "Describe this image in detail.") -> str:
    """Отвечает на вопрос для base64-encoded изображения."""
    from PIL import Image

    model = _get_moondream()
    image = Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")
    return model.query(image, question)["answer"]


# ──────────────────────────────────────────────
# YOLO: подсчёт объектов на фреймах
# ──────────────────────────────────────────────

COCO_CLASSES = {
    "bird": 14, "cat": 15, "dog": 16, "person": 0,
    "car": 2, "airplane": 4, "boat": 8,
    "horse": 17, "sheep": 18, "cow": 19,
}


def count_objects_in_frames(frame_paths: list[str], object_name: str = "bird") -> dict:
    model    = _get_yolo()
    class_id = COCO_CLASSES.get(object_name.lower())

    per_frame = []
    max_count = 0
    max_frame = ""

    for path in frame_paths:
        results = model(path, classes=[class_id] if class_id is not None else None, verbose=False)
        count   = len(results[0].boxes)
        per_frame.append({"path": path, "count": count})
        if count > max_count:
            max_count = count
            max_frame = path

    return {"max_count": max_count, "max_at_frame": max_frame, "per_frame": per_frame}


# ──────────────────────────────────────────────
# CHESS: изображение → FEN → лучший ход
# ──────────────────────────────────────────────

def extract_fen_from_image(image_path: str) -> str:
    prompt = (
        "This is a chess board. "
        "Please read the position carefully and provide the FEN notation "
        "of this position. Output ONLY the FEN string, nothing else."
    )
    return caption_image(image_path, question=prompt).strip().split("\n")[0].strip()


def get_best_chess_move(image_path: str, turn: str = "black") -> str:
    import chess
    import chess.engine

    fen_raw = extract_fen_from_image(image_path)
    print(f"[chess] FEN от Moondream2: {fen_raw!r}")

    try:
        board = chess.Board(fen_raw)
    except Exception:
        parts = fen_raw.split()
        if len(parts) >= 2:
            parts[1] = "b" if turn == "black" else "w"
        board = chess.Board(" ".join(parts) if len(parts) >= 2 else chess.STARTING_FEN)

    board.turn = chess.BLACK if turn == "black" else chess.WHITE

    engine_path = next(
        (p for p in ["stockfish", "/usr/bin/stockfish", "/usr/local/bin/stockfish", "/usr/games/stockfish"]
         if not subprocess.run([p, "quit"], capture_output=True, timeout=2).returncode
         if True),
        None,
    )

    if engine_path is None:
        return caption_image(
            image_path,
            question=f"This is a chess board. It is {turn}'s turn. "
                     "What is the best move? Provide only the move in algebraic notation."
        )

    with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
        result   = engine.play(board, chess.engine.Limit(time=3.0))
        move_san = board.san(result.move)

    print(f"[chess] Лучший ход: {move_san}")
    return move_san


# ──────────────────────────────────────────────
# PDF: извлечение текста
# ──────────────────────────────────────────────

def read_pdf(source: str, max_chars: int = 8000) -> str:
    import pdfplumber
    import requests as req

    try:
        if source.startswith("http"):
            resp      = req.get(source, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
            pdf_bytes = io.BytesIO(resp.content)
        else:
            pdf_bytes = open(source, "rb")

        with pdfplumber.open(pdf_bytes) as pdf:
            full_text = "\n\n".join(p.extract_text() or "" for p in pdf.pages)

        if not source.startswith("http"):
            pdf_bytes.close()

        return full_text[:max_chars]
    except Exception as e:
        return f"Error reading PDF from {source}: {e}"


# ──────────────────────────────────────────────
# YOUTUBE
# ──────────────────────────────────────────────

def _ydl_info(url: str) -> dict:
    import yt_dlp
    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True}) as ydl:
        return ydl.extract_info(url, download=False)


def youtube_get_metadata(url: str) -> dict:
    info = _ydl_info(url)
    return {
        "title":                        info.get("title", ""),
        "description":                  info.get("description", ""),
        "duration_sec":                 info.get("duration", 0),
        "view_count":                   info.get("view_count", 0),
        "upload_date":                  info.get("upload_date", ""),
        "chapters":                     info.get("chapters", []),
        "tags":                         info.get("tags", []),
        "subtitles_available":          list(info.get("subtitles", {}).keys()),
        "automatic_captions_available": list(info.get("automatic_captions", {}).keys()),
    }


def youtube_get_subtitles(url: str, lang: str = "en") -> str:
    import yt_dlp

    with tempfile.TemporaryDirectory() as tmpdir:
        ydl_opts = {
            "quiet": True, "no_warnings": True, "skip_download": True,
            "writesubtitles": True, "writeautomaticsub": True,
            "subtitleslangs": [lang], "subtitlesformat": "vtt",
            "outtmpl": os.path.join(tmpdir, "subs"),
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        for f in Path(tmpdir).glob("*.vtt"):
            return _parse_vtt(f.read_text(encoding="utf-8"))
    return ""


def _parse_vtt(vtt_text: str) -> str:
    import re
    result = []
    for line in vtt_text.splitlines():
        line = line.strip()
        if not line or line.startswith("WEBVTT") or "-->" in line:
            continue
        clean = re.sub(r"<[^>]+>", "", line)
        if clean and (not result or result[-1] != clean):
            result.append(clean)
    return " ".join(result)


def youtube_download_audio(url: str, out_dir: str | None = None) -> str:
    import yt_dlp

    out_dir  = out_dir or tempfile.mkdtemp()
    ydl_opts = {
        "format": "bestaudio/best", "quiet": True, "no_warnings": True,
        "outtmpl": os.path.join(out_dir, "%(id)s.%(ext)s"),
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
    return os.path.join(out_dir, f"{info['id']}.mp3")


def _adaptive_frame_count(duration_sec: int) -> int:
    interval = 3 if duration_sec <= 30 else (5 if duration_sec <= 300 else 10)
    return min(100, max(10, duration_sec // interval))


def youtube_extract_frames(url: str, out_dir: str | None = None, max_frames: int | None = None) -> list[dict]:
    import cv2, yt_dlp

    out_dir  = out_dir or tempfile.mkdtemp()
    info     = _ydl_info(url)
    duration = info.get("duration", 60)
    n_frames = max_frames or _adaptive_frame_count(duration)

    video_path = os.path.join(out_dir, f"{info['id']}.mp4")
    with yt_dlp.YoutubeDL({
        "format": "bestvideo[ext=mp4][height<=480]+bestaudio/best[height<=480]",
        "quiet": True, "no_warnings": True,
        "outtmpl": video_path, "merge_output_format": "mp4",
    }) as ydl:
        ydl.download([url])

    cap          = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25
    frames_dir   = os.path.join(out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    frames = []
    for idx in [int(i * total_frames / n_frames) for i in range(n_frames)]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame_path = os.path.join(frames_dir, f"frame_{idx:06d}.jpg")
        cv2.imwrite(frame_path, frame)
        frames.append({"path": frame_path, "timestamp_sec": round(idx / fps, 2)})
    cap.release()
    return frames


def youtube_audio_only(url: str, out_dir: str | None = None) -> dict:
    out_dir = out_dir or tempfile.mkdtemp()
    return {"metadata": youtube_get_metadata(url), "transcript": transcribe_audio(youtube_download_audio(url, out_dir))}


def youtube_frames_only(url: str, out_dir: str | None = None, question: str = "Describe this image in detail.") -> dict:
    out_dir = out_dir or tempfile.mkdtemp()
    frames  = youtube_extract_frames(url, out_dir)
    return {
        "metadata": youtube_get_metadata(url),
        "frames": [{"timestamp_sec": f["timestamp_sec"], "caption": caption_image(f["path"], question=question)} for f in frames],
    }


def youtube_count_objects(url: str, object_name: str = "bird", out_dir: str | None = None) -> dict:
    out_dir    = out_dir or tempfile.mkdtemp()
    frames     = youtube_extract_frames(url, out_dir)
    timestamps = {f["path"]: f["timestamp_sec"] for f in frames}
    result     = count_objects_in_frames([f["path"] for f in frames], object_name)
    return {
        "max_count":         result["max_count"],
        "max_timestamp_sec": timestamps.get(result["max_at_frame"], 0),
        "per_frame": [{"timestamp_sec": timestamps.get(i["path"], 0), "count": i["count"]} for i in result["per_frame"]],
    }


def youtube_full_analysis(url: str, out_dir: str | None = None) -> dict:
    out_dir = out_dir or tempfile.mkdtemp()
    meta    = youtube_get_metadata(url)

    subtitles_text = youtube_get_subtitles(url) if (meta["subtitles_available"] or meta["automatic_captions_available"]) else ""
    transcription  = transcribe_audio(youtube_download_audio(url, out_dir))
    frames         = youtube_extract_frames(url, out_dir)
    frame_captions = [{"timestamp_sec": f["timestamp_sec"], "caption": caption_image(f["path"])} for f in frames]

    segments = transcription["segments"]
    aligned  = [
        {
            "timestamp_sec":      fc["timestamp_sec"],
            "caption":            fc["caption"],
            "transcript_segment": min(segments, key=lambda s: abs((s["start"] + s["end"]) / 2 - fc["timestamp_sec"]), default={"text": ""})["text"],
        }
        for fc in frame_captions
    ]

    return {"metadata": meta, "subtitles": subtitles_text, "transcript": transcription, "frames": frame_captions, "aligned": aligned}