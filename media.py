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

_whisper_model = None
_moondream_model = None
_moondream_tokenizer = None
_yolo_model = None


def _get_whisper(model_size: str = "large-v3"):
    global _whisper_model
    if _whisper_model is None:
        import whisper
        print(f"[media] Загружаю Whisper ({model_size})…")
        _whisper_model = whisper.load_model(model_size)
    return _whisper_model


def _get_moondream():
    """
    Загружает Moondream2 (~2GB VRAM).
    Принимает произвольный вопрос об изображении.
    """
    global _moondream_model, _moondream_tokenizer
    if _moondream_model is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print("[media] Загружаю Moondream2…")
        revision = "2025-01-09"
        _moondream_tokenizer = AutoTokenizer.from_pretrained(
            "vikhyatk/moondream2", revision=revision
        )
        _moondream_model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2",
            revision=revision,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _moondream_model = _moondream_model.to(device)
        _moondream_model.eval()
        print(f"[media] Moondream2 загружен на {device}")
    return _moondream_model, _moondream_tokenizer


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
    """
    Транскрибирует аудиофайл с помощью Whisper large-v3.

    Returns:
        {
          "text": str,
          "segments": [{"start", "end", "text"}, ...]
        }
    """
    model = _get_whisper()
    result = model.transcribe(
        audio_path,
        word_timestamps=word_timestamps,
        verbose=False,
    )
    segments = [
        {"start": s["start"], "end": s["end"], "text": s["text"].strip()}
        for s in result.get("segments", [])
    ]
    return {"text": result["text"].strip(), "segments": segments}


# ──────────────────────────────────────────────
# MOONDREAM2: картинка + вопрос → ответ
# ──────────────────────────────────────────────

def caption_image(image_path: str, question: str = "Describe this image in detail.") -> str:
    """
    Отвечает на произвольный вопрос об изображении через Moondream2.

    Args:
        image_path: путь к файлу или URL
        question:   что именно спросить об изображении
    """
    import requests as req
    from PIL import Image

    model, tokenizer = _get_moondream()

    if image_path.startswith("http"):
        raw = req.get(image_path, timeout=10)
        image = Image.open(io.BytesIO(raw.content)).convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")

    enc = model.encode_image(image)
    answer = model.answer_question(enc, question, tokenizer)
    return answer


def caption_image_base64(b64_str: str, question: str = "Describe this image in detail.") -> str:
    """Отвечает на вопрос для base64-encoded изображения."""
    from PIL import Image

    model, tokenizer = _get_moondream()
    img_bytes = base64.b64decode(b64_str)
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    enc = model.encode_image(image)
    return model.answer_question(enc, question, tokenizer)


# ──────────────────────────────────────────────
# YOLO: подсчёт объектов на фреймах
# ──────────────────────────────────────────────

# COCO class ids
COCO_CLASSES = {
    "bird": 14,
    "cat": 15,
    "dog": 16,
    "person": 0,
    "car": 2,
    "airplane": 4,
    "boat": 8,
    "horse": 17,
    "sheep": 18,
    "cow": 19,
}


def count_objects_in_frames(
    frame_paths: list[str],
    object_name: str = "bird",
) -> dict:
    """
    Считает объекты заданного класса на каждом фрейме через YOLOv8x.

    Returns:
        {
          "max_count": int,
          "max_at_frame": str,
          "per_frame": [{"path", "count"}, ...]
        }
    """
    model = _get_yolo()
    class_id = COCO_CLASSES.get(object_name.lower())

    per_frame = []
    max_count = 0
    max_frame = ""

    for path in frame_paths:
        if class_id is not None:
            results = model(path, classes=[class_id], verbose=False)
        else:
            results = model(path, verbose=False)

        count = len(results[0].boxes)
        per_frame.append({"path": path, "count": count})

        if count > max_count:
            max_count = count
            max_frame = path

    return {
        "max_count": max_count,
        "max_at_frame": max_frame,
        "per_frame": per_frame,
    }


# ──────────────────────────────────────────────
# CHESS: изображение → FEN → лучший ход
# ──────────────────────────────────────────────

def extract_fen_from_image(image_path: str) -> str:
    """Извлекает FEN нотацию шахматной позиции из изображения через Moondream2."""
    prompt = (
        "This is a chess board. "
        "Please read the position carefully and provide the FEN notation "
        "of this position. Output ONLY the FEN string, nothing else."
    )
    fen = caption_image(image_path, question=prompt)
    return fen.strip().split("\n")[0].strip()


def get_best_chess_move(image_path: str, turn: str = "black") -> str:
    """
    Полный пайплайн: изображение шахматной доски → лучший ход.

    1. Moondream2 извлекает FEN из картинки
    2. python-chess проверяет корректность
    3. Stockfish находит лучший ход

    Args:
        image_path: путь к изображению шахматной доски
        turn:       чья очередь ходить ("black" или "white")

    Returns:
        Ход в алгебраической нотации (e.g. "Qh4#")
    """
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
            fen_fixed = " ".join(parts)
        else:
            fen_fixed = chess.STARTING_FEN
        board = chess.Board(fen_fixed)

    board.turn = chess.BLACK if turn == "black" else chess.WHITE

    stockfish_paths = [
        "stockfish",
        "/usr/bin/stockfish",
        "/usr/local/bin/stockfish",
        "/usr/games/stockfish",
    ]
    engine_path = None
    for p in stockfish_paths:
        try:
            subprocess.run([p, "quit"], capture_output=True, timeout=2)
            engine_path = p
            break
        except Exception:
            continue

    if engine_path is None:
        prompt = (
            f"This is a chess board. It is {turn}'s turn. "
            "What is the best move for the current player? "
            "Provide only the move in standard algebraic notation."
        )
        return caption_image(image_path, question=prompt)

    with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
        result = engine.play(board, chess.engine.Limit(time=3.0))
        move_san = board.san(result.move)

    print(f"[chess] Лучший ход: {move_san}")
    return move_san


# ──────────────────────────────────────────────
# PDF: извлечение текста
# ──────────────────────────────────────────────

def read_pdf(source: str, max_chars: int = 8000) -> str:
    """
    Извлекает текст из PDF файла или URL.

    Args:
        source:    путь к локальному файлу или URL
        max_chars: максимальное число символов в ответе
    """
    import pdfplumber
    import requests as req

    try:
        if source.startswith("http"):
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = req.get(source, headers=headers, timeout=30)
            resp.raise_for_status()
            pdf_bytes = io.BytesIO(resp.content)
        else:
            pdf_bytes = open(source, "rb")

        with pdfplumber.open(pdf_bytes) as pdf:
            pages_text = []
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages_text.append(text)
            full_text = "\n\n".join(pages_text)

        if not source.startswith("http"):
            pdf_bytes.close()

        return full_text[:max_chars]

    except Exception as e:
        return f"Error reading PDF from {source}: {e}"


# ──────────────────────────────────────────────
# YOUTUBE: скачать видео/аудио
# ──────────────────────────────────────────────

def _ydl_info(url: str) -> dict:
    """Получает метаданные видео без скачивания."""
    import yt_dlp
    ydl_opts = {"quiet": True, "no_warnings": True, "skip_download": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        return ydl.extract_info(url, download=False)


def youtube_get_metadata(url: str) -> dict:
    info = _ydl_info(url)
    return {
        "title": info.get("title", ""),
        "description": info.get("description", ""),
        "duration_sec": info.get("duration", 0),
        "view_count": info.get("view_count", 0),
        "upload_date": info.get("upload_date", ""),
        "chapters": info.get("chapters", []),
        "tags": info.get("tags", []),
        "subtitles_available": list(info.get("subtitles", {}).keys()),
        "automatic_captions_available": list(info.get("automatic_captions", {}).keys()),
    }


def youtube_get_subtitles(url: str, lang: str = "en") -> str:
    import yt_dlp

    with tempfile.TemporaryDirectory() as tmpdir:
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": [lang],
            "subtitlesformat": "vtt",
            "outtmpl": os.path.join(tmpdir, "subs"),
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        for f in Path(tmpdir).glob("*.vtt"):
            return _parse_vtt(f.read_text(encoding="utf-8"))

    return ""


def _parse_vtt(vtt_text: str) -> str:
    import re
    lines = vtt_text.splitlines()
    result = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("WEBVTT") or "-->" in line:
            continue
        clean = re.sub(r"<[^>]+>", "", line)
        if clean and (not result or result[-1] != clean):
            result.append(clean)
    return " ".join(result)


def youtube_download_audio(url: str, out_dir: str | None = None) -> str:
    import yt_dlp

    out_dir = out_dir or tempfile.mkdtemp()
    ydl_opts = {
        "format": "bestaudio/best",
        "quiet": True,
        "no_warnings": True,
        "outtmpl": os.path.join(out_dir, "%(id)s.%(ext)s"),
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_id = info["id"]

    return os.path.join(out_dir, f"{video_id}.mp3")


def _adaptive_frame_count(duration_sec: int) -> int:
    if duration_sec <= 30:
        interval = 3
    elif duration_sec <= 300:
        interval = 5
    else:
        interval = 10
    return min(100, max(10, duration_sec // interval))


def youtube_extract_frames(
    url: str,
    out_dir: str | None = None,
    max_frames: int | None = None,
) -> list[dict]:
    import cv2
    import yt_dlp

    out_dir = out_dir or tempfile.mkdtemp()

    info = _ydl_info(url)
    duration = info.get("duration", 60)
    n_frames = max_frames or _adaptive_frame_count(duration)

    video_path = os.path.join(out_dir, f"{info['id']}.mp4")
    ydl_opts = {
        "format": "bestvideo[ext=mp4][height<=480]+bestaudio/best[height<=480]",
        "quiet": True,
        "no_warnings": True,
        "outtmpl": video_path,
        "merge_output_format": "mp4",
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    indices = [int(i * total_frames / n_frames) for i in range(n_frames)]
    frames = []

    frames_dir = os.path.join(out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        timestamp = idx / fps
        frame_path = os.path.join(frames_dir, f"frame_{idx:06d}.jpg")
        cv2.imwrite(frame_path, frame)
        frames.append({"path": frame_path, "timestamp_sec": round(timestamp, 2)})

    cap.release()
    return frames


def youtube_audio_only(url: str, out_dir: str | None = None) -> dict:
    out_dir = out_dir or tempfile.mkdtemp()
    meta = youtube_get_metadata(url)
    audio_path = youtube_download_audio(url, out_dir)
    transcription = transcribe_audio(audio_path)
    return {"metadata": meta, "transcript": transcription}


def youtube_frames_only(
    url: str,
    out_dir: str | None = None,
    question: str = "Describe this image in detail.",
) -> dict:
    out_dir = out_dir or tempfile.mkdtemp()
    meta = youtube_get_metadata(url)
    frames = youtube_extract_frames(url, out_dir)
    captions = [
        {
            "timestamp_sec": f["timestamp_sec"],
            "caption": caption_image(f["path"], question=question),
        }
        for f in frames
    ]
    return {"metadata": meta, "frames": captions}


def youtube_count_objects(
    url: str,
    object_name: str = "bird",
    out_dir: str | None = None,
) -> dict:
    out_dir = out_dir or tempfile.mkdtemp()
    frames = youtube_extract_frames(url, out_dir)
    frame_paths = [f["path"] for f in frames]
    timestamps = {f["path"]: f["timestamp_sec"] for f in frames}

    yolo_result = count_objects_in_frames(frame_paths, object_name)

    per_frame_with_ts = [
        {
            "timestamp_sec": timestamps.get(item["path"], 0),
            "count": item["count"],
        }
        for item in yolo_result["per_frame"]
    ]

    max_ts = timestamps.get(yolo_result["max_at_frame"], 0)

    return {
        "max_count": yolo_result["max_count"],
        "max_timestamp_sec": max_ts,
        "per_frame": per_frame_with_ts,
    }


def youtube_full_analysis(url: str, out_dir: str | None = None) -> dict:
    out_dir = out_dir or tempfile.mkdtemp()

    print(f"[youtube] Метаданные: {url}")
    meta = youtube_get_metadata(url)
    duration = meta["duration_sec"]
    print(f"[youtube] Длительность: {duration}с")

    subtitles_text = ""
    if meta["subtitles_available"] or meta["automatic_captions_available"]:
        print("[youtube] Загружаю субтитры…")
        subtitles_text = youtube_get_subtitles(url)

    print("[youtube] Скачиваю аудио…")
    audio_path = youtube_download_audio(url, out_dir)
    print("[youtube] Транскрибирую…")
    transcription = transcribe_audio(audio_path)

    print("[youtube] Извлекаю фреймы…")
    frames = youtube_extract_frames(url, out_dir)
    print(f"[youtube] Генерирую описания для {len(frames)} фреймов…")
    frame_captions = [
        {
            "timestamp_sec": f["timestamp_sec"],
            "caption": caption_image(f["path"]),
        }
        for f in frames
    ]

    segments = transcription["segments"]
    aligned = []
    for fc in frame_captions:
        t = fc["timestamp_sec"]
        nearest = min(
            segments,
            key=lambda s: abs((s["start"] + s["end"]) / 2 - t),
            default=None,
        )
        aligned.append({
            "timestamp_sec": t,
            "caption": fc["caption"],
            "transcript_segment": nearest["text"] if nearest else "",
        })

    return {
        "metadata": meta,
        "subtitles": subtitles_text,
        "transcript": transcription,
        "frames": frame_captions,
        "aligned": aligned,
    }