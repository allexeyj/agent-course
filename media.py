"""
media.py — локальные медиа-инструменты для GAIA агента
  - Whisper  : аудио → текст
  - BLIP     : картинка → caption
  - yt-dlp   : YouTube → аудио / фреймы / оба
"""

from __future__ import annotations

import base64
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

# ──────────────────────────────────────────────
# Ленивая загрузка моделей (чтобы не грузить при импорте)
# ──────────────────────────────────────────────

_whisper_model = None
_blip_processor = None
_blip_model = None


def _get_whisper(model_size: str = "base"):
    """Загружает Whisper один раз и кэширует."""
    global _whisper_model
    if _whisper_model is None:
        import whisper
        print(f"[media] Загружаю Whisper ({model_size})…")
        _whisper_model = whisper.load_model(model_size)
    return _whisper_model


def _get_blip():
    """Загружает BLIP один раз и кэширует."""
    global _blip_processor, _blip_model
    if _blip_model is None:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        import torch
        print("[media] Загружаю BLIP…")
        _blip_processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        _blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _blip_model = _blip_model.to(device)
    return _blip_processor, _blip_model


# ──────────────────────────────────────────────
# WHISPER: аудио → текст
# ──────────────────────────────────────────────

def transcribe_audio(audio_path: str, word_timestamps: bool = False) -> dict:
    """
    Транскрибирует аудиофайл с помощью Whisper.

    Returns:
        {
          "text": str,                        # полный текст
          "segments": [{"start", "end", "text"}, ...]  # по сегментам
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
# BLIP: картинка → caption
# ──────────────────────────────────────────────

def caption_image(image_path: str) -> str:
    """
    Генерирует текстовое описание изображения через BLIP.
    Принимает путь к файлу или URL.
    """
    from PIL import Image
    import torch
    import requests as req

    processor, model = _get_blip()
    device = next(model.parameters()).device

    if image_path.startswith("http"):
        raw = req.get(image_path, timeout=10)
        import io
        image = Image.open(io.BytesIO(raw.content)).convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")

    inputs = processor(image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=150)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption


def caption_image_base64(b64_str: str) -> str:
    """Caption для base64-encoded изображения."""
    import io
    from PIL import Image
    import torch

    processor, model = _get_blip()
    device = next(model.parameters()).device

    img_bytes = base64.b64decode(b64_str)
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    inputs = processor(image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=150)
    return processor.decode(out[0], skip_special_tokens=True)


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
    """
    Возвращает метаданные YouTube видео: title, description,
    duration_sec, view_count, upload_date, chapters, tags.
    Никакого скачивания — мгновенно.
    """
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
    """
    Пытается скачать субтитры (авто или ручные).
    Возвращает текст субтитров или пустую строку.
    """
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

        # ищем скачанный файл субтитров
        for f in Path(tmpdir).glob("*.vtt"):
            text = _parse_vtt(f.read_text(encoding="utf-8"))
            return text

    return ""


def _parse_vtt(vtt_text: str) -> str:
    """Извлекает только текст из VTT субтитров."""
    import re
    lines = vtt_text.splitlines()
    result = []
    for line in lines:
        line = line.strip()
        # пропускаем таймкоды, заголовок и пустые строки
        if not line or line.startswith("WEBVTT") or "-->" in line:
            continue
        # убираем HTML-теги
        clean = re.sub(r"<[^>]+>", "", line)
        if clean and (not result or result[-1] != clean):
            result.append(clean)
    return " ".join(result)


def youtube_download_audio(url: str, out_dir: str | None = None) -> str:
    """
    Скачивает аудио из YouTube видео в mp3.
    Возвращает путь к файлу.
    """
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
            "preferredquality": "128",
        }],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_id = info["id"]

    mp3_path = os.path.join(out_dir, f"{video_id}.mp3")
    return mp3_path


def _adaptive_frame_count(duration_sec: int) -> int:
    """
    Адаптивное число фреймов:
      ≤30с  → каждые 3с  (≤10 фреймов)
      ≤5мин → каждые 5с  (≤60 фреймов)
      >5мин → каждые 10с (cap 100)
    """
    if duration_sec <= 30:
        interval = 3
    elif duration_sec <= 300:
        interval = 5
    else:
        interval = 10

    n = max(10, duration_sec // interval)
    return min(100, n)


def youtube_extract_frames(
    url: str,
    out_dir: str | None = None,
    max_frames: int | None = None,
) -> list[dict]:
    """
    Скачивает видео и извлекает фреймы равномерно.

    Returns:
        [{"path": str, "timestamp_sec": float}, ...]
    """
    import yt_dlp
    import cv2

    out_dir = out_dir or tempfile.mkdtemp()

    # получаем длительность
    info = _ydl_info(url)
    duration = info.get("duration", 60)

    n_frames = max_frames or _adaptive_frame_count(duration)

    # скачиваем видео
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

    # извлекаем фреймы
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


def youtube_full_analysis(url: str, out_dir: str | None = None) -> dict:
    """
    Полный анализ YouTube видео:
      1. Субтитры (если есть) — быстро
      2. Аудио → Whisper транскрипция
      3. Фреймы → BLIP captions
      4. Выравнивание по таймстампам

    Returns:
        {
          "metadata": {...},
          "transcript": [{"start", "end", "text"}],
          "frames": [{"timestamp_sec", "caption"}],
          "aligned": [{"timestamp_sec", "transcript_segment", "caption"}]
        }
    """
    out_dir = out_dir or tempfile.mkdtemp()

    print(f"[youtube] Получаю метаданные: {url}")
    meta = youtube_get_metadata(url)
    duration = meta["duration_sec"]
    print(f"[youtube] Длительность: {duration}с, фреймов: {_adaptive_frame_count(duration)}")

    # субтитры
    subtitles_text = ""
    if meta["subtitles_available"] or meta["automatic_captions_available"]:
        print("[youtube] Пробую субтитры…")
        subtitles_text = youtube_get_subtitles(url)

    # аудио → транскрипция
    print("[youtube] Скачиваю аудио…")
    audio_path = youtube_download_audio(url, out_dir)
    print("[youtube] Транскрибирую…")
    transcription = transcribe_audio(audio_path, word_timestamps=False)

    # фреймы → captions
    print("[youtube] Скачиваю видео и извлекаю фреймы…")
    frames = youtube_extract_frames(url, out_dir)
    print(f"[youtube] Генерирую captions для {len(frames)} фреймов…")
    frame_captions = []
    for f in frames:
        cap = caption_image(f["path"])
        frame_captions.append({"timestamp_sec": f["timestamp_sec"], "caption": cap})

    # выравнивание: для каждого фрейма находим ближайший сегмент транскрипции
    aligned = []
    segments = transcription["segments"]
    for fc in frame_captions:
        t = fc["timestamp_sec"]
        # ближайший сегмент
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


def youtube_audio_only(url: str, out_dir: str | None = None) -> dict:
    """Только аудио → Whisper транскрипция."""
    out_dir = out_dir or tempfile.mkdtemp()
    meta = youtube_get_metadata(url)
    audio_path = youtube_download_audio(url, out_dir)
    transcription = transcribe_audio(audio_path)
    return {"metadata": meta, "transcript": transcription}


def youtube_frames_only(url: str, out_dir: str | None = None) -> dict:
    """Только фреймы → BLIP captions."""
    out_dir = out_dir or tempfile.mkdtemp()
    meta = youtube_get_metadata(url)
    frames = youtube_extract_frames(url, out_dir)
    captions = [
        {"timestamp_sec": f["timestamp_sec"], "caption": caption_image(f["path"])}
        for f in frames
    ]
    return {"metadata": meta, "frames": captions}
