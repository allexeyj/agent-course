# GAIA Agent

ReAct агент для решения задач [GAIA benchmark](https://huggingface.co/spaces/gaia-benchmark/leaderboard).

## Results

**Score: 15/20 (75%)** — официальный результат на leaderboard.

Фактически **15/18 (83%)** — два вопроса про YouTube пропущены, т.к. yt-dlp блокируется в Colab без cookies.

## Stack

| Компонент | Что |
|-----------|-----|
| LLM | `inclusionai/ring-2.6-1t:free` via OpenRouter |
| Agent framework | LangGraph ReAct |
| Web search | Tavily |
| Vision | Moondream2 (`2025-06-21`) |
| Audio | Whisper `large-v3` |
| Object detection | YOLOv8x |
| Chess | Moondream2 → FEN → Stockfish |
| PDF | pdfplumber |

## Tools

`web_search` · `visit_url` · `read_pdf_url` · `describe_image` · `chess_move` · `transcribe_audio_file` · `youtube_info` · `youtube_transcribe` · `youtube_describe_frames` · `youtube_count_objects` · `youtube_full` · `analyze_python_logic` · `read_python_file` · `execute_python` · `read_excel`

## Run

```bash
pip install -r requirements.txt
apt-get install -y libvips42 stockfish ffmpeg

export OPENROUTER_API_KEY=...
export TAVILY_API_KEY=...

python run.py
```