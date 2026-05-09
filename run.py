"""
run.py — запуск GAIA агента и сабмит ответов
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import requests

from agent import GAIAAgent

# ──────────────────────────────────────────────
# КОНФИГ
# ──────────────────────────────────────────────

API_BASE        = "https://agents-course-unit4-scoring.hf.space"
DATASET_PATH    = "gaia_dataset.json"
ANSWERS_PATH    = "answers.json"

HF_USERNAME     = os.environ.get("HF_USERNAME", "your_username")
HF_SPACE_URL    = os.environ.get(
    "HF_SPACE_URL",
    "https://huggingface.co/spaces/your_username/your_space/tree/main",
)

VERBOSE         = True
TIMEOUT_SEC     = 280

# Вопросы которые пропускаем (YouTube заблокирован в Colab).
# Убери task_id из этого списка когда решишь проблему с cookies.
SKIP_TASK_IDS = {
    "a1e91b78-d3d8-4675-bb8d-62741b4b68a6",   # Q2  — YouTube bird species
    "9d191bce-651d-4746-be2d-7ef8ecadb9c2",   # Q7  — YouTube Teal'c
}


# ──────────────────────────────────────────────
# ЗАГРУЗКА ДАТАСЕТА
# ──────────────────────────────────────────────

def load_dataset() -> list[dict]:
    with open(DATASET_PATH, encoding="utf-8") as f:
        return json.load(f)


# ──────────────────────────────────────────────
# СОХРАНЕНИЕ / ЗАГРУЗКА ОТВЕТОВ
# ──────────────────────────────────────────────

def load_answers() -> dict[str, str]:
    if Path(ANSWERS_PATH).exists():
        with open(ANSWERS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_answers(answers: dict[str, str]) -> None:
    with open(ANSWERS_PATH, "w", encoding="utf-8") as f:
        json.dump(answers, f, ensure_ascii=False, indent=2)


# ──────────────────────────────────────────────
# ПРОГОН АГЕНТА
# ──────────────────────────────────────────────

def run_agent(dataset: list[dict], agent: GAIAAgent) -> dict[str, str]:
    answers = load_answers()

    already_done  = sum(1 for q in dataset if q["task_id"] in answers)
    skipped_count = sum(1 for q in dataset if q["task_id"] in SKIP_TASK_IDS)
    if already_done:
        print(f"↩️  Пропускаю {already_done} уже решённых вопросов")
    if skipped_count:
        print(f"⏭️  Пропускаю {skipped_count} вопросов из SKIP_TASK_IDS\n")

    total = len(dataset)

    for i, item in enumerate(dataset, 1):
        task_id   = item["task_id"]
        question  = item["question"]
        file_path = item.get("file_path")

        # Пропускаем уже решённые
        if task_id in answers:
            continue

        # Пропускаем YouTube-вопросы пока не решена проблема с cookies
        if task_id in SKIP_TASK_IDS:
            print(f"⏭️  [{i}/{total}] SKIP {task_id[:8]}… (YouTube, решим позже)")
            continue

        print(f"{'─'*60}")
        print(f"[{i}/{total}] {task_id}")
        print(f"Q: {question[:120]}{'…' if len(question) > 120 else ''}")
        if file_path:
            print(f"📎 {file_path}")
        print()

        start = time.time()
        try:
            answer = agent.solve(
                question=question,
                file_path=file_path,
                task_id=task_id,
                verbose=VERBOSE,
            )
        except Exception as e:
            answer = f"ERROR: {e}"
            print(f"❌ Ошибка: {e}")

        elapsed = time.time() - start
        print(f"✅ Ответ: {answer!r}  ({elapsed:.1f}с)\n")

        answers[task_id] = answer
        save_answers(answers)

    return answers


# ──────────────────────────────────────────────
# САБМИТ
# ──────────────────────────────────────────────

def submit(answers: dict[str, str]) -> dict:
    payload = {
        "username": HF_USERNAME,
        "agent_code": HF_SPACE_URL,
        "answers": [
            {"task_id": tid, "submitted_answer": ans}
            for tid, ans in answers.items()
        ],
    }
    resp = requests.post(f"{API_BASE}/submit", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main() -> None:
    missing = []
    if not os.environ.get("OPENROUTER_API_KEY"):
        missing.append("OPENROUTER_API_KEY")
    if not os.environ.get("TAVILY_API_KEY"):
        missing.append("TAVILY_API_KEY")
    if missing:
        raise EnvironmentError(f"Не заданы переменные окружения: {', '.join(missing)}")

    dataset = load_dataset()
    print(f"📋 Загружено {len(dataset)} вопросов\n")

    agent = GAIAAgent()

    answers = run_agent(dataset, agent)

    print("=" * 60)
    print("📤 Отправляю ответы…")
    try:
        result = submit(answers)
        print(f"\n🏆 Результат:")
        print(f"   Score          : {result.get('score', '?')}%")
        print(f"   Correct        : {result.get('correct_count', '?')} / {result.get('total_attempted', '?')}")
        print(f"   Message        : {result.get('message', '')}")
        print(f"   Timestamp      : {result.get('timestamp', '')}")
    except Exception as e:
        print(f"❌ Ошибка сабмита: {e}")
        print("Ответы сохранены в answers.json — можно сабмитнуть вручную.")


if __name__ == "__main__":
    main()