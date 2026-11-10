"""Local harness to fetch questions, run the agent, and optionally submit.

Typical flows:

    python evaluate.py --limit 1                    # one-question smoke test
    python evaluate.py --dump answers.jsonl         # all 20, save, no submit
    python evaluate.py --submit \\
        --username <hf-user> \\
        --agent-code https://huggingface.co/spaces/<u>/<s>/tree/main
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

from agent import build_agent, run_on_question
from prompts import extract_final_answer
from tools.files import fetch_file_for_task

load_dotenv(Path("env") / ".env")
load_dotenv()

DEFAULT_API_BASE = os.getenv(
    "GAIA_API_BASE", "https://agents-course-unit4-scoring.hf.space"
)
CACHE_DIR = Path(os.getenv("GAIA_CACHE_DIR", "./cache")).resolve()
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def fetch_questions(api_base: str = DEFAULT_API_BASE) -> list[dict[str, Any]]:
    resp = requests.get(f"{api_base.rstrip('/')}/questions", timeout=30)
    resp.raise_for_status()
    return resp.json()


def submit_answers(
    username: str,
    agent_code: str,
    answers: list[dict[str, str]],
    api_base: str = DEFAULT_API_BASE,
) -> dict[str, Any]:
    payload = {"username": username, "agent_code": agent_code, "answers": answers}
    resp = requests.post(
        f"{api_base.rstrip('/')}/submit",
        json=payload,
        timeout=120,
    )
    if not resp.ok:
        # Course API sends structured errors — surface them.
        try:
            detail = resp.json()
        except ValueError:
            detail = resp.text
        raise RuntimeError(f"Submission failed ({resp.status_code}): {detail}")
    return resp.json()


def _cache_path(task_id: str) -> Path:
    return CACHE_DIR / f"{task_id}.json"


def _load_cached(task_id: str) -> dict[str, Any] | None:
    path = _cache_path(task_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _save_cache(task_id: str, record: dict[str, Any]) -> None:
    _cache_path(task_id).write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def answer_question(
    agent, question_item: dict[str, Any], use_cache: bool = True
) -> dict[str, Any]:
    task_id = question_item["task_id"]
    if use_cache:
        cached = _load_cached(task_id)
        if cached and cached.get("submitted_answer"):
            return cached

    question = question_item.get("question") or question_item.get("Question") or ""
    file_name = question_item.get("file_name") or None
    file_path = None
    if file_name:
        try:
            file_path = fetch_file_for_task(task_id, file_name)
        except requests.RequestException as exc:
            print(f"[{task_id}] file download failed: {exc}", file=sys.stderr)

    try:
        raw = run_on_question(agent, question, file_path=file_path)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        record = {
            "task_id": task_id,
            "question": question,
            "file_name": file_name,
            "raw": "",
            "submitted_answer": "",
            "error": str(exc),
        }
        _save_cache(task_id, record)
        return record

    answer = extract_final_answer(raw)
    record = {
        "task_id": task_id,
        "question": question,
        "file_name": file_name,
        "raw": raw,
        "submitted_answer": answer,
    }
    _save_cache(task_id, record)
    return record


def run_all(
    limit: int | None = None,
    use_cache: bool = True,
) -> list[dict[str, Any]]:
    questions = fetch_questions()
    if limit is not None:
        questions = questions[:limit]
    print(f"Fetched {len(questions)} questions.", flush=True)

    agent = build_agent()
    results: list[dict[str, Any]] = []
    for i, item in enumerate(questions, start=1):
        tid = item["task_id"]
        print(f"\n=== [{i}/{len(questions)}] {tid} ===", flush=True)
        qtext = item.get("question") or item.get("Question") or ""
        print(f"Q: {qtext[:200]}{'…' if len(qtext) > 200 else ''}", flush=True)
        record = answer_question(agent, item, use_cache=use_cache)
        print(f"A: {record.get('submitted_answer', '')!r}", flush=True)
        if err := record.get("error"):
            print(f"  ! error: {err}", flush=True)
        results.append(record)
    return results


def dump_jsonl(results: list[dict[str, Any]], path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(
                json.dumps(
                    {
                        "task_id": r["task_id"],
                        "submitted_answer": r.get("submitted_answer", ""),
                        "reasoning_trace": r.get("raw", ""),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"Wrote {len(results)} rows to {out}", flush=True)


def answers_payload(results: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {"task_id": r["task_id"], "submitted_answer": r.get("submitted_answer", "")}
        for r in results
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N questions.")
    parser.add_argument("--no-cache", action="store_true", help="Ignore cache and rerun everything.")
    parser.add_argument("--dump", type=str, default=None, help="Write JSONL of answers to this path.")
    parser.add_argument("--submit", action="store_true", help="POST answers to the scoring endpoint.")
    parser.add_argument("--username", type=str, default=os.getenv("HF_USERNAME"), help="HF username for submission.")
    parser.add_argument(
        "--agent-code",
        type=str,
        default=os.getenv("GAIA_AGENT_CODE"),
        help="Public URL to the agent code (HF Space /tree/main or GitHub).",
    )
    args = parser.parse_args(argv)

    results = run_all(limit=args.limit, use_cache=not args.no_cache)

    if args.dump:
        dump_jsonl(results, args.dump)

    if args.submit:
        if not args.username or not args.agent_code:
            print("--submit requires --username and --agent-code.", file=sys.stderr)
            return 2
        payload = answers_payload(results)
        print(f"\nSubmitting {len(payload)} answers as {args.username}…", flush=True)
        try:
            outcome = submit_answers(args.username, args.agent_code, payload)
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(json.dumps(outcome, indent=2, ensure_ascii=False))
    else:
        correct = sum(1 for r in results if r.get("submitted_answer"))
        print(f"\nCompleted {correct}/{len(results)} with non-empty answers (no submission).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
