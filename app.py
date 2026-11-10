"""Gradio UI for the GAIA Level 1 agent — deployable to HF Spaces.

Mirrors the course-provided ``agents-course/Final_Assignment_Template`` flow:
log in with Hugging Face, click "Run & Submit", get a score back. The actual
agent logic lives in ``agent.py`` and the fetch/submit plumbing in
``evaluate.py`` — this file just wires them to a Gradio front-end.
"""

from __future__ import annotations

import os
import traceback

import gradio as gr
import pandas as pd
from pathlib import Path

from dotenv import load_dotenv

from evaluate import answers_payload, fetch_questions, run_all, submit_answers

load_dotenv(Path("env") / ".env")
load_dotenv()

SPACE_ID = os.getenv("SPACE_ID")  # set automatically by HF Spaces runtime


def _default_agent_code() -> str:
    if SPACE_ID:
        return f"https://huggingface.co/spaces/{SPACE_ID}/tree/main"
    return os.getenv("GAIA_AGENT_CODE", "")


def run_and_submit(profile: gr.OAuthProfile | None):
    if profile is None:
        return "Please log in with Hugging Face first.", None

    username = profile.username
    agent_code = _default_agent_code()
    if not agent_code:
        return (
            "Could not determine agent_code URL. Set SPACE_ID (auto on Spaces) "
            "or GAIA_AGENT_CODE env var.",
            None,
        )

    try:
        results = run_all()
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return f"Error running agent: {exc}", None

    table = pd.DataFrame(
        [
            {
                "task_id": r["task_id"],
                "question": (r.get("question") or "")[:120],
                "submitted_answer": r.get("submitted_answer", ""),
                "error": r.get("error", ""),
            }
            for r in results
        ]
    )

    try:
        outcome = submit_answers(username, agent_code, answers_payload(results))
    except RuntimeError as exc:
        return f"Answers computed, but submission failed:\n{exc}", table

    message = (
        f"Submission complete!\n"
        f"User: {outcome.get('username')}\n"
        f"Score: {outcome.get('score')}%  "
        f"({outcome.get('correct_count')} / {outcome.get('total_attempted')})\n"
        f"Message: {outcome.get('message', '')}"
    )
    return message, table


def preview_questions():
    try:
        qs = fetch_questions()
    except Exception as exc:  # noqa: BLE001
        return f"Error fetching questions: {exc}"
    return pd.DataFrame(
        [
            {
                "task_id": q["task_id"],
                "file_name": q.get("file_name") or "",
                "question": (q.get("question") or q.get("Question") or "")[:200],
            }
            for q in qs
        ]
    )


with gr.Blocks(title="GAIA Level 1 Agent") as demo:
    gr.Markdown(
        """
        # GAIA Level 1 Agent — HF Agents Course Unit 4

        LangGraph ReAct agent on Claude Opus 4.7 with web search, page fetch,
        Python exec, file readers, and YouTube/audio/vision tools.

        1. Log in with your Hugging Face account (required — your username is
           submitted with the answers).
        2. Click **Run & Submit**. The agent will answer all 20 questions and
           submit to the course scoring endpoint. This takes several minutes
           and uses your Anthropic API quota.
        """
    )

    gr.LoginButton()

    with gr.Row():
        preview_btn = gr.Button("Preview questions", variant="secondary")
        run_btn = gr.Button("Run & Submit", variant="primary")

    status = gr.Markdown(label="Status")
    table = gr.DataFrame(label="Results", wrap=True)

    preview_btn.click(fn=preview_questions, outputs=table)
    run_btn.click(fn=run_and_submit, outputs=[status, table])


if __name__ == "__main__":
    demo.launch(debug=False)
