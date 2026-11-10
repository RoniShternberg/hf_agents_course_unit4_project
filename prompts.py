"""GAIA system prompt and answer extraction.

The system prompt is verbatim from the GAIA paper (arXiv:2311.12983) — the
scorer is calibrated against answers produced under this prompt. The extractor
pulls the text after the final ``FINAL ANSWER:`` and strips the literal phrase
before submission (course requires the submitted value to exclude that phrase).
"""

from __future__ import annotations

import re

GAIA_SYSTEM_PROMPT = (
    "You are a general AI assistant. I will ask you a question. Report your "
    "thoughts, and finish your answer with the following template: "
    "FINAL ANSWER: [YOUR FINAL ANSWER]. YOUR FINAL ANSWER should be a number "
    "OR as few words as possible OR a comma separated list of numbers and/or "
    "strings. If you are asked for a number, don't use comma to write your "
    "number neither use units such as $ or percent sign unless specified "
    "otherwise. If you are asked for a string, don't use articles, neither "
    "abbreviations (e.g. for cities), and write the digits in plain text "
    "unless specified otherwise. If you are asked for a comma separated list, "
    "apply the above rules depending of whether the element to be put in the "
    "list is a number or a string."
)

_FINAL_ANSWER_RE = re.compile(r"FINAL ANSWER:\s*(.+?)\s*$", re.IGNORECASE | re.DOTALL)


def extract_final_answer(text: str) -> str:
    """Return the submittable answer.

    Prefers the last ``FINAL ANSWER:`` match; falls back to the last non-empty
    line so we never submit an empty string.
    """
    if not text:
        return ""
    matches = list(_FINAL_ANSWER_RE.finditer(text))
    if matches:
        answer = matches[-1].group(1).strip()
    else:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        answer = lines[-1] if lines else text.strip()
    # Defensive: strip stray "FINAL ANSWER:" prefix, surrounding quotes,
    # trailing period only when the answer is not a decimal number.
    answer = re.sub(r"^FINAL ANSWER:\s*", "", answer, flags=re.IGNORECASE).strip()
    answer = answer.strip("\"'")
    if answer.endswith(".") and not re.search(r"\d\.\d", answer):
        answer = answer[:-1].strip()
    return answer
