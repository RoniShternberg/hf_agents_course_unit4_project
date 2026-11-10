"""File download and readers for GAIA task attachments."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import requests
from langchain_core.tools import tool
from pypdf import PdfReader

DOWNLOAD_DIR = Path(os.getenv("GAIA_DOWNLOAD_DIR", "./downloads")).resolve()
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_API_BASE = os.getenv(
    "GAIA_API_BASE", "https://agents-course-unit4-scoring.hf.space"
)

_MAX_CHARS = 12000


def _truncate(text: str) -> str:
    if len(text) <= _MAX_CHARS:
        return text
    return text[:_MAX_CHARS] + f"\n…[truncated, full length {len(text)} chars]"


def fetch_file_for_task(task_id: str, file_name: str | None = None) -> Path:
    """Download the file for ``task_id`` and return its local path.

    Called by the harness (not the agent) before invocation so the agent can
    reference a stable local path.
    """
    #api_base = DEFAULT_API_BASE.rstrip("/")
    #url = f"{api_base}/files/{task_id}"
    #url = f"{api_base}/files/2023/validation/{file_name}"
    url = f"https://huggingface.co/datasets/gaia-benchmark/GAIA/resolve/main/2023/validation/{file_name}"
    headers = {}
    token = os.getenv("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    resp = requests.get(url, headers=headers, timeout=60)
    resp.raise_for_status()
    suffix = Path(file_name).suffix if file_name else ""
    if not suffix:
        # Guess from Content-Disposition
        cd = resp.headers.get("Content-Disposition", "")
        if "filename=" in cd:
            guessed = cd.split("filename=")[-1].strip("\"' ")
            suffix = Path(guessed).suffix
    path = DOWNLOAD_DIR / f"{task_id}{suffix}"
    path.write_bytes(resp.content)
    return path


@tool
def download_task_file(task_id: str) -> str:
    """Download the file attached to a GAIA task and return the local path.

    Only needed if the harness did not pre-download it. Prefer using the path
    already provided in the question context when available.
    """
    try:
        path = fetch_file_for_task(task_id)
    except requests.RequestException as exc:
        return f"Error downloading task file: {exc}"
    return str(path)


@tool
def read_pdf(path: str) -> str:
    """Extract text from a PDF file at ``path``."""
    try:
        reader = PdfReader(path)
    except Exception as exc:  # noqa: BLE001
        return f"Error opening PDF: {exc}"
    pages: list[str] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            pages.append(f"--- page {i} ---\n{page.extract_text() or ''}")
        except Exception as exc:  # noqa: BLE001
            pages.append(f"--- page {i} ---\n[error extracting: {exc}]")
    return _truncate("\n".join(pages).strip() or "[PDF had no extractable text]")


@tool
def read_excel(path: str, sheet_name: str | None = None) -> str:
    """Read an Excel (.xlsx/.xls) file and return a CSV-like string.

    If ``sheet_name`` is None, all sheets are concatenated with headers.
    """
    try:
        sheets = pd.read_excel(path, sheet_name=sheet_name)
    except Exception as exc:  # noqa: BLE001
        return f"Error reading Excel: {exc}"
    if isinstance(sheets, dict):
        parts = [f"# sheet: {name}\n{df.to_csv(index=False)}" for name, df in sheets.items()]
        return _truncate("\n\n".join(parts))
    return _truncate(sheets.to_csv(index=False))


@tool
def read_csv(path: str) -> str:
    """Read a CSV file and return its contents as text."""
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        return f"Error reading CSV: {exc}"
    return _truncate(df.to_csv(index=False))


@tool
def read_text(path: str) -> str:
    """Read a UTF-8 text file at ``path``. Works for .txt, .md, .log, etc."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"Error reading file: {exc}"
    return _truncate(text)


@tool
def read_json(path: str) -> str:
    """Read a JSON file and return a pretty-printed representation."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return f"Error reading JSON: {exc}"
    return _truncate(json.dumps(data, indent=2, ensure_ascii=False))


@tool
def read_python(path: str) -> str:
    """Read a Python source file at ``path``."""
    return read_text.invoke({"path": path})
