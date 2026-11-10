"""Tool registry for the GAIA agent."""

from __future__ import annotations

from .files import (
    download_task_file,
    read_csv,
    read_excel,
    read_json,
    read_pdf,
    read_python,
    read_text,
)
from .media import transcribe_audio, youtube_transcript
from .python_exec import python_repl
from .web import visit_webpage, web_search

ALL_TOOLS = [
    web_search,
    visit_webpage,
    python_repl,
    download_task_file,
    read_pdf,
    read_excel,
    read_csv,
    read_text,
    read_json,
    read_python,
    youtube_transcript,
    transcribe_audio,
]

__all__ = ["ALL_TOOLS"]
