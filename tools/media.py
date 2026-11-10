"""YouTube transcript and audio transcription tools.

Image analysis is handled natively by Claude (vision) and is injected into the
HumanMessage by the harness, so it is not exposed as a tool here.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from langchain_core.tools import tool

_YT_ID_RE = re.compile(
    r"(?:youtube\.com/(?:watch\?v=|embed/|v/|shorts/)|youtu\.be/)([A-Za-z0-9_-]{11})"
)


def _extract_youtube_id(url_or_id: str) -> str | None:
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url_or_id):
        return url_or_id
    m = _YT_ID_RE.search(url_or_id)
    return m.group(1) if m else None


@tool
def youtube_transcript(url_or_id: str) -> str:
    """Return the transcript for a YouTube video URL or 11-character video id.

    Prefers English; falls back to any available transcript if English is
    missing. Use this when a question references a YouTube link.
    """
    video_id = _extract_youtube_id(url_or_id)
    if not video_id:
        return f"Error: could not parse YouTube id from {url_or_id!r}"
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        return "Error: youtube-transcript-api is not installed."
    try:
        transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
        try:
            t = transcripts.find_transcript(["en", "en-US", "en-GB"])
        except Exception:  # noqa: BLE001
            t = next(iter(transcripts))
        fetched = t.fetch()
    except Exception as exc:  # noqa: BLE001
        return f"Error fetching transcript: {exc}"
    lines = [f"[{int(item['start'])}s] {item['text']}" for item in fetched]
    text = "\n".join(lines)
    if len(text) > 12000:
        text = text[:12000] + "\n…[truncated]"
    return text


@tool
def transcribe_audio(path: str) -> str:
    """Transcribe an audio file (mp3, wav, m4a, etc.) using OpenAI Whisper.

    Requires ``OPENAI_API_KEY``. If the key is absent, returns an explanatory
    error so the agent can decide whether to continue without the audio.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return (
            "Error: OPENAI_API_KEY is not set. Audio transcription requires "
            "an OpenAI API key for Whisper."
        )
    try:
        from openai import OpenAI
    except ImportError:
        return "Error: openai package is not installed."
    if not Path(path).exists():
        return f"Error: file not found at {path}"
    try:
        client = OpenAI(api_key=api_key)
        with open(path, "rb") as f:
            resp = client.audio.transcriptions.create(model="whisper-1", file=f)
    except Exception as exc:  # noqa: BLE001
        return f"Error transcribing audio: {exc}"
    return resp.text
