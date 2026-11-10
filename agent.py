"""LangGraph ReAct agent factory for GAIA Level 1.

The agent wraps Anthropic Claude Opus 4.7 with the tool set in ``tools/`` and
uses the verbatim GAIA system prompt so answers arrive in the format the
course scorer expects.

Entry points:
- ``build_agent()`` → compiled LangGraph runnable.
- ``run_on_question(agent, question, file_path=None)`` → str (raw final message)
"""

from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from prompts import GAIA_SYSTEM_PROMPT
from tools import ALL_TOOLS

DEFAULT_MODEL = os.getenv("GAIA_MODEL", "claude-opus-4-7")
DEFAULT_MAX_TOKENS = int(os.getenv("GAIA_MAX_TOKENS", "4096"))
DEFAULT_RECURSION_LIMIT = int(os.getenv("GAIA_RECURSION_LIMIT", "40"))

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}


def build_agent(model: str = DEFAULT_MODEL):
    """Return a compiled LangGraph ReAct agent bound to ``model``."""
    llm = ChatAnthropic(
        model=model,
        max_tokens=DEFAULT_MAX_TOKENS,
    )
    return create_react_agent(llm, ALL_TOOLS)


def _image_content_block(path: Path) -> dict[str, Any]:
    mime, _ = mimetypes.guess_type(str(path))
    if not mime:
        mime = "image/png"
    data = base64.standard_b64encode(path.read_bytes()).decode("ascii")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": mime, "data": data},
    }


def _build_human_message(question: str, file_path: Path | None) -> HumanMessage:
    if file_path is None:
        return HumanMessage(content=question)

    suffix = file_path.suffix.lower()
    if suffix in _IMAGE_EXTS:
        # Inline the image for Claude's vision.
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    f"{question}\n\n"
                    "An image file is attached below. Use it directly to answer."
                ),
            },
            _image_content_block(file_path),
        ]
        return HumanMessage(content=content)

    # Non-image file — tell the agent where it lives so it can use a read_* tool.
    return HumanMessage(
        content=(
            f"{question}\n\n"
            f"An attached file has been saved locally to: {file_path}\n"
            f"(file name: {file_path.name}). Use the appropriate read_* tool "
            "(read_pdf, read_excel, read_csv, read_text, read_json, read_python) "
            "to inspect it."
        )
    )


def run_on_question(
    agent,
    question: str,
    file_path: Path | None = None,
    recursion_limit: int = DEFAULT_RECURSION_LIMIT,
) -> str:
    """Invoke the agent and return the final AI message text.

    The extracted GAIA answer is derived from this string by
    ``prompts.extract_final_answer``.
    """
    messages = [
        SystemMessage(content=GAIA_SYSTEM_PROMPT),
        _build_human_message(question, file_path),
    ]
    state = agent.invoke(
        {"messages": messages},
        config={"recursion_limit": recursion_limit},
    )
    final = state["messages"][-1]
    if isinstance(final, AIMessage):
        content = final.content
    else:
        content = getattr(final, "content", "")
    if isinstance(content, list):
        # Claude sometimes returns a list of content blocks.
        parts = [block.get("text", "") for block in content if isinstance(block, dict)]
        return "\n".join(p for p in parts if p).strip()
    return str(content).strip()
