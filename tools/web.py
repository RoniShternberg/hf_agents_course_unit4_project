"""Web search and webpage fetching tools."""

from __future__ import annotations

import os
from typing import Any

import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from markdownify import markdownify as html_to_md

_MAX_PAGE_CHARS = 8000
_MAX_SEARCH_RESULTS = 6
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


@tool
def web_search(query: str) -> str:
    """Search the web for factual information.

    Returns up to six results with title, URL, and a short snippet.
    Prefer this as the first step when a question asks for external facts.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return (
            "Error: TAVILY_API_KEY is not set. Configure the key in the "
            "environment to use web_search."
        )
    try:
        from tavily import TavilyClient
    except ImportError:
        return "Error: tavily-python is not installed."

    client = TavilyClient(api_key=api_key)
    try:
        resp: Any = client.search(
            query=query,
            max_results=_MAX_SEARCH_RESULTS,
            search_depth="advanced",
            include_answer=True,
        )
    except Exception as exc:  # noqa: BLE001 — surface API failure to the agent
        return f"Error from Tavily: {exc}"

    parts: list[str] = []
    if answer := resp.get("answer"):
        parts.append(f"Quick answer: {answer}\n")
    for i, item in enumerate(resp.get("results", []), start=1):
        title = item.get("title", "").strip()
        url = item.get("url", "").strip()
        snippet = (item.get("content") or "").strip().replace("\n", " ")
        if len(snippet) > 400:
            snippet = snippet[:400] + "…"
        parts.append(f"[{i}] {title}\n{url}\n{snippet}")
    return "\n\n".join(parts) if parts else "No results."


@tool
def visit_webpage(url: str) -> str:
    """Fetch a webpage and return its main text content as markdown.

    Use this after `web_search` to read a specific result. Output is truncated
    to 8000 characters; request a different URL or use Python to fetch a
    specific section if more is needed.
    """
    try:
        resp = requests.get(
            url,
            timeout=20,
            headers={"User-Agent": _USER_AGENT},
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        return f"Error fetching {url}: {exc}"

    content_type = resp.headers.get("Content-Type", "")
    if "html" not in content_type and "xml" not in content_type:
        text = resp.text
        return text[:_MAX_PAGE_CHARS] + ("\n…[truncated]" if len(text) > _MAX_PAGE_CHARS else "")

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "iframe"]):
        tag.decompose()
    main = soup.find("main") or soup.find("article") or soup.body or soup
    md = html_to_md(str(main), heading_style="ATX").strip()
    md = "\n".join(line for line in md.splitlines() if line.strip())
    if len(md) > _MAX_PAGE_CHARS:
        md = md[:_MAX_PAGE_CHARS] + "\n…[truncated]"
    return md
