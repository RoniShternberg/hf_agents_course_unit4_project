# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

A LangGraph ReAct agent targeting the 20-question GAIA Level 1 subset used by the HF Agents Course Unit 4 final project. Goal: submit ≥30% correct (≥6/20) via the course scoring API.

## Common commands

Dependency management uses `uv` (pyproject.toml + uv.lock).

```bash
uv sync                                          # install / refresh venv from lockfile

# Local evaluation harness
uv run python evaluate.py --limit 1              # smoke test on one question
uv run python evaluate.py --dump answers.jsonl   # run all 20, write JSONL, do not submit
uv run python evaluate.py --no-cache             # ignore ./cache/*.json and rerun
uv run python evaluate.py --submit \\
    --username <hf-user> \\
    --agent-code https://huggingface.co/spaces/<u>/<s>/tree/main

# Gradio UI (the HF Space entrypoint, also runs locally)
uv run python app.py
```

There is no test suite and no linter configured — validate changes with `python -m py_compile` and the `--limit 1` harness.

## Required environment

Create `.env` from `.env.example`:
- `ANTHROPIC_API_KEY` — Claude Opus 4.7 for the agent's reasoning
- `TAVILY_API_KEY` — `web_search` tool
- `OPENAI_API_KEY` — optional, only needed for `transcribe_audio`

On HF Space, set these as Space secrets (Settings → Variables and secrets). The Space runtime auto-sets `SPACE_ID`, which `app.py` uses to derive the `agent_code` URL passed to the scorer.

## Architecture

The flow is always the same: **[evaluate.py | app.py] → agent.py → tools/\* → LLM → prompts.extract_final_answer → submission payload**.

### Entry points

- [evaluate.py](evaluate.py) — CLI harness. `run_all()` fetches `/questions`, downloads any `file_name` attachment via `tools.files.fetch_file_for_task`, invokes the agent per question, caches the result to `./cache/<task_id>.json`, and (optionally) POSTs `/submit`.
- [app.py](app.py) — Gradio UI that mirrors `agents-course/Final_Assignment_Template`. Auth via `gr.LoginButton()` (HF OAuth → `profile.username`). Delegates to `evaluate.run_all` + `evaluate.submit_answers`. This is what gets deployed to the HF Space.

### Agent core ([agent.py](agent.py))

- `build_agent()` returns a `langgraph.prebuilt.create_react_agent(ChatAnthropic("claude-opus-4-7"), ALL_TOOLS)` compiled graph.
- `run_on_question(agent, question, file_path)` is the canonical invocation. It assembles `[SystemMessage(GAIA_SYSTEM_PROMPT), HumanMessage]`.
- **Images are handled natively in the HumanMessage** (base64 content blocks for Claude vision) — not exposed as a tool. Non-image files are flagged by path in the message text so the agent picks the right `read_*` tool.
- The final AI message is returned as raw text; **[prompts.py](prompts.py) `extract_final_answer`** parses out the text after `FINAL ANSWER:` and strips that literal phrase (the course's scoring API rejects submissions containing it).

### Tools ([tools/__init__.py](tools/__init__.py))

All tools are plain `@tool`-decorated functions exported via `ALL_TOOLS`. To add one: write the function in the relevant `tools/*.py`, append it to `ALL_TOOLS`, no agent/graph rewiring needed.

- [tools/web.py](tools/web.py) — `web_search` (Tavily, with `include_answer=True` and up to 6 results) and `visit_webpage` (requests + BeautifulSoup + markdownify, truncated to 8000 chars).
- [tools/files.py](tools/files.py) — downloads to `./downloads/<task_id><ext>` via `fetch_file_for_task` (harness-only helper, not a tool). Agent-visible readers: `read_pdf`, `read_excel`, `read_csv`, `read_text`, `read_json`, `read_python`, plus `download_task_file` as a fallback. All outputs are truncated to 12000 chars.
- [tools/python_exec.py](tools/python_exec.py) — `python_repl` is a persistent `PythonREPL` (state carries across tool calls within a session). Not sandboxed; runs in the host process.
- [tools/media.py](tools/media.py) — `youtube_transcript` (accepts URL or 11-char id) and `transcribe_audio` (OpenAI Whisper).

### Scoring contract (critical — from `scorer.py` research)

Do not regress these invariants:
1. Submitted answers must **not** contain the literal phrase `FINAL ANSWER` — the extractor strips it.
2. The scorer normalizes case/whitespace/punctuation for strings and strips `$`, `%`, `,` for numbers, **but** for list answers the element count must match exactly. The GAIA system prompt (verbatim from arXiv:2311.12983) is what constrains the model into the right format — changes to `prompts.GAIA_SYSTEM_PROMPT` risk dropping score.
3. API base is `https://agents-course-unit4-scoring.hf.space`. Submission payload shape: `{username, agent_code, answers: [{task_id, submitted_answer}]}`.

## Caching behavior

`evaluate.answer_question` reads/writes `./cache/<task_id>.json`. A cached record with a non-empty `submitted_answer` short-circuits re-running that question. To force a single question to re-run, delete its cache file rather than passing `--no-cache` for the whole set.

## Model / cost notes

Opus 4.7 at `temperature=0`, `max_tokens=4096`, `recursion_limit=40` per question. A full 20-question run is roughly $3–12 depending on how much tool chatter the agent produces. Tune via env vars: `GAIA_MODEL`, `GAIA_MAX_TOKENS`, `GAIA_RECURSION_LIMIT`.

## HF Space deployment

`README.md` has the required Space frontmatter (sdk: gradio, app_file: app.py). `requirements.txt` is the HF-Space-friendly mirror of `pyproject.toml` dependencies — keep them in sync when adding packages.
