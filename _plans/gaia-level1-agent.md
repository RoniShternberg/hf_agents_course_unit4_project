# Plan: HF Agents Course Unit 4 — GAIA Level 1 Agent

## Context

This project is the final deliverable for the Hugging Face Agents Course Unit 4. The goal is an AI agent that scores **≥30%** on a 20-question subset of the GAIA Level 1 benchmark (≥6 correct) and can be submitted through the course's scoring API.

GAIA Level 1 questions typically require <5 steps, but demand real tool use: web search, page fetching, Python computation, file reading (PDF/Excel/CSV/text), YouTube transcripts, and occasionally audio/image parsing. Standalone LLMs fail these because the answers depend on up-to-date retrieval and multi-modal content the model can't know from training data alone.

The working directory (`c:\Users\ronis\main\repositories\hf_agents_course_unit4_project`) is currently empty — this plan scaffolds the entire project from scratch.

**User decisions (confirmed):**
- Framework: **LangGraph** (user already uses it in `../ai_agent_project`)
- LLM: **Anthropic Claude Opus 4.7** (`claude-opus-4-7`)
- Deployment: **HF Space + local dev** (local for iteration, Space for the required public `agent_code` URL)
- Tools: web search + page fetch, Python exec, file readers, YouTube/audio/image

**Scoring API (confirmed by research):**
- Base URL: `https://agents-course-unit4-scoring.hf.space`
- `GET /questions` → `[{task_id, question, file_name?, ...}]`
- `GET /files/{task_id}` → binary attachment (when `file_name` is present)
- `POST /submit` with body `{username, agent_code, answers:[{task_id, submitted_answer}]}`
- Scorer normalizes whitespace/punctuation/case for strings; strips `$ % ,` for numbers; list answers must match element count exactly.
- Submission should NOT contain the literal text "FINAL ANSWER" — strip it before sending.

---

## Architecture

```
hf_agents_course_unit4_project/
├── pyproject.toml              # uv/pip deps (matches style of ai_agent_project)
├── requirements.txt            # mirror for HF Space
├── .env.example                # ANTHROPIC_API_KEY, TAVILY_API_KEY, HF_TOKEN
├── .gitignore
├── README.md                   # HF Space frontmatter + usage
├── app.py                      # Gradio UI: login → run → submit
├── agent.py                    # LangGraph ReAct agent factory
├── prompts.py                  # GAIA system prompt + FINAL ANSWER extractor
├── tools/
│   ├── __init__.py             # exports TOOLS list
│   ├── web.py                  # web_search (Tavily), visit_webpage
│   ├── files.py                # download_task_file, read_pdf/excel/csv/text/json/python
│   ├── python_exec.py          # sandboxed PythonREPL tool
│   └── media.py                # youtube_transcript, transcribe_audio, analyze_image
└── evaluate.py                 # local harness: fetch qs → run agent → dump JSONL → POST /submit
```

### Agent design (LangGraph)

Use `langgraph.prebuilt.create_react_agent` — it's the standard LangGraph pattern for tool-using agents and keeps the graph implicit. The agent's `ChatAnthropic(model="claude-opus-4-7")` is bound to the tool list.

Claude Opus 4.7 is natively multimodal, so **images are passed as `image` content parts directly in the HumanMessage**, not via a separate tool. The `analyze_image` "tool" is really a file-loader that returns a ready-to-use content block.

Execution flow per question:
1. `evaluate.py` fetches one question + downloads file if `file_name` is set.
2. Build an initial `HumanMessage`. If a file is present, append either: image content blocks (Claude vision), or a system note like "An attached file has been saved to `./downloads/<task_id>_<name>`; use `read_*` tools to read it."
3. Invoke the agent. The GAIA system prompt (from the paper) instructs the model to emit `FINAL ANSWER: <value>`.
4. Post-process: extract everything after the last `FINAL ANSWER:`, strip the literal phrase, trim, submit as `submitted_answer`.

### Tools (critical files & what they use)

| Tool | File | Library |
|------|------|---------|
| `web_search(query)` | [tools/web.py](../tools/web.py) | `tavily-python` (better than DDG for factual Qs; free tier 1k/mo) |
| `visit_webpage(url)` | [tools/web.py](../tools/web.py) | `requests` + `beautifulsoup4` + `markdownify` (return markdown ≤8000 chars) |
| `python_repl(code)` | [tools/python_exec.py](../tools/python_exec.py) | `langchain_experimental.utilities.PythonREPL` |
| `download_task_file(task_id)` | [tools/files.py](../tools/files.py) | `requests` → saves to `./downloads/` |
| `read_pdf(path)` | [tools/files.py](../tools/files.py) | `pypdf` |
| `read_excel(path, sheet?)` | [tools/files.py](../tools/files.py) | `pandas` + `openpyxl` |
| `read_csv(path)` / `read_text(path)` / `read_json(path)` | [tools/files.py](../tools/files.py) | stdlib / `pandas` |
| `youtube_transcript(url)` | [tools/media.py](../tools/media.py) | `youtube-transcript-api` |
| `transcribe_audio(path)` | [tools/media.py](../tools/media.py) | OpenAI Whisper API (optional — skip if no key; image/text Level 1 is most common) |
| image attachments | handled inline in `evaluate.py` | native Claude vision |

All tools use LangChain's `@tool` decorator so `create_react_agent` can bind them cleanly.

### System prompt (prompts.py)

Exact text from the GAIA paper verbatim — the model must emit `FINAL ANSWER: …` so the extractor works. Answer extractor regex: `r"FINAL ANSWER:\s*(.+?)\s*$"` (DOTALL, pick last match), then strip.

### Gradio app (app.py)

Mirror the template at `agents-course/Final_Assignment_Template` for compatibility:
- `gr.LoginButton()` → HF OAuth → `profile.username`
- "Run & Submit" button: calls `evaluate.run_all(username, agent_code_url)` → displays score + per-question result table.
- `agent_code` is the HF Space tree URL, e.g. `https://huggingface.co/spaces/<user>/<space>/tree/main`.

### Local harness (evaluate.py)

Can run standalone (no Gradio) for fast iteration:
```
python evaluate.py --limit 5              # sanity
python evaluate.py --dump answers.jsonl   # run all, save, don't submit
python evaluate.py --submit               # run all + POST /submit
```
Caches per-task answers to `./cache/<task_id>.json` so reruns skip solved questions; useful when tuning a single failing case.

### Dependencies (pyproject.toml)

```
langgraph>=0.2
langchain>=0.3
langchain-anthropic>=0.3
langchain-core>=0.3
langchain-community>=0.3
langchain-experimental>=0.3
python-dotenv
requests
beautifulsoup4
markdownify
pypdf
openpyxl
pandas
tavily-python
youtube-transcript-api
gradio>=4.44
```
(`requirements.txt` mirrors the runtime subset for HF Space.)

### HF Space deployment

`README.md` frontmatter:
```yaml
---
title: GAIA Level 1 Agent
emoji: 🧭
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---
```
Space secrets: `ANTHROPIC_API_KEY`, `TAVILY_API_KEY`. Hardware: CPU Basic (free) is sufficient — Claude runs remotely.

---

## Implementation order

1. **Scaffold**: `pyproject.toml`, `.env.example`, `.gitignore`, `README.md` frontmatter.
2. **Tools**: `tools/web.py`, `tools/files.py`, `tools/python_exec.py`, `tools/media.py` — each standalone, testable in isolation.
3. **Agent**: `prompts.py` + `agent.py` wiring `create_react_agent` with Claude Opus 4.7.
4. **Harness**: `evaluate.py` — fetch, run one, print answer. Iterate on a single question until stable.
5. **Full run**: `evaluate.py --dump` against all 20, review failures, adjust prompts/tools.
6. **Gradio**: `app.py` with login + submit button. Test locally with `gradio app.py`.
7. **Deploy**: git init, push to HF Space, configure secrets, run from Space UI, submit.

---

## Verification

- **Per-tool smoke test** (manual one-liner for each): call `web_search("capital of France")`, `visit_webpage("https://example.com")`, `read_pdf` on a sample PDF, `youtube_transcript("dQw4w9WgXcQ")`.
- **End-to-end single question**: `python evaluate.py --limit 1` — confirm the agent returns a non-empty `submitted_answer` and the `FINAL ANSWER:` phrase is stripped.
- **Format test against scorer**: construct a fake payload with known-good answers (numbers, strings, lists) and verify normalization expectations. In particular, confirm list answers are comma-separated, no trailing units, no articles for strings.
- **Full-run target**: score **≥30%** (6/20). If below, inspect the cached JSON for each wrong task_id and iterate on tools/prompts. Most-likely failure modes: (a) web_search not finding the right page → add a retry/site-specific fallback; (b) file attachment not read correctly → verify `download_task_file` saw the right extension; (c) answer format wrong (extra words, wrong pluralization) → tighten the extractor or add few-shot examples.
- **Final submission**: run from the deployed HF Space (so `agent_code` URL is live and public), record score.

---

## Risks & notes

- **Cost**: Opus 4.7 × ReAct × 20 questions ≈ $3–12 per full run. Budget accordingly; cache aggressively.
- **Rate limits**: Tavily free tier is 1k searches/month — plenty. Anthropic has per-minute RPM limits; add simple sequential execution (no parallelism) in the harness.
- **Deterministic scoring**: The scorer does case/whitespace normalization, but for lists and numbers, formatting mistakes are fatal. The GAIA system prompt handles most of this if obeyed strictly.
- **File handling**: Not every question has a file. `file_name` field on `/questions` items signals attachment — only download when present.
