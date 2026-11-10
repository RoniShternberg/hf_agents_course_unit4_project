---
title: GAIA Level 1 Agent
emoji: 🧭
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# GAIA Level 1 Agent — HF Agents Course Unit 4

A LangGraph ReAct agent built around Anthropic Claude Opus 4.7, equipped for the
20-question GAIA Level 1 subset used in the Hugging Face Agents Course final project.

## Local usage

```bash
cp .env.example .env    # fill in ANTHROPIC_API_KEY and TAVILY_API_KEY
pip install -r requirements.txt

# Single-question smoke test
python evaluate.py --limit 1

# Run all 20, save answers, do not submit
python evaluate.py --dump answers.jsonl

# Run all 20 and submit
python evaluate.py --submit --username <hf-username> --agent-code <space-url>
```

## HF Space

`app.py` is a Gradio front-end that logs in with HF OAuth, runs the agent against
all fetched questions, and POSTs to the course scoring endpoint.

Required Space secrets: `ANTHROPIC_API_KEY`, `TAVILY_API_KEY`.

## Layout

- `agent.py` — LangGraph ReAct agent factory.
- `prompts.py` — GAIA system prompt and final-answer extractor.
- `tools/` — web search, webpage fetch, file readers, Python REPL, media tools.
- `evaluate.py` — local harness (fetch → run → cache → submit).
- `app.py` — Gradio UI mirroring the course template.
