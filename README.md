---
title: MCP RAG Portfolio
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
license: mit
---

# MCP RAG Portfolio

> Ask an LLM questions about my CV and LinkedIn profile — answered with RAG over a
> local vector store, exposed both as an **MCP server** and a **FastAPI** endpoint.
> **100% free to run: local models via Ollama, local embeddings, no API key.**

![CI](https://github.com/guillelpnz/mcp-rag-portfolio/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Why this exists

A production-minded personal project exploring an **MCP server backed by a RAG
pipeline**. It indexes my professional profile so an MCP client or plain `curl`
can ask grounded questions about my experience. The same engine powers both the
MCP tools and the HTTP API: one retrieval path, two front doors.

This repository is separate from my production MCP work. Its ChromaDB and local
embedding implementation should not be interpreted as the storage architecture
used in that production system.

## Architecture

```
                 ┌──────────────────┐
   MCP client ──▶│  MCP server      │
 (Claude, IDE)   │  app/mcp_server  │─┐
                 └──────────────────┘ │     ┌──────────────────┐     ┌────────────┐
                                      ├────▶│   RAG engine     │────▶│  ChromaDB  │
                 ┌──────────────────┐ │     │   app/rag.py     │     │ (vectors)  │
   curl / web ──▶│  FastAPI         │─┘     │                  │     └────────────┘
                 │  app/main.py     │       │  retrieve top-k  │
                 └──────────────────┘       │  + build prompt  │     ┌────────────┐
                                            └────────┬─────────┘────▶│ LLM provider│
                                                     │               │  Ollama /   │
                          embeddings (local,         │               │  Bedrock /  │
                          sentence-transformers) ────┘               │  OpenAI     │
                                                                     └────────────┘
```

- **Retrieval**: profile docs are chunked, embedded locally with
  `all-MiniLM-L6-v2`, and stored in ChromaDB.
- **Generation**: a pluggable `LLMProvider` (Ollama by default) writes the final
  answer using only the retrieved context.
- **Factual guardrails**: known high-risk false premises are corrected
  deterministically before generation.
- **Swappable backends**: change one env var to go from local Ollama to AWS Bedrock
  or OpenAI — no code changes.

## Quickstart (one command, free)

```bash
docker compose up
```

This starts Ollama (pulls `llama3.2` on first run) and the API. Then:

```bash
# Health + how many chunks were indexed
curl localhost:8000/health

# Ask about my profile
curl -X POST localhost:8000/ask \
  -H "content-type: application/json" \
  -d '{"question": "What did Guillermo do at Plénitas?"}'
```

Interactive docs at **http://localhost:8000/docs**.

## Run the MCP server

For an MCP client over stdio:

```bash
pip install -r requirements.txt
python -m app.mcp_server
```

Exposed tools:
- **`ask_profile(question)`** — grounded answer about my CV/experience.
- **`list_sources()`** — the profile documents currently indexed.

## Use a cloud LLM instead (optional)

```bash
# AWS Bedrock (needs boto3 + AWS creds)
LLM_PROVIDER=bedrock docker compose up

# OpenAI (needs openai + OPENAI_API_KEY)
LLM_PROVIDER=openai docker compose up
```

## Tech stack

**Python · FastAPI · MCP · ChromaDB · sentence-transformers · Ollama · Docker · Pytest**

## Tests

```bash
pytest -v
```

Tests run fully offline (the LLM provider is mocked) and cover chunking, the
provider factory, error paths, and the API endpoints. CI runs them on every push.

## Project layout

```
app/
  main.py            FastAPI app (HTTP front door)
  mcp_server.py      MCP server (tools front door)
  rag.py             chunk → embed → retrieve → generate
  llm/
    base.py          LLMProvider protocol + LLMError
    ollama_provider.py    free local default
    cloud_providers.py    optional Bedrock / OpenAI
    __init__.py      provider factory (LLM_PROVIDER)
data/
  profile.md         the knowledge base (my CV / profile)
tests/               offline test suite
```

## License

MIT — see [LICENSE](LICENSE).
