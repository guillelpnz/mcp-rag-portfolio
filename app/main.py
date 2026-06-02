"""FastAPI layer — HTTP access to the same RAG engine the MCP server uses.

Lets anyone try the demo with curl or a browser, no MCP client required.
Ingestion runs once on startup via the lifespan handler.

A Gradio chat UI is mounted at `/` for the human-facing demo; the JSON API
lives at `/ask` and `/health`, and Swagger docs at `/docs`.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

import gradio as gr
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .llm import LLMError
from .rag import RAGEngine

_engine: RAGEngine | None = None


def _get_engine() -> RAGEngine:
    """Build the engine on first use (keeps import-time light and testable)."""
    global _engine
    if _engine is None:
        _engine = RAGEngine()
    return _engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        count = _get_engine().ingest()
        app.state.chunks = count
    except LLMError:
        # Don't crash the app if data is missing; /health will report it.
        app.state.chunks = 0
    yield


app = FastAPI(
    title="MCP RAG Portfolio",
    description="Ask questions about Guillermo's profile. Free, local, no API key.",
    version="1.0.0",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, examples=["What did he do at Plénitas?"])


class AskResponse(BaseModel):
    answer: str
    sources: list[str]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "indexed_chunks": getattr(app.state, "chunks", 0)}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    try:
        result = _get_engine().query(req.question)
    except LLMError as exc:
        # 503: the engine/LLM backend is the thing that failed, not the request.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return AskResponse(**result)


# --- Gradio UI -----------------------------------------------------------
# Mounted at `/` so the public demo URL shows a chat box, not raw JSON.

def _chat(question: str) -> str:
    if not question.strip():
        return "Please enter a question."
    try:
        result = _get_engine().query(question)
    except LLMError as exc:
        return f"⚠️ Backend error: {exc}"
    sources = ", ".join(result["sources"]) or "—"
    return f"{result['answer']}\n\n---\n*Sources: {sources}*"


with gr.Blocks(theme=gr.themes.Soft(), title="MCP RAG Portfolio") as _demo:
    gr.Markdown(
        "# 🤖 MCP RAG Portfolio\n"
        "Ask anything about **Guillermo's** professional experience. "
        "Answers are grounded in his CV via Retrieval-Augmented Generation — "
        "the LLM only sees the relevant chunks of his profile, no hallucinations."
    )
    _question = gr.Textbox(
        label="Your question",
        placeholder="What did Guillermo do at Plénitas?",
        lines=2,
    )
    _submit = gr.Button("Ask", variant="primary")
    _answer = gr.Markdown(label="Answer")
    gr.Examples(
        examples=[
            "What did Guillermo do at Plénitas?",
            "Tell me about his AWS and cloud experience.",
            "What is he working on right now?",
        ],
        inputs=_question,
    )
    _submit.click(_chat, inputs=_question, outputs=_answer)
    _question.submit(_chat, inputs=_question, outputs=_answer)
    gr.Markdown(
        "*Also available as a JSON API: `POST /ask` · interactive docs at `/docs` · "
        "source on [GitHub](https://github.com/guillelpnz/mcp-rag-portfolio).*"
    )

app = gr.mount_gradio_app(app, _demo, path="/")
