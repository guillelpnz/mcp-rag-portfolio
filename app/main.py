"""FastAPI layer — HTTP access to the same RAG engine the MCP server uses.

Lets anyone try the demo with curl or a browser, no MCP client required.
Ingestion runs once on startup via the lifespan handler.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

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
