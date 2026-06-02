"""FastAPI layer — HTTP access to the same RAG engine the MCP server uses.

Serves a minimal static landing page at `/` for the human-facing demo. The
JSON API lives at `/ask` and `/health`, with Swagger docs at `/docs`.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .llm import LLMError
from .rag import RAGEngine

_engine: RAGEngine | None = None
_STATIC_DIR = Path(__file__).parent / "static"


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
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, examples=["What did he do at Plénitas?"])


class AskResponse(BaseModel):
    answer: str
    sources: list[str]


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


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
