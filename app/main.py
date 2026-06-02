"""FastAPI layer — HTTP access to the same RAG engine the MCP server uses.

Serves a minimal static landing page at `/` for the human-facing demo. The
JSON API lives at `/ask` and `/health`, with Swagger docs at `/docs`.

Public-internet hardening: `/ask` is rate-limited per client IP and
answers are cached in-memory for 24h, so a burst of identical questions
or an abusive client can't drain the upstream LLM quota.
"""
from __future__ import annotations

import hashlib
from contextlib import asynccontextmanager
from pathlib import Path

from cachetools import TTLCache
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded

from .llm import LLMError
from .rag import RAGEngine

_engine: RAGEngine | None = None
_STATIC_DIR = Path(__file__).parent / "static"

# Cache hits never touch the upstream LLM, so they're the strongest defence
# against repeated identical prompts (the most likely abuse pattern).
_answer_cache: TTLCache = TTLCache(maxsize=500, ttl=24 * 60 * 60)


def _get_engine() -> RAGEngine:
    """Build the engine on first use (keeps import-time light and testable)."""
    global _engine
    if _engine is None:
        _engine = RAGEngine()
    return _engine


def _client_ip(request: Request) -> str:
    """Read the real client IP. HF Spaces sits behind a proxy, so the direct
    socket is the proxy — the actual client lands in X-Forwarded-For."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _cache_key(question: str) -> str:
    return hashlib.sha256(question.strip().lower().encode()).hexdigest()


limiter = Limiter(key_func=_client_ip, default_limits=[])


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
app.state.limiter = limiter
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Too many requests. Please slow down."},
    )


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
@limiter.limit("10/minute;200/day")
def ask(request: Request, req: AskRequest) -> AskResponse:
    key = _cache_key(req.question)
    cached = _answer_cache.get(key)
    if cached is not None:
        return AskResponse(**cached)
    try:
        result = _get_engine().query(req.question)
    except LLMError as exc:
        # 503: the engine/LLM backend is the thing that failed, not the request.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    _answer_cache[key] = result
    return AskResponse(**result)
