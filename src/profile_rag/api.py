"""FastAPI surface.  uv run uvicorn profile_rag.api:app

Guards: CORS allowlist, per-IP rate limit, question length cap. There is no LLM
and no tool behind /ask, so input hardening is about CPU abuse, not injection.
"""
import hashlib
import os
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from . import config
from .retrieve import answer, suggestions

RATE_PER_MINUTE = int(os.getenv("PROFILE_RAG_RATE_PER_MINUTE", "30"))
MAX_QUESTION = 500
ALLOWED_ORIGINS = [
    "https://chunduri-aditya.github.io",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    # Comma separated extras for previews and local drives, e.g. a static server.
    *[o.strip() for o in os.getenv("PROFILE_RAG_EXTRA_ORIGINS", "").split(",") if o.strip()],
]

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="profile-rag", docs_url=None, redoc_url=None)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["content-type"],
)


class Ask(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION)


def _corpus_sha() -> str:
    return hashlib.sha256(config.CORPUS.read_bytes()).hexdigest()


def _chunk_count() -> int:
    with config.CORPUS.open() as f:
        return sum(1 for line in f if line.strip())


@app.get("/health")
def health():
    return {"ok": True, "chunks": _chunk_count(), "corpus_sha256": _corpus_sha()}


@app.get("/suggestions")
def suggest():
    return {"suggestions": suggestions()}


@app.post("/ask")
@limiter.limit(f"{RATE_PER_MINUTE}/minute")
def ask(request: Request, body: Ask):
    t0 = time.perf_counter()
    out = answer(body.question)
    out["latency_ms"] = round((time.perf_counter() - t0) * 1000)
    return out
