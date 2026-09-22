"""FastAPI surface.  uv run uvicorn profile_rag.api:app"""
import time

from fastapi import FastAPI
from pydantic import BaseModel

from .retrieve import answer, suggestions

app = FastAPI(title="profile-rag")


class Ask(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/suggestions")
def suggest():
    return {"suggestions": suggestions()}


@app.post("/ask")
def ask(body: Ask):
    t0 = time.perf_counter()
    out = answer(body.question)
    out["latency_ms"] = round((time.perf_counter() - t0) * 1000)
    return out
