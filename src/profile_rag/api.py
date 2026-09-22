"""FastAPI surface.  uv run uvicorn profile_rag.api:app"""
from fastapi import FastAPI
from pydantic import BaseModel

from .retrieve import search

app = FastAPI(title="profile-rag")


class Ask(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/ask")
def ask(body: Ask):
    hits = search(body.question)
    return {
        "answer": hits[0].node.get_content() if hits else None,
        "sources": [
            {"id": h.node.node_id, "score": h.score, **h.node.metadata} for h in hits
        ],
        "mode": "extract",
    }
