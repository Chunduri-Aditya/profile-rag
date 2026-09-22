import pytest
from fastapi.testclient import TestClient

from profile_rag.api import app
from profile_rag.eval import run
from profile_rag.retrieve import answer, candidates, faq_match, rerank


def test_rerank_puts_overview_first_for_what_is():
    q = "what is Agent Shield"
    fused = candidates(q, mode="hybrid")
    assert fused[0].node.node_id != "project/agent-shield/overview", "fusion already right; test proves nothing"
    top = rerank(q, fused)[0]
    assert top.node.node_id == "project/agent-shield/overview"


def test_rerank_is_order_independent():
    q = "how does RemixMate detect downbeats"
    fused = candidates(q, mode="hybrid")
    a = [n.node.node_id for n in rerank(q, fused)]
    b = [n.node.node_id for n in rerank(q, list(reversed(fused)))]
    assert a == b


def test_faq_hit_and_miss():
    hit = faq_match("where did aditya go to school?")
    assert hit is not None and hit["source"].startswith("education/")
    assert faq_match("what is the capital of France") is None


def test_answer_extracts_from_a_source_chunk():
    out = answer("why was the tools anchor withdrawn")
    assert out["mode"] == "extract"
    assert 1 <= out["answer"].count(".") <= 4
    assert any(out["answer"][:40] in s["text"] for s in out["sources"])


def test_answer_falls_back_off_topic():
    out = answer("give me a recipe for banana bread")
    assert out["mode"] == "fallback"
    assert out["suggestions"]


def test_api_ask_shape():
    r = TestClient(app).post("/ask", json={"question": "what is jarvis"})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] in {"faq", "extract"}
    assert body["sources"][0]["url"].startswith("https://chunduri-aditya.github.io/Portfolio/")


@pytest.mark.slow
def test_eval_floor():
    out = run(modes=("bm25", "hybrid+rerank"))
    by = {r["mode"]: r for r in out["rows"]}
    assert by["hybrid+rerank"]["recall@3"] >= 0.97  # ratchet: 0.979 on 2026-09-22; rerank removed reads 0.958
    assert by["hybrid+rerank"]["recall@3"] >= by["bm25"]["recall@3"]
