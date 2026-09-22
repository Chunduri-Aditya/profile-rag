from fastapi.testclient import TestClient

from profile_rag import api

ORIGIN = "https://chunduri-aditya.github.io"


def client():
    return TestClient(api.app)


def test_health_reports_index_identity():
    body = client().get("/health").json()
    assert body["ok"] is True
    assert body["chunks"] > 300
    assert len(body["corpus_sha256"]) == 64


def test_question_length_allow_and_deny():
    c = client()
    assert c.post("/ask", json={"question": "x" * 500}).status_code == 200
    assert c.post("/ask", json={"question": "x" * 501}).status_code == 422


def test_cors_allows_portfolio_and_blocks_others():
    c = client()
    ok = c.options("/ask", headers={"Origin": ORIGIN, "Access-Control-Request-Method": "POST"})
    assert ok.headers.get("access-control-allow-origin") == ORIGIN
    bad = c.options("/ask", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in bad.headers


def test_rate_limit_allow_then_deny():
    api.limiter.reset()
    c = TestClient(api.app)
    codes = [c.post("/ask", json={"question": "what is jarvis"}).status_code for _ in range(api.RATE_PER_MINUTE + 1)]
    assert codes[: api.RATE_PER_MINUTE] == [200] * api.RATE_PER_MINUTE
    assert codes[-1] == 429
    api.limiter.reset()
