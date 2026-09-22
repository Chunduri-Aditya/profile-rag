from fastapi.testclient import TestClient

from profile_rag.api import app
from profile_rag.build import build


def test_thin_path():
    assert build() > 100
    r = TestClient(app).post("/ask", json={"question": "what is Agent Shield"})
    assert r.status_code == 200, r.text
    top = r.json()["sources"][0]
    print("TOP:", top["id"], top["score"])
    assert top["id"].startswith("project/agent-shield/")
