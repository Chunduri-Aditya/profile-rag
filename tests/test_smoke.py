from fastapi.testclient import TestClient

from profile_rag.api import app


def test_thin_path(built_index):
    assert built_index.startswith("indexed ")
    assert int(built_index.split()[1]) > 300
    r = TestClient(app).post("/ask", json={"question": "what is Agent Shield"})
    assert r.status_code == 200, r.text
    top = r.json()["sources"][0]
    print("TOP:", top["id"], top["score"])
    assert top["id"].startswith("project/agent-shield/")
