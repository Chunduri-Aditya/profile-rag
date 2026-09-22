import subprocess
import sys


def run(*args):
    return subprocess.run([sys.executable, "-m", "profile_rag.ask", *args], capture_output=True, text=True)


def test_cli_answers_and_cites(built_index):
    r = run("what is Agent Shield")
    assert r.returncode == 0, r.stderr[-500:]
    assert "Inspect AI" in r.stdout and "work/agent-shield/" in r.stdout
    assert "MockLLM" not in r.stdout + r.stderr


def test_cli_fallback_exits_nonzero(built_index):
    r = run("banana bread recipe")
    assert r.returncode == 1
    assert "No answer on the site" in r.stdout
