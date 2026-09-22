"""Build the index once per session, in its own process, before any test opens a
Chroma client (Chroma allows one client per path per process, and a rebuild
under an open client fails with a read-only database error)."""
import subprocess
import sys

import pytest


@pytest.fixture(scope="session", autouse=True)
def built_index():
    r = subprocess.run([sys.executable, "-m", "profile_rag.build"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-800:]
    assert "indexed" in r.stdout, r.stdout
    return r.stdout.strip().splitlines()[-1]
