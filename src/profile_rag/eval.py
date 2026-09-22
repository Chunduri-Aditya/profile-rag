"""Frozen retrieval eval.  uv run python -m profile_rag.eval

Reports recall@1, recall@3 and MRR per mode over eval/questions.jsonl, plus the
FAQ hit rate. A hit is any returned id that starts with an expected prefix.
"""
import json
import sys

from . import config
from .retrieve import candidates, faq_match, rerank

MODES = ("bm25", "dense", "hybrid", "hybrid+rerank")


def load_bank() -> list[dict]:
    with (config.ROOT / "eval" / "questions.jsonl").open() as f:
        return [json.loads(line) for line in f if line.strip()]


def ranked_ids(q: str, mode: str) -> list[str]:
    if mode == "hybrid+rerank":
        nodes = rerank(q, candidates(q, mode="hybrid"))
    else:
        nodes = candidates(q, mode=mode)
    return [n.node.node_id for n in nodes]


def score(bank: list[dict], mode: str) -> dict:
    r1 = r3 = rr = 0.0
    for item in bank:
        ids = ranked_ids(item["q"], mode)
        rank = next((i for i, cid in enumerate(ids) if any(cid.startswith(p) for p in item["expect"])), None)
        if rank is not None:
            rr += 1 / (rank + 1)
            r3 += rank < 3
            r1 += rank == 0
    n = len(bank)
    return {"mode": mode, "n": n, "recall@1": r1 / n, "recall@3": r3 / n, "mrr": rr / n}


def faq_hits(bank: list[dict]) -> float:
    return sum(faq_match(item["q"]) is not None for item in bank) / len(bank)


def run(modes=MODES) -> dict:
    bank = load_bank()
    rows = [score(bank, m) for m in modes]
    return {"rows": rows, "faq_hit_rate": faq_hits(bank)}


if __name__ == "__main__":
    out = run()
    print(f"{'mode':16} {'n':>3} {'r@1':>6} {'r@3':>6} {'mrr':>6}")
    for r in out["rows"]:
        print(f"{r['mode']:16} {r['n']:>3} {r['recall@1']:6.3f} {r['recall@3']:6.3f} {r['mrr']:6.3f}")
    print(f"faq hit rate on bank: {out['faq_hit_rate']:.3f}")
    sys.exit(0)
