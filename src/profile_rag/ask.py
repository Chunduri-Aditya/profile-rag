"""Answer one question from the terminal, retrieval only.

    uv run python -m profile_rag.ask "what is Agent Shield"
    uv run python -m profile_rag.ask --json "where did Aditya study"
    uv run python -m profile_rag.ask --mode dense "retinal segmentation"   # raw vector search

No language model is involved at any step: FAQ match by embedding similarity,
BM25 plus dense vector search fused by rank, a cross-encoder reranker, and
sentence selection by term overlap. `--mode dense|bm25|hybrid` skips the FAQ
and rerank layers and prints the raw ranked chunks for that retriever.
"""
import argparse
import json
import sys

from .retrieve import answer, candidates


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="profile_rag.ask", description=__doc__.split("\n")[0])
    ap.add_argument("question", nargs="+")
    ap.add_argument("--json", action="store_true", help="print the full answer object")
    ap.add_argument("--mode", choices=["dense", "bm25", "hybrid"], help="raw ranked chunks from one retriever")
    args = ap.parse_args(argv)
    q = " ".join(args.question)

    if args.mode:
        for n in candidates(q, mode=args.mode):
            print(f"{n.score:8.4f}  {n.node.node_id}\n          {n.node.get_content()[:160]}")
        return 0

    out = answer(q)
    if args.json:
        json.dump(out, sys.stdout, indent=2, ensure_ascii=False)
        print()
        return 0
    if out["mode"] == "fallback":
        print("No answer on the site. Try:")
        for s in out["suggestions"][:6]:
            print(f"  - {s}")
        return 1
    print(out["answer"])
    print()
    for s in out["sources"]:
        print(f"  [{s['section']}] {s['title']}  {s['url']}")
    print(f"\n  mode={out['mode']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
