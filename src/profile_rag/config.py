from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "corpus" / "portfolio.jsonl"
DATA = ROOT / "data"
CHROMA_DIR = DATA / "chroma"
BM25_DIR = DATA / "bm25"
COLLECTION = "portfolio"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
TOP_K = 5
