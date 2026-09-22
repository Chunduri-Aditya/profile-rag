"""Answer path with no LLM: FAQ match -> hybrid retrieval -> cross-encoder rerank
-> sentence extraction. Everything runs on CPU with ONNX models."""
import re
from functools import lru_cache

import chromadb
import numpy as np
import yaml
from fastembed.rerank.cross_encoder import TextCrossEncoder
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.chroma import ChromaVectorStore

from . import config

CANDIDATES = 10          # fused candidates handed to the reranker
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
FAQ_THRESHOLD = 0.88     # cosine between query and FAQ question embeddings
ANSWER_FLOOR = -4.0      # cross-encoder logit below which we do not answer
MAX_SENTENCES = 3
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")


@lru_cache(maxsize=1)
def _embed() -> FastEmbedEmbedding:
    Settings.llm = None
    Settings.embed_model = FastEmbedEmbedding(model_name=config.EMBED_MODEL)
    return Settings.embed_model


@lru_cache(maxsize=1)
def _retrievers():
    _embed()
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    store = ChromaVectorStore(chroma_collection=client.get_or_create_collection(config.COLLECTION))
    index = VectorStoreIndex.from_vector_store(store)
    dense = index.as_retriever(similarity_top_k=CANDIDATES)
    bm25 = BM25Retriever.from_persist_dir(str(config.BM25_DIR))
    bm25.similarity_top_k = CANDIDATES
    hybrid = QueryFusionRetriever(
        [dense, bm25], similarity_top_k=CANDIDATES, num_queries=1,
        mode="reciprocal_rerank", use_async=False, llm=None,
    )
    return {"dense": dense, "bm25": bm25, "hybrid": hybrid}


@lru_cache(maxsize=1)
def _reranker() -> TextCrossEncoder:
    return TextCrossEncoder(model_name=RERANK_MODEL)


def candidates(question: str, mode: str = "hybrid") -> list[NodeWithScore]:
    return _retrievers()[mode].retrieve(question)


def rerank(question: str, nodes: list[NodeWithScore]) -> list[NodeWithScore]:
    if not nodes:
        return []
    texts = [n.node.get_content() for n in nodes]
    scores = list(_reranker().rerank(question, texts))
    out = [NodeWithScore(node=n.node, score=float(s)) for n, s in zip(nodes, scores)]
    out.sort(key=lambda n: (-n.score, n.node.node_id))
    return out


def extract(question: str, text: str) -> str:
    sents = [s.strip() for s in _SENT.split(text) if s.strip()]
    if len(sents) <= MAX_SENTENCES:
        return text.strip()
    scores = list(_reranker().rerank(question, sents))
    keep = sorted(range(len(sents)), key=lambda i: -scores[i])[:MAX_SENTENCES]
    return " ".join(sents[i] for i in sorted(keep))


@lru_cache(maxsize=1)
def _faq():
    entries = yaml.safe_load((config.ROOT / "faq.yaml").read_text())
    vecs = np.array(_embed().get_text_embedding_batch([e["q"] for e in entries]))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return entries, vecs


def faq_match(question: str) -> dict | None:
    entries, vecs = _faq()
    q = np.array(_embed().get_query_embedding(question))
    q /= np.linalg.norm(q)
    sims = vecs @ q
    i = int(sims.argmax())
    return {**entries[i], "score": float(sims[i])} if sims[i] >= FAQ_THRESHOLD else None


def _source(n: NodeWithScore) -> dict:
    return {"id": n.node.node_id, "score": n.score, "text": n.node.get_content(), **n.node.metadata}


def suggestions() -> list[str]:
    return [e["q"] for e in _faq()[0]]


def answer(question: str) -> dict:
    hit = faq_match(question)
    if hit:
        src = next((n for n in candidates(hit["q"]) if n.node.node_id == hit["source"]), None)
        sources = [_source(src)] if src else []
        return {"answer": hit["a"], "mode": "faq", "score": hit["score"], "sources": sources}
    ranked = rerank(question, candidates(question, mode="hybrid"))
    if not ranked or ranked[0].score < ANSWER_FLOOR:
        return {"answer": None, "mode": "fallback", "score": ranked[0].score if ranked else None,
                "sources": [], "suggestions": suggestions()}
    top = ranked[0]
    return {
        "answer": extract(question, top.node.get_content()),
        "mode": "extract",
        "score": top.score,
        "sources": [_source(n) for n in ranked[:3]],
    }
