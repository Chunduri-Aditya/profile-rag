"""Hybrid retrieval: Chroma dense + BM25, fused by reciprocal rank. No LLM."""
from functools import lru_cache

import chromadb
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.chroma import ChromaVectorStore

from . import config


@lru_cache(maxsize=1)
def retriever() -> QueryFusionRetriever:
    Settings.llm = None
    Settings.embed_model = FastEmbedEmbedding(model_name=config.EMBED_MODEL)
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    store = ChromaVectorStore(chroma_collection=client.get_or_create_collection(config.COLLECTION))
    index = VectorStoreIndex.from_vector_store(store)
    dense = index.as_retriever(similarity_top_k=config.TOP_K)
    bm25 = BM25Retriever.from_persist_dir(str(config.BM25_DIR))
    return QueryFusionRetriever(
        [dense, bm25],
        similarity_top_k=config.TOP_K,
        num_queries=1,
        mode="reciprocal_rerank",
        use_async=False,
        llm=None,
    )


def search(question: str) -> list[NodeWithScore]:
    return retriever().retrieve(question)


def rerank(nodes: list[NodeWithScore]) -> list[NodeWithScore]:
    raise NotImplementedError("rerank")


def faq(question: str):
    raise NotImplementedError("faq")
