"""Build the Chroma vector index and the persisted BM25 index from the corpus.

    uv run python -m profile_rag.build
"""
import json
import shutil

import chromadb
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.llms import MockLLM
from llama_index.core.schema import TextNode
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.chroma import ChromaVectorStore

from . import config


def load_nodes() -> list[TextNode]:
    nodes = []
    with config.CORPUS.open() as f:
        for line in f:
            row = json.loads(line)
            nodes.append(
                TextNode(
                    id_=row["id"],
                    text=row["text"],
                    metadata={"section": row["section"], "title": row["title"], "url": row["url"]},
                    excluded_embed_metadata_keys=["url"],
                    excluded_llm_metadata_keys=["url"],
                )
            )
    return nodes


def build() -> int:
    Settings.llm = MockLLM()
    Settings.embed_model = FastEmbedEmbedding(model_name=config.EMBED_MODEL)
    for d in (config.CHROMA_DIR, config.BM25_DIR):
        shutil.rmtree(d, ignore_errors=True)
    config.DATA.mkdir(exist_ok=True)

    nodes = load_nodes()
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    collection = client.get_or_create_collection(config.COLLECTION)
    store = ChromaVectorStore(chroma_collection=collection)
    ctx = StorageContext.from_defaults(vector_store=store)
    VectorStoreIndex(nodes, storage_context=ctx, show_progress=False)

    bm25 = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=config.TOP_K)
    bm25.persist(str(config.BM25_DIR))
    return len(nodes)


if __name__ == "__main__":
    print(f"indexed {build()} chunks")
