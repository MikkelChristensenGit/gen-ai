from __future__ import annotations

import asyncio

from fastembed import SparseTextEmbedding

from rag.retrieval.base import SparseVector

# FastEmbed's SparseTextEmbedding returns sparse vectors with indices and values,
# which is exactly what Qdrant expects for sparse vectors.
# Qdrant/bm25 is a supported sparse model in FastEmbed.
# Notes:
# The first run will download the model: expect a short delay
# This is a modern approach: sparse embeddings without implementing the BM25 algorithm ourselves.
# We can later swap to SPLADE by changing model_name.


class SparseBM25Embedder:
    def __init__(self, model_name: str = "Qdrant/bm25", *, batch_size: int):
        # model download happens on first init
        self.model = SparseTextEmbedding(model_name=model_name)
        self.batch_size = batch_size

    def _embed_sync(self, texts: list[str]) -> list[SparseVector]:
        # FastEmbed returns SparseEmbedding objects with .indices and .values
        sparse_list = list(self.model.embed(texts, batch_size=self.batch_size))
        out: list[SparseVector] = []
        for emb in sparse_list:
            indices = [int(x) for x in emb.indices]
            values = [float(x) for x in emb.values]
            out.append({"indices": indices, "values": values})
        return out

    async def aembed_batch(self, texts: list[str]) -> list[SparseVector]:
        # Run the sync embedder in a thread to avoid blocking the event loop
        return await asyncio.to_thread(self._embed_sync, texts)
