from __future__ import annotations

import asyncio

from langchain_openai import OpenAIEmbeddings

from rag.retrieval.base import DenseVector


class DenseEmbedder:
    """
    A thin wrapper around embedding provider with aembed_batch.
    It defines the `aembed_batch` from Embedder(Protocol). This gives it
    a stable interface, and enforces batching by design.
    Batching is cheaper, faster, and it fits async concurrency patterns.
    """

    timeout_seconds = 10

    def __init__(self, model: str) -> None:
        self.emb = OpenAIEmbeddings(model=model)

    async def aembed_batch(self, queries: list[str]) -> list[DenseVector]:
        """The batching is simply that we pass `queries: list[str]` instead of a single string."""
        try:
            async with asyncio.timeout(self.timeout_seconds):
                vectors = await self.emb.aembed_documents(queries)
                return [DenseVector(v) for v in vectors]
        except TimeoutError:
            vectors = await self.emb.aembed_documents(queries)
            return [DenseVector(v) for v in vectors]
