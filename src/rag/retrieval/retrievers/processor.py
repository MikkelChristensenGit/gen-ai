from __future__ import annotations

import asyncio
from itertools import chain
from typing import Any

from langchain_core.documents import Document
from qdrant_client import AsyncQdrantClient

from rag.retrieval.base import EmbeddedItem, RetrievalType, RetrieverConfig, VectorType
from rag.retrieval.retrievers.qdrant_retriever import QdrantRetriever

"""
This is the retriever execution engine for configs:
- chooses which embeddings are relevant for each RetrieverConfig
- runs each config asynchronously
- gathers and flattens results

Why configs matter:
Instead of hardcoding retrieval logic in a single function, we can express it as
- parser set
- retrieval type
- request args

This enables "ensemble retrieval" by configuration.

Step A:
We only implement `RetrievalType.DENSE` for now.
But now the processor should:
- skip unsupported configs
- still be structured to support SPARSE/HYBRID later

Learning lesson:
This file teaches how "configuration-driven wiring" works:
- cfg.parser tells us which parsed outputs to use
- cfg.type tells us which retriever method to run
cfg.request_args tells us how to query Qdrant (limit, threshold, etc).
This prevents the "if/else explosion" in big systems.
"""


def _dense_vectors_for_config(embeddings: list[EmbeddedItem], cfg: RetrieverConfig) -> list[list[float]]:
    """
    Filter embeddings for those relevant to the config and of DENSE type. It builds a list of vectors and returns it.
    """
    wanted_parsers = set(cfg.parser)
    dense: list[list[float]] = []
    for emb in embeddings:
        if emb["parser"] not in wanted_parsers:
            continue
        if emb["vector_type"] == VectorType.DENSE:
            dense.append(emb["vector"])
    return dense


class RetrievalComponent:
    """
    Orchestrates retrieval runs against Qdrant using config-driven logic.
    __init__ stores a QdrantRetriever instance.
    from_default creates a default QdrantRetriever from URL and API key.
    run executes retrievals for each RetrieverConfig asynchronously, gathering and flattening results.
    """

    def __init__(self, retriever: QdrantRetriever) -> None:
        self.retriever = retriever

    @classmethod
    def from_default(cls, *, qdrant_url: str, api_key: str | None = None) -> RetrievalComponent:
        client = AsyncQdrantClient(url=qdrant_url, api_key=api_key)
        return cls(retriever=QdrantRetriever(client))

    async def run(
        self, embedding: list[EmbeddedItem], configs: list[RetrieverConfig]
    ) -> list[list[tuple[Document, float]]]:
        tasks: list[asyncio.Task[Any]] = []

        for cfg in configs:
            opts = cfg.request_args.model_dump()
            if cfg.type is not RetrievalType.DENSE:
                # Step A supports only DENSE
                continue

            dense = _dense_vectors_for_config(embedding, cfg)
            if not dense:
                continue

            coro = self.retriever.dense_batch_search(vectors=dense, **opts)
            tasks.append(asyncio.create_task(coro))

        if not tasks:
            return []

        results = await asyncio.gather(*tasks)
        # Flatten list[list[list[(doc,score)]]]
        return list(chain.from_iterable(results))
