from __future__ import annotations

import asyncio
from itertools import chain
from typing import Any

from langchain_core.documents import Document
from qdrant_client import AsyncQdrantClient

from rag.retrieval.base import DenseVector, EmbeddedItem, RetrievalType, RetrieverConfig, SparseVector, VectorType
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


def _dense_vectors_for_config(embeddings: list[EmbeddedItem], cfg: RetrieverConfig) -> list[DenseVector]:
    """
    Filter embeddings for those relevant to the config and of DENSE type. It builds a list of vectors and returns it.
    """
    wanted_parsers = set(cfg.parser)
    dense: list[DenseVector] = []
    for emb in embeddings:
        if emb["parser"] not in wanted_parsers:
            continue
        if emb["vector_type"] == VectorType.DENSE:
            dense.append(emb["vector"])
    return dense


def _sparse_vectors_for_config(embeddings: list[EmbeddedItem], cfg: RetrieverConfig) -> list[SparseVector]:
    """ """
    wanted_parsers = set(cfg.parser)
    sparse: list[SparseVector] = []
    for emb in embeddings:
        if emb["parser"] not in wanted_parsers:
            continue
        if emb["vector_type"] == VectorType.SPARSE:
            sparse.append(emb["vector"])
    return sparse


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
        dense_tasks: list[asyncio.Task[Any]] = []
        sparse_tasks: list[asyncio.Task[Any]] = []

        for cfg in configs:
            opts = cfg.request_args.model_dump()
            if cfg.type not in {RetrievalType.DENSE, RetrievalType.SPARSE, RetrievalType.HYBRID}:
                continue

            # Dense
            if cfg.type == RetrievalType.DENSE:
                dense = _dense_vectors_for_config(embedding, cfg)
                if not dense:
                    continue
                coro_dense = self.retriever.dense_batch_search(vectors=dense, **opts)
                dense_tasks.append(asyncio.create_task(coro_dense))

            # Sparse
            if cfg.type == RetrievalType.SPARSE:
                sparse = _sparse_vectors_for_config(embedding, cfg)
                if not sparse:
                    continue
                coro_sparse = self.retriever.sparse_batch_search(vectors=sparse, **opts)
                sparse_tasks.append(asyncio.create_task(coro_sparse))

            # Hybrid
            if cfg.type == RetrievalType.HYBRID:
                dense = _dense_vectors_for_config(embedding, cfg)
                sparse = _sparse_vectors_for_config(embedding, cfg)
                if not dense and not sparse:
                    continue
                coro_dense = self.retriever.dense_batch_search(vectors=dense, **opts)
                coro_sparse = self.retriever.sparse_batch_search(vectors=sparse, **opts)
                if dense:
                    dense_tasks.append(asyncio.create_task(coro_dense))
                if sparse:
                    sparse_tasks.append(asyncio.create_task(coro_sparse))

        if not dense_tasks and not sparse_tasks:
            return []
        dense_results: list[list[list[tuple[Document, float]]]] = []
        sparse_results: list[list[list[tuple[Document, float]]]] = []
        if dense_tasks:
            dense_results = await asyncio.gather(*dense_tasks)
        if sparse_tasks:
            sparse_results = await asyncio.gather(*sparse_tasks)
        # Flatten list[list[list[(doc,score)]]]
        return list(chain.from_iterable(dense_results + sparse_results))
