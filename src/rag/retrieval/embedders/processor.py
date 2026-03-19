from __future__ import annotations

import asyncio
from collections.abc import Mapping

from rag.retrieval.base import (
    DenseVector,
    EmbeddedItem,
    ParsedQuery,
    ParserType,
    QueryParser,
    VectorType,
)
from rag.retrieval.embedders.dense import DenseEmbedder
from rag.retrieval.embedders.sparse_bm25 import SparseBM25Embedder

# This is where the architecture becomes real
# It has two orchestration components:
# 1. ParserComponent
#  1.1. Runs all parsers concurrently
#  1.2. returns typed ParsedQuery objects
# 2. EmbedComponent
#  2.1. Decides which parser outputs should be embedded as dense/sparse
#  2.2. batches by vector type
#  2.3. returns typed EmbeddedItem objects
# Designed after "batch by type".
# Main idea:
# The pipeline may produce several parsed queries (identity + multi query + hyde etc)
# Not all parsed queries need the same embedding type.
# So we map parser -> vector types and embed in batches.
# Important: vector_types_by_parser
# This mapping makes it easy to scale.
# Today: QUERY_IDENTITY -> DENSE
# Later: QUERY_IDENTITY -> DENSE + SPARSE
# Pipeline code does not change - only the mapping change.

# Async: all parsers run concurrently via asyncio.gather.
# Batching: after all parser outputs are collected, the embedder groups by vector type
# and makes one API call per type with all texts in that group
# (e.g., one dense batch, one sparse batch).


class ParserComponent:
    def __init__(self, parsers: list[QueryParser]) -> None:
        self.parsers = parsers

    async def run(self, query: str) -> list[ParsedQuery]:
        payload = {"query": query}
        # Run all parsers concurrently
        tasks = [asyncio.create_task(parser.ainvoke(payload)) for parser in self.parsers]
        outputs = await asyncio.gather(*tasks)

        parsed: list[ParsedQuery] = []
        for parser, output in zip(self.parsers, outputs, strict=True):
            # self.parsers = [QueryIdentity(), QueryExpansion()]
            # outputs = ["original query", ["exp1", "exp2"]]
            if isinstance(output, str):
                texts = [output]
            elif isinstance(output, list) and all(isinstance(item, str) for item in output):
                texts = output
            else:
                raise ValueError("Parser output must be a `str` or `list[str]`")

            for text in texts:
                parsed.append({"parser": parser.parser_type, "text": text})
        return parsed


class EmbedComponent:
    """Batch embeds parsed queries based on parser -> vector type mapping."""

    def __init__(
        self,
        *,
        dense: DenseEmbedder,
        sparse: SparseBM25Embedder,
        vector_types_by_parser: Mapping[ParserType, set[VectorType]],
    ) -> None:
        self.dense = dense
        self.sparse = sparse
        self.vector_types_by_parser = vector_types_by_parser

    @classmethod
    def from_default(
        cls,
        *,
        dense_embed_model: str,
        sparse_embed_model: str,
        sparse_batch_size: int,
    ) -> EmbedComponent:
        vector_types_by_parser = {
            ParserType.QUERY_IDENTITY: {VectorType.DENSE, VectorType.SPARSE},
            ParserType.QUERY_REPHRASE: {VectorType.DENSE, VectorType.SPARSE},
            ParserType.QUERY_EXPANSION: {VectorType.DENSE, VectorType.SPARSE},
        }
        return cls(
            dense=DenseEmbedder(model=dense_embed_model),
            sparse=SparseBM25Embedder(model_name=sparse_embed_model, batch_size=sparse_batch_size),
            vector_types_by_parser=vector_types_by_parser,
        )

    async def run(self, parsed: list[ParsedQuery]) -> list[EmbeddedItem]:
        # Collect all dense text in one batch
        dense_texts: list[tuple[ParserType, str]] = []
        sparse_texts: list[tuple[ParserType, str]] = []
        for pq in parsed:
            parser_type = pq["parser"]
            text = pq["text"]
            if VectorType.DENSE in self.vector_types_by_parser.get(parser_type, set()):
                dense_texts.append((parser_type, text))
            if VectorType.SPARSE in self.vector_types_by_parser.get(parser_type, set()):
                sparse_texts.append((parser_type, text))

        dense_embeddings: list[EmbeddedItem] = []
        sparse_embeddings: list[EmbeddedItem] = []

        if not dense_texts and not sparse_texts:
            raise ValueError("No texts to embed based on the provided vector_types_by_parser mapping.")

        # Dense
        # Separate the texts from their parser types for embedding
        dense_texts_list = [text for _, text in dense_texts]
        # One dense batch call
        if dense_texts_list:
            dense_vectors: list[DenseVector] = await self.dense.aembed_batch(dense_texts_list)
            # Reattach metadata + build EmbeddingItem objects
            for (parser_type, dense_text), dense_vec in zip(dense_texts, dense_vectors, strict=True):
                dense_item: EmbeddedItem = {
                    "parser": parser_type,
                    "vector_type": VectorType.DENSE,
                    "text": dense_text,
                    "vector": dense_vec,
                }
                dense_embeddings.append(dense_item)

        # Sparse
        sparse_texts_list = [text for _, text in sparse_texts]
        # One sparse batch call
        if sparse_texts_list:
            sparse_vectors = await self.sparse.aembed_batch(sparse_texts_list)
            for (parser_type, sparse_text), sparse_vec in zip(sparse_texts, sparse_vectors, strict=True):
                sparse_item: EmbeddedItem = {
                    "parser": parser_type,
                    "vector_type": VectorType.SPARSE,
                    "text": sparse_text,
                    "vector": sparse_vec,
                }
                sparse_embeddings.append(sparse_item)

        return dense_embeddings + sparse_embeddings
