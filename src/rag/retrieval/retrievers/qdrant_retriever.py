from __future__ import annotations

from collections.abc import Iterable, Sequence

from langchain_core.documents import Document
from qdrant_client import AsyncQdrantClient, models

from rag.retrieval.base import DenseVector, SparseVector

# A DB adapter around AsyncQdrantClient that:
# - accepts vectors and requests args
# - performs batch queries
# - maps Qdrant results into a standard output type (Document + score)
# Why do we need this?
# It keeps "Qdrant API details" out of the pipeline.
# The pipeline should NOT know about:
# - QueryRequest objects
# - Qdrant ScoredPoint
# - with_vector, with_payload, etc.
# It should only know: "retriever returns results".
# Important mapping decision: payload keys.
# The adapter expects payload to contain:
# - page_content
# - metadata
# This is a contract with ingestion. In scalable systems, retrieval is only reliable if
#  ingestion payload format is stable and explicit.
# Why we add _id and collection_name into metadata
# This enables later steps:
# - deduplication across retrievers/configs
# - stable doc identity in aggregators
# - traceability ("where did this chunk come from?")
# Scaling insight: observability and stable IDs are production requirements
#  adding them now saves the work later.

# For now, only dense batch search and mapping results.

# Thin wrapper around AsyncQdrantClient
# Takes a batch of dense vectors + request options, runs a Qdrant batch query,
# and converts Qdrant's ScoredPoint results into (Document, score) pairs.
# Enforces payload contract: every stored point payload must include: page_content and metadata.


class QdrantRetriever:
    """DB adapter. Tiny wrapper around AsyncQdrantClient."""

    CONTENT_KEY: str = "page_content"
    METADATA_KEY: str = "metadata"

    def __init__(self, client: AsyncQdrantClient) -> None:
        self.client = client

    def _map_points(
        self, points: Iterable[models.ScoredPoint], *, collection_name: str
    ) -> list[tuple[Document, float]]:
        out: list[tuple[Document, float]] = []
        for point in points:
            payload = point.payload or {}
            metadata = (payload.get(self.METADATA_KEY) or {}) if isinstance(payload, dict) else {}
            metadata["_id"] = point.id
            metadata["collection_name"] = collection_name
            out.append(
                (
                    Document(page_content=payload.get(self.CONTENT_KEY, ""), metadata=metadata),
                    point.score,
                )
            )
        return out

    async def dense_batch_search(
        self,
        *,
        vectors: Sequence[DenseVector],
        collection_name: str,
        limit: int,
        filter: models.Filter | None = None,
        using: str | None = None,
        score_threshold: float | None = None,
    ) -> list[list[tuple[Document, float]]]:
        """
        Builds a batch of Qdrant QueryRequests from the input dense vectors,
        runs them in one call, then converts each result set into (Document, score) pairs with metadata.
        Takes a list of dense vectors plus query options (collection_name, limit
        and optional filter, using, score_threshold).
        For each vector, builds a models.QueryRequest with:
        - with_payload=True (so it fetches stored payload)
        - with_vector=False (we don't need the stored vector back)
        Runs client.query_batch_points ONCE with the full list of requests.
        Maps each query's ScoredPoint results into a list of (Document, score) where:
        - Document.page_content comes from payload['page_content']
        - Document.metadata comes from payload['metadata'] plus _id and collection_name
        Returns a list of results, one per input vector `list[list[(Document, score)]]`.
        """
        requests = [
            models.QueryRequest(
                query=v,
                limit=limit,
                filter=filter,
                using=using,
                score_threshold=score_threshold,
                with_payload=True,
                with_vector=False,
                offset=0,
            )
            for v in vectors
        ]
        results = await self.client.query_batch_points(
            collection_name=collection_name,
            requests=requests,
        )
        return [self._map_points(r.points, collection_name=collection_name) for r in results]

    async def sparse_batch_search(
        self,
        vectors: Sequence[SparseVector],
        collection_name: str,
        limit: int,
        filter: models.Filter | None = None,
        using: str | None = None,
        score_threshold: float | None = None,
    ) -> list[list[tuple[Document, float]]]:
        """
        Builds a batch of Qdrant QueryRequests from the input sparse vectors,
        runs them in one call, then converts each result set into (Document, score) pairs with metadata.
        Uses QueryRequest.query with a SparseVector payload and `using` to select the sparse vector name.
        Example:
        1. Input query: "What is the capital of France?"
        Sparse embedder converts that into a sparse vector like ~ {"indices": [12, 45, 78], "values": [0.8, 0.5, 0.3]}.
        2. sparse_batch_search input: we call it with a list of vectors (batch)
        vectors = [
            {"indices": [12, 45, 78], "values": [0.8, 0.5, 0.3]},
        ]
        3. Inside sparse_batch_search, for each v in vectors: build SparseVector:
        sparse_vec = models.SparseVector(indices=v["indices"], values=v["values"])
        Build QueryRequest.
        4. All requests are sent in one batch: results = await ...
        5. Qdrant scores results
        Qdrant computes sparse similarity (dot product between query vector and stored sparse vectors).
        Points that overlap with terms like "capital" and "France" will get higher scores.
        6. Mapping results: The _map_points converts Qdrant results into:
        [
            [
            (Document(page_content="Paris is the capital of France ...", metadata={...}), 0.92),
            (Document(page_content="France is a country in Europe ...", metadata={...}), 0.61),
            ]
        ]
        so we end with a list of results per input vector.

        Summary in one sentence:
        sparse_batch_search turns the query into a sparse vector, asks
        Qdrant to match it against stored sparse vectors, and returns ranked documents with scores.
        """
        requests = []
        for v in vectors:
            sparse_vec = models.SparseVector(indices=v["indices"], values=v["values"])

            requests.append(
                models.QueryRequest(
                    query=sparse_vec,
                    limit=limit,
                    filter=filter,
                    using=using,
                    score_threshold=score_threshold,
                    with_payload=True,
                    with_vector=False,
                    offset=0,
                )
            )
        results = await self.client.query_batch_points(
            collection_name=collection_name,
            requests=requests,
        )
        return [self._map_points(r.points, collection_name=collection_name) for r in results]
