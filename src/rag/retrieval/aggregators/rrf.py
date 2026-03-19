from __future__ import annotations

from langchain_core.documents import Document


class RRFAggregator:
    """Reciprocal Rank Fusion for combining multiple ranked lists."""

    def __init__(self, *, k: int = 60) -> None:
        self.k = k

    def aggregate(
        self,
        results: list[list[tuple[Document, float]]],
        *,
        top_k: int,
    ) -> list[tuple[Document, float]]:
        # `results` is a list of ranked lists (one per retriever/config).
        # Example:
        # results = [
        #   [(docA, 0.91), (docB, 0.83)],   # dense retriever ranking
        #   [(docB, 0.77), (docC, 0.61)],   # sparse retriever ranking
        # ]
        # Highest scored docs ranks the highest in each list.
        # RRF combines these lists by using rank positions (not raw scores).
        # A document ranked 1st contributes more than one ranked 10th.
        # Docs that appear in multiple lists accumulate more score.
        scores: dict[str, float] = {}
        docs: dict[str, Document] = {}

        for ranked_list in results:
            for rank, (doc, _score) in enumerate(ranked_list, start=1):
                doc_id = str(doc.metadata.get("_id", "")) + "|" + str(doc.metadata.get("collection_name", ""))
                docs[doc_id] = doc
                # Reciprocal Rank Fusion: 1 / (k + rank)
                # k controls how quickly the contribution decays with rank.
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (self.k + rank)

        # Convert the fused scores back to a ranked list
        fused = [(docs[doc_id], score) for doc_id, score in scores.items()]
        fused.sort(key=lambda x: x[1], reverse=True)
        return fused[:top_k]
