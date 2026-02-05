from __future__ import annotations

from langchain_core.documents import Document

"""
A minimal aggregator that:
- flattens result lists
- deduplication by stable ID
- sorts by score
- return top-k

Only one retriever for now, so doesn't make sense, but it will when we have multiple.
Retrievers -> Aggregator -> Documents is the pattern.
Even with one retriever config now, we keep this stage so that:
- Adding another retriever config later doesn't change pipeline shape.
- Swapping to RRF later is isolated to this folder.

TODO: Changing later:
- replace SimpleScoreAggregator with RRFAggregator (and optionally weight by config).
At that point, the pipeline doesn't change at all - only the aggregator.
"""


class SimpleScoreAggregator:
    """Combine multiple ranked lists, deduplicate, sort by score descending."""

    def aggregate(
        self,
        results: list[list[tuple[Document, float]]],
        *,
        top_k: int,
    ) -> list[tuple[Document, float]]:
        seen: set[str] = set()
        flat: list[tuple[Document, float]] = []
        # Flatten and deduplicate
        for batch in results:
            for doc, score in batch:
                doc_id = str(doc.metadata.get("_id", "")) + "|" + str(doc.metadata.get("collection_name", ""))
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                flat.append((doc, score))
        # Sort by score descending
        flat.sort(key=lambda x: x[1], reverse=True)
        return flat[:top_k]
