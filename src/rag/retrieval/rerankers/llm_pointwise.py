from __future__ import annotations

import asyncio
import re

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

_NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")  # regex pattern to extract numbers


def _parse_score(raw: str) -> int:
    """
    Guardrails: Extracts a number from raw model output.
    LLM might return "Score: 8" or "8\n" instead of just "8".
    Enforce integer-only format. It rejects decimals like "8.5".
    Enforces valid range [1, 10]. If parsing fails or value is out of range, it raises ValueError with details.
    """
    match = _NUMBER_PATTERN.search(raw)
    if match is None:
        raise ValueError(f"Could not parse rerank score from model output: {raw!r}")

    token = match.group(0)
    if "." in token:
        raise ValueError(f"Rerank score must be an integer in [1, 10], got: {token}")

    score = int(token)
    if not 1 <= score <= 10:
        raise ValueError(f"Rerank score out of range [1, 10]: {score}")
    return score


class LLMPointwiseReranker:
    """Pointwise reranker that scores each (query, passage) pair with an LLM."""

    def __init__(self, *, llm, prompt: ChatPromptTemplate, max_concurrency: int) -> None:
        """
        Store dependencies and validate config.
        It receives llm, prompt and max_concurrency, and fails fast if max_concurrency is less than 1.
        """
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        self.llm = llm
        self.prompt = prompt
        self.max_concurrency = max_concurrency

    def chain(self):
        """Builds the LangChain runnable."""
        return self.prompt | self.llm | StrOutputParser()

    async def _score_one(self, *, query: str, doc: Document) -> int:
        """
        Scores one (query, document) pair.
        It sends query + doc.page_content into the chain, gets raw text back, then parses/validates it with _parse_score
        """
        raw = await self.chain().ainvoke({"query": query, "chunk": doc.page_content})
        return _parse_score(raw)

    async def arerank(
        self,
        *,
        query: str,
        candidates: list[tuple[Document, float]],  # fused results as list[(doc, score)]
        top_k: int,  # how many to return
    ) -> list[tuple[Document, float]]:
        """
        1) Input:
            [(DocumentA, 0.91), (DocumentB, 0.83), (DocumentC, 0.61), ...]  # fused results from retriever/aggregator
            Important: The second value (0.91, 0.83, etc) is the fused score, but it is NOT used for reranking.
            `arerank` ignores it (note the _fused_score variable name), because reranker produces a new score.
        2) Semaphore is the traffic controller. max_concurrency controls how many LLM calls run at the same time.
        It controls latency/cost/rate-limiting.
        3) _score_with_index add index on purpose. Each document gets wrapped as (index, doc, score).
        This allows us to preserve original order in case of ties in rerank score.
        4) asyncio.gather(*tasks) waits for all scoring tasks to complete.
        Tasks are scheduled together, but semaphore limits actual concurrent LLM calls.
        Each task calls _score_one, which prompts the LLM and parses an integer 1..10 back.
        E.g. scored = [(0, DocumentA, 9), (1, DocumentB, 8), (2, DocumentC, 6)].
        If one task fails, the exception is raised to the caller.
        5. Sorting logic. -item[2] means sort by rerank score descending. If two docs have equal score,
           earlier fused-ranked docs win: [(1, DocB, 9), (2, DocC, 9), (0, DocA, 7), (3, DocD, 4)].
           DocB and DocC tie with score 9, but DocB comes first because it had lower original index (1 vs 2).
        6. Convert and cut to final size. It converts to [(DocumentB, 9), (DocumentC, 9), (DocumentA, 7)]
        Then [:top_k] with top_k=2 would return [(DocumentB, 9), (DocumentC, 9)].
        """
        if top_k < 1:
            return []
        if not candidates:
            return []

        # Concurrency control. This limits how many LLM scoring calls run at once.
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def _score_with_index(index: int, doc: Document) -> tuple[int, Document, int]:
            """
            Helper to score a single document with its index in the original fused list.
            """
            async with semaphore:
                score = await self._score_one(query=query, doc=doc)
                return (index, doc, score)

        # Create tasks for all candidates. One async task per candidate document.
        tasks = [
            asyncio.create_task(_score_with_index(index=i, doc=doc)) for i, (doc, _fused_score) in enumerate(candidates)
        ]
        scored = await asyncio.gather(*tasks)

        # Sort by rerank score descending; for ties preserve original fused order.
        scored.sort(key=lambda item: (-item[2], item[0]))
        reranked: list[tuple[Document, float]] = [(doc, score) for _, doc, score in scored]
        return reranked[:top_k]
