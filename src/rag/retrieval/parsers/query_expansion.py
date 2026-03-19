from __future__ import annotations

from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from rag.retrieval.base import ParserType


class QueryExpansion:
    """
    Expand the user query into up to `max_expansions` distinct expansions.
    """

    parser_type = ParserType.QUERY_EXPANSION

    def __init__(self, prompt: ChatPromptTemplate, llm, *, max_expansions: int) -> None:
        if max_expansions < 1:
            raise ValueError("max_expansions must be >= 1")
        self.prompt = prompt
        self.llm = llm
        self.max_expansions = max_expansions

    def chain(self):
        return self.prompt | self.llm | StrOutputParser()

    async def ainvoke(self, payload: dict[str, Any]) -> list[str]:
        query = payload.get("query", "")
        if not isinstance(query, str):
            raise ValueError("payload must be of type `str`")

        expansions = await self.chain().ainvoke({"query": query, "max_expansions": self.max_expansions})
        # split by new lines, remove empty results, and trim whitespace
        cleaned = [expansion.strip() for expansion in expansions.splitlines() if expansion.strip()]
        # make sure we don't return more than max_expansions
        return cleaned[: self.max_expansions]
