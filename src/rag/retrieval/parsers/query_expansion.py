from __future__ import annotations

from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from rag.retrieval.base import ParserType


class QueryExpansion:
    """
    Expand the user query into 1-3 distinct expansions that keep the original intent intact.
    """

    parser_type = ParserType.QUERY_EXPANSION

    def __init__(self, prompt: ChatPromptTemplate, llm) -> None:
        self.prompt = prompt
        self.llm = llm

    def chain(self):
        return self.prompt | self.llm | StrOutputParser()

    async def ainvoke(self, payload: dict[str, Any]) -> list[str]:
        query = payload.get("query", "")
        if not isinstance(query, str):
            raise ValueError("payload must be of type `str`")

        expansions = await self.chain().ainvoke({"query": query})
        # split by new lines, remove empty results, and trim whitespace
        return [expansion.strip() for expansion in expansions.splitlines() if expansion.strip()]
