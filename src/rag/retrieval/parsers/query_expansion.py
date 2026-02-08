from __future__ import annotations

from typing import Any

# from langchain.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from rag.retrieval.base import ParserType


class QueryExpansion:
    """
    Expand the user query into 1-3 distinct expansions that keep the original intent intact.
    """

    parser_type = ParserType.QUERY_EXPANSION

    def __init__(self, payload: dict[str, Any], prompt: ChatPromptTemplate, llm) -> None:
        self.query = payload.get("query", "")
        self.prompt = prompt
        self.llm = llm

    def chain(self):
        chain = self.prompt | self.llm | StrOutputParser()
        return chain

    async def ainvoke(self, payload: dict[str, Any]) -> list[str]:
        if not isinstance(payload.get("query", ""), str):
            raise ValueError("payload must be of type `str`")
        self.query = payload["query"]
        queries = self.chain()
        return queries
