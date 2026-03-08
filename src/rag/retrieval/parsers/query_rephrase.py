from __future__ import annotations

from typing import Any

# from langchain.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from rag.retrieval.base import ParserType


class QueryRephrase:
    """
    Rephrase the user query into into a better search query
    by resolving pronouns, adding missing context from the chat,
    fixing typos, and making the query more explicit,
    while keeping the original intent intact.
    """

    parser_type = ParserType.QUERY_REPHRASE

    def __init__(self, prompt: ChatPromptTemplate, llm) -> None:
        self.prompt = prompt
        self.llm = llm

    def chain(self):
        return self.prompt | self.llm | StrOutputParser()

    async def ainvoke(self, payload: dict[str, Any]) -> str:
        query = payload.get("query", "")
        if not isinstance(query, str):
            raise ValueError("payload must be of type `str`")

        rephrase = await self.chain().ainvoke({"query": query})
        # print(f"Rephrased query: {rephrase}")
        # split by new lines, remove empty results, and trim whitespace
        return rephrase
