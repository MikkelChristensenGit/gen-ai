from __future__ import annotations

from typing import Any

from rag.retrieval.base import ParserType


class QueryIdentity:
    """
    Return the query as-in. Baseline parser.
    Establish the shape and cycle of parsers:
    - async API (ainvoke)
    - standard payload ({"query": ...})
    - stable identity (parser_type = ParserType.QUERY_IDENTITY)
    """

    parser_type = ParserType.QUERY_IDENTITY

    async def ainvoke(self, payload: dict[str, Any]) -> str:
        query = payload.get("query")
        if not isinstance(query, str):
            raise ValueError("payload must contain non-empty `query` str")
        return query
