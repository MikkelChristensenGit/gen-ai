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
    query.strip() removes leading and trailing whitespace characters (spaces, tabs,
        newlines, etc.) from the query string. For example, "  hello world  ".strip()
        returns "hello world"
    """

    parser_type = ParserType.QUERY_IDENTITY

    async def ainvoke(self, payload: dict[str, Any]) -> str:
        query = payload.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("payload must contain non-empty `query` str")
        return query.strip()
