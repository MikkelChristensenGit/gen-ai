from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from prompts.query_expansion import query_expansion_prompt
from qdrant.settings import qdrant_settings
from rag.retrieval.aggregators.rrf import RRFAggregator
from rag.retrieval.aggregators.simple import SimpleScoreAggregator
from rag.retrieval.base import ParserType, RequestArgs, RetrievalType, RetrieverConfig
from rag.retrieval.embedders.processor import EmbedComponent, ParserComponent
from rag.retrieval.parsers.identity import QueryIdentity
from rag.retrieval.parsers.query_expansion import QueryExpansion
from rag.retrieval.retrievers.processor import RetrievalComponent
from rag.settings import retrieval_settings

"""
Top level orchestrator.
Query -> Parsers -> Embedders -> Retrievers -> Aggregator

It contains composition, not business logic.

Why `from_default()`is a big deal:
- constructs default components
- constructs default configs
- returns a ready-to-use pipeline instance

This gives us a clean "single entrypoint" for retrieval:
- our CLI or API calls `RetrievalPipeline.from_default(...)`
- then uses await pipeline.ainvoke(query)

Scaling:
As complexity grows, we want to keep construction separate from execution:
- execution stays simple and testable
- construction becomes config-driven
"""


@dataclass(slots=True)
class RetrievalPipeline:
    parser_component: ParserComponent
    embed_component: EmbedComponent
    retrieval_component: RetrievalComponent
    aggregator: SimpleScoreAggregator | RRFAggregator
    configs: list[RetrieverConfig]
    top_k: int

    @classmethod
    def from_default(
        cls,
        *,
        qdrant_url: str,
        qdrant_api_key: str | None,
        collection_name: str,
        dense_embed_model: str,
        candidate_limit: int,
        top_k: int,
    ) -> RetrievalPipeline:
        llm = ChatOpenAI(model=retrieval_settings.CHAT_MODEL)
        parser_component = ParserComponent(
            parsers=[QueryIdentity(), QueryExpansion(prompt=query_expansion_prompt, llm=llm)]
        )
        embed_component = EmbedComponent.from_default(
            dense_embed_model=dense_embed_model,
            sparse_embed_model=retrieval_settings.SPARSE_EMBED_MODEL,
            sparse_batch_size=retrieval_settings.SPARSE_BATCH_SIZE,
        )
        retrieval_component = RetrievalComponent.from_default(qdrant_url=qdrant_url, api_key=qdrant_api_key)

        configs = [
            RetrieverConfig(
                parser=[ParserType.QUERY_IDENTITY, ParserType.QUERY_EXPANSION],
                type=RetrievalType.DENSE,
                request_args=RequestArgs(
                    collection_name=collection_name,
                    limit=candidate_limit,
                    using=qdrant_settings.DENSE_VECTOR_NAME,
                    score_threshold=None,
                    filter=None,
                ),
            ),
            RetrieverConfig(
                parser=[ParserType.QUERY_IDENTITY, ParserType.QUERY_EXPANSION],
                type=RetrievalType.SPARSE,
                request_args=RequestArgs(
                    collection_name=collection_name,
                    limit=candidate_limit,
                    using=qdrant_settings.SPARSE_VECTOR_NAME,
                    score_threshold=None,
                    filter=None,
                ),
            ),
        ]

        has_dense = any(cfg.type is RetrievalType.DENSE for cfg in configs)
        has_sparse = any(cfg.type is RetrievalType.SPARSE for cfg in configs)
        aggregator: SimpleScoreAggregator | RRFAggregator
        if has_dense and has_sparse:
            aggregator = RRFAggregator()
        else:
            aggregator = SimpleScoreAggregator()

        return cls(
            parser_component=parser_component,
            embed_component=embed_component,
            retrieval_component=retrieval_component,
            aggregator=aggregator,
            configs=configs,
            top_k=top_k,
        )

    async def ainvoke(self, query: str) -> list[tuple[Document, float]]:
        parsed = await self.parser_component.run(query)
        embeddings = await self.embed_component.run(parsed)
        results = await self.retrieval_component.run(embeddings, self.configs)
        return self.aggregator.aggregate(results, top_k=self.top_k)
