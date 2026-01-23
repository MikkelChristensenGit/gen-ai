from __future__ import annotations

from enum import StrEnum, auto
from typing import Any, Protocol, TypedDict

from pydantic import BaseModel, Field
from qdrant_client import models

# base.py is a shared contracts module.
# In a modular system, multiple parts of the code needs to agree on:
# - what kind of parsers exists (identity vs HyDE vs multi-query)
# - what kinds of vectors exists (dense vs sparse)
# - what kinds of retrieval modes exists (dense vs hybrid)
# - how a retriever configuration is described (limit, thresholds, collection, filter)
# - what the intermediate objects look like as data flows through the pipeline
# - what methods components must implement (parser, embedder, retriever interfaces)

# Putting this in one place gives us
# a) Separation of concerns (scales cleanly)
#  Pipeline should orchestrate
#  Components should do 1 job each
#  Shared contract should be stable and reused everywhere
#  Without base.py, slightly different dict keys will be used in different files
# b) A single language for the whole system
# c) static typing + runtime validation
#  mypy catches interface mismatches early
#  pydantic validates configuration values at runtime


# -- Enums --
# Replace raw strings like "dense" with typed values - prevents typos


class ParserType(StrEnum):
    """
    auto() is a helper that tells Python "choose the value for this enum member automatically.
    We don't have to manually type values like "QUERY_IDENTITY". Reduces string type errors.
    Combining it with StrEnum results in a string value equal to
    the member name (e.g. "QUERY_IDENTIY").
    """

    QUERY_IDENTITY = auto()
    # later: HYDE, MULTI_QUERY, ENTITY_EXTRACTION


class VectorType(StrEnum):
    """
    VectorType.DENSE.value is "dense". Get rid of strings.
    Also good for serialization, i.e., when you turn objects into JSON (or similar).
    The enum values naturally become plain strings without extra work.
    """

    DENSE = auto()
    SPARSE = auto()


class RetrievalType(StrEnum):
    DENSE = auto()
    SPARSE = auto()
    HYBRID = auto()
    MULTIVECTOR = auto()


# -- Config models --
# Pydantic models, so they validate inputs at runtime.


class RequestArgs(BaseModel):
    """
    filter: models.Filter is a Pydantic model from Qdrant client thatrepresents a query filter.
    It lets you express boolean filter logic for vector searches.
    It means we can constrain retrieval results, e.g. qdrant metadata.
    If user asks about Vand, we can restrict it to `category="vand"`.
    """

    collection_name: str
    limit: int = Field(ge=1)
    using: str | None = None
    score_threshold: float | None = Field(default=None, ge=0)
    filter: models.Filter | None = None


class RetrieverConfig(BaseModel):
    """
    Using Pydantic's BaseModel gives us
    1) Runtime validation.
      If type is not of type RetrievalType we can an error immediately.
    2) Serialization - easy to convert to dict/JSON (good for logging).
    We validate here because it defails with configuration data, like:
     - limits
     - thresholds
     - filters  etc..
    If these are wrong, we want to fail fast with a clear error message.
    We validate configs, as these aren't frequently changed, but intermediate objects
    can be created 100-1000 times per second, so validating might be overkill.
    """

    parser: list[ParserType]  # list: should be able to hold more retrievers as we scale
    type: RetrievalType
    request_args: RequestArgs


# -- Internal pipeline data structures --
# Describe the shape of internal dicts.
# Lightweight (just dicts) but type checkers warn if we forgot a key or uses wrong type.


class ParsedQuery(TypedDict):
    """
    A TypedDict is a type hint for dictionaries. It lets type checkers validate dict-shaped data,
    while keeping the actual value a plain dict.
    Here, ParsedQuery must have `parser: ParserType` and `text: str`.
    Catches more errors before production.
    """

    parser: ParserType
    text: str


class EmbeddedItem(TypedDict):
    """
    the `parser` key in the dict must be a ParserType enum value such as ParserType.QUERY_IDENTITY.
    A valid object looks like {"parser": ParserType.QUERY_IDENTITY, "text": "How does combat work?"}
    """

    parser: ParserType
    vector_type: VectorType
    text: str
    vector: Any


# -- Protocols --
# These define interface that other classes must implement.
# Let's us swap implementations, different parser, embedder, retriever
#  without changing the pipeline code.


class QueryParser(Protocol):
    ptype: ParserType

    async def ainvoke(self, payload: dict[str, Any]) -> str: ...


class Embedder(Protocol):
    async def aembed_batch(self, queries: list[str]) -> list[Any]: ...


class Retriever(Protocol):
    async def dense_batch_search(self, *, vectors: list[Any], **kwargs: Any) -> list[list[Any]]: ...
