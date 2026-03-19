from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class QdrantSettings(BaseSettings):
    QDRANT_URL: str = Field(default="http://localhost:6333")
    QDRANT_API_KEY: str | None = Field(default=None)
    COLLECTION: str = Field(default="boardgame_rules_v0")
    # Vector field names in Qdrant
    DENSE_VECTOR_NAME: str = Field(default="dense")
    SPARSE_VECTOR_NAME: str = Field(default="bm25")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="allow")


@lru_cache
def get_settings() -> QdrantSettings:
    return QdrantSettings()


qdrant_settings: QdrantSettings = get_settings()
