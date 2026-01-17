from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    QDRANT_URL: str = Field(default="http://localhost:6333")
    COLLECTION: str = Field(default="boardgame_rules_v0")
    EMBED_MODEL: str = Field(default="text-embedding-3-small")
    CHAT_MODEL: str = Field(default="gpt-4o-mini")
    TOP_K: int = Field(default=5)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings: Settings = get_settings()
