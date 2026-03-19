from functools import lru_cache

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    CHAT_MODEL: str = Field(default="gpt-4o-mini")
    TOP_K: int = Field(default=5)
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="allow")
    # Dense
    DENSE_EMBED_MODEL: str = Field(default="text-embedding-3-small")
    # Sparse embedding settings
    SPARSE_EMBED_MODEL: str = Field(default="Qdrant/bm25")
    SPARSE_BATCH_SIZE: int = Field(default=32)


@lru_cache
def get_settings() -> Settings:
    return Settings()


retrieval_settings: Settings = get_settings()
