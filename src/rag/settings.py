from functools import lru_cache

from dotenv import load_dotenv
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    CHAT_MODEL: str = Field(default="gpt-4o-mini")
    TOP_K: int = Field(default=5, ge=1)
    CANDIDATE_LIMIT: int = Field(default=20, ge=1)
    QUERY_EXPANSION_MAX: int = Field(default=3, ge=1)
    RERANK_ENABLED: bool = Field(default=True)
    RERANK_MODEL: str = Field(default="gpt-4o-mini")
    RERANK_CANDIDATE_K: int = Field(default=10, ge=1)
    RERANK_MAX_CONCURRENCY: int = Field(default=5, ge=1)
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="allow")
    # Dense
    DENSE_EMBED_MODEL: str = Field(default="text-embedding-3-small")
    # Sparse embedding settings
    SPARSE_EMBED_MODEL: str = Field(default="Qdrant/bm25")
    SPARSE_BATCH_SIZE: int = Field(default=32)

    @model_validator(mode="after")
    def validate_retrieval_limits(self) -> "Settings":
        if self.CANDIDATE_LIMIT < self.TOP_K:
            raise ValueError("CANDIDATE_LIMIT must be greater than or equal to TOP_K")
        if self.RERANK_CANDIDATE_K < self.TOP_K:
            raise ValueError("RERANK_CANDIDATE_K must be greater than or equal to TOP_K")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()


retrieval_settings: Settings = get_settings()
