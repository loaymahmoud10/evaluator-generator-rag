"""Central configuration loaded from environment variables.

All tunable parameters for the Evaluator-Generator platform live here so
that the UI, CLI, and tests share a single source of truth.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


def _get_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class Settings:
    """Read-only view of the platform configuration."""

    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "").strip()
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()
    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "whisper-large-v3-turbo").strip()

    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379").strip()
    REDIS_ENABLED: bool = _get_bool("REDIS_ENABLED", default=True)

    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    ).strip()
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store").strip()
    CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "knowledge").strip()

    MAX_ITERATIONS: int = int(os.getenv("MAX_ITERATIONS", "4").strip() or "4")
    TOP_K: int = int(os.getenv("TOP_K", "4").strip() or "4")
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000").strip() or "1000")
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200").strip() or "200")

    CACHE_TTL: int = int(os.getenv("CACHE_TTL", "3600").strip() or "3600")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").strip().upper()

    @property
    def has_groq(self) -> bool:
        return bool(self.GROQ_API_KEY)


settings = Settings()
