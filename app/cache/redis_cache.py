"""Redis caching layer for the Evaluator-Generator platform.

The cache reduces expensive operations (document extraction, embeddings,
retrieval results, generated answers, evaluation results). It is designed
to degrade gracefully: if Redis is unreachable or disabled, every method
returns ``None``/``False`` and the caller transparently falls back to
computing the value.

Caching discipline
-------------------
* Every cached value is namespaced (``doc:``, ``retrieval:``, ``answer:``,
  ``eval:``) so unrelated data never collides.
* Cache keys are hashes of the inputs, so a change in the underlying source
  or question produces a different key and never returns stale data.
* A TTL is applied to all entries so that long-lived entries expire.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import redis

from app.config import settings
from app.utils.logging import get_logger

logger = get_logger("cache")


class RedisCache:
    """Thin, fault-tolerant wrapper around a Redis client."""

    def __init__(
        self,
        url: str | None = None,
        ttl: int | None = None,
        enabled: bool | None = None,
    ) -> None:
        self._url = url or settings.REDIS_URL
        self._ttl = ttl if ttl is not None else settings.CACHE_TTL
        self._enabled = enabled if enabled is not None else settings.REDIS_ENABLED
        self._client: redis.Redis | None = None

        if self._enabled:
            self._connect()

    # ------------------------------------------------------------------ #
    # Connection handling
    # ------------------------------------------------------------------ #
    def _connect(self) -> None:
        try:
            client = redis.Redis.from_url(
                self._url,
                socket_connect_timeout=2,
                socket_timeout=2,
                decode_responses=True,
            )
            client.ping()
            self._client = client
            logger.info("Redis cache connected at %s", self._url)
        except Exception as exc:  # pragma: no cover - depends on environment
            self._client = None
            logger.warning(
                "Redis unavailable (%s). Running with caching disabled.", exc
            )

    @property
    def enabled(self) -> bool:
        """Whether caching is active (client reachable)."""
        return self._client is not None

    @property
    def client(self) -> redis.Redis | None:
        """Expose the raw client (used by callers that need raw access)."""
        return self._client

    # ------------------------------------------------------------------ #
    # Low level helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _hash(*parts: str) -> str:
        joined = "|".join(str(p) for p in parts)
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()

    def _get_json(self, key: str) -> Any | None:
        if self._client is None:
            return None
        try:
            raw = self._client.get(key)
        except Exception as exc:  # pragma: no cover - network failure
            logger.warning("Redis GET failed: %s", exc)
            return None
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None

    def _set_json(self, key: str, value: Any, ttl: int | None = None) -> None:
        if self._client is None:
            return
        try:
            self._client.setex(
                key,
                ttl if ttl is not None else self._ttl,
                json.dumps(value, default=str),
            )
        except Exception as exc:  # pragma: no cover - network failure
            logger.warning("Redis SET failed: %s", exc)

    # ------------------------------------------------------------------ #
    # Semantic caches
    # ------------------------------------------------------------------ #
    def get_processed_document(self, source_id: str) -> list[dict] | None:
        """Return cached, extracted document chunks for a source id."""
        return self._get_json(f"doc:{self._hash(source_id)}")

    def cache_processed_document(self, source_id: str, chunks: list[dict]) -> None:
        self._set_json(f"doc:{self._hash(source_id)}", chunks)

    def get_retrieval(self, question: str, version: str = "") -> dict | None:
        """Return cached retrieval for an identical question + KB version."""
        return self._get_json(f"retrieval:{self._hash(question, version)}")

    def cache_retrieval(
        self, question: str, result: dict, version: str = ""
    ) -> None:
        self._set_json(f"retrieval:{self._hash(question, version)}", result)

    def get_answer(
        self, question: str, context_hash: str, version: str = ""
    ) -> dict | None:
        """Return a cached workflow answer for question + context + version."""
        return self._get_json(
            f"answer:{self._hash(question, context_hash, version)}"
        )

    def cache_answer(
        self,
        question: str,
        context_hash: str,
        result: dict,
        version: str = "",
    ) -> None:
        self._set_json(
            f"answer:{self._hash(question, context_hash, version)}", result
        )

    def clear_namespace(self, prefix: str) -> int:
        """Delete all keys with the given prefix. Returns count removed."""
        if self._client is None:
            return 0
        try:
            keys = list(self._client.scan_iter(match=f"{prefix}*"))
            if keys:
                return self._client.delete(*keys)
        except Exception as exc:  # pragma: no cover - network failure
            logger.warning("Redis clear failed: %s", exc)
        return 0

    def clear_all(self) -> None:
        """Clear all platform cache keys."""
        self.clear_namespace("doc:")
        self.clear_namespace("retrieval:")
        self.clear_namespace("answer:")
        self.clear_namespace("eval:")


def create_redis_cache() -> RedisCache:
    """Build a RedisCache from application settings."""
    return RedisCache()
