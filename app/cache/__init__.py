"""Initialization for the caching layer."""

from __future__ import annotations

from app.cache.redis_cache import RedisCache, create_redis_cache

__all__ = ["RedisCache", "create_redis_cache"]
