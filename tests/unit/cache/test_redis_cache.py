"""Tests for the Redis caching layer (no real Redis required)."""

from unittest.mock import MagicMock

from app.cache.redis_cache import RedisCache


def _make_cache(enabled=True, client=None):
    cache = RedisCache(enabled=enabled)
    if client is not None:
        cache._client = client
    return cache


def test_disabled_cache_returns_none_and_noops():
    cache = _make_cache(enabled=False)
    assert cache.enabled is False
    assert cache.get_processed_document("x") is None
    # Should not raise even with no client.
    cache.cache_processed_document("x", [{"page_content": "a", "metadata": {}}])
    assert cache.get_answer("q", "h") is None


def test_enabled_cache_roundtrip():
    client = MagicMock()
    client.get.return_value = None
    cache = _make_cache(client=client)

    assert cache.enabled is True
    cache.cache_processed_document(
        "src1", [{"page_content": "hello", "metadata": {"source_type": "pdf"}}]
    )
    client.setex.assert_called_once()

    # Simulate a populated cache.
    client.get.return_value = '[{"page_content": "hello", "metadata": {"source_type": "pdf"}}]'
    result = cache.get_processed_document("src1")
    assert result == [{"page_content": "hello", "metadata": {"source_type": "pdf"}}]


def test_answer_cache_shortcircuit():
    client = MagicMock()
    client.get.return_value = '{"final_answer": "cached", "validated": true, "iterations": 2}'
    cache = _make_cache(client=client)

    got = cache.get_answer("question", "hash")
    assert got["final_answer"] == "cached"
    assert got["validated"] is True


def test_retrieval_cache_roundtrip():
    client = MagicMock()
    client.get.return_value = None
    cache = _make_cache(client=client)
    payload = {"retrieved_context": "ctx", "sources": []}
    cache.cache_retrieval("q", payload)
    client.setex.assert_called_once()

    client.get.return_value = '{"retrieved_context": "ctx", "sources": []}'
    assert cache.get_retrieval("q") == payload


def test_keys_are_namespaced_and_hashed():
    client = MagicMock()
    client.get.return_value = None
    cache = _make_cache(client=client)
    cache.cache_processed_document("alpha", [])
    key = client.setex.call_args.args[0]
    assert key.startswith("doc:")

    cache.cache_retrieval("beta", {})
    key = client.setex.call_args.args[0]
    assert key.startswith("retrieval:")
