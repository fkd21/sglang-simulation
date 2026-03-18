"""Pytest configuration and fixtures."""

import pytest
from request.request import SimReq
from memory.radix_cache import SimRadixCache
from memory.token_to_kv_pool import SimTokenToKVPool


@pytest.fixture
def sample_request():
    """Create a sample request for testing."""
    return SimReq(
        rid="test_req_1",
        arrival_time=1.0,
        context_tokens=100,
        generated_tokens=50
    )


@pytest.fixture
def sample_requests():
    """Create multiple sample requests for testing."""
    return [
        SimReq(f"req_{i}", float(i), 100 + i*10, 50)
        for i in range(5)
    ]


@pytest.fixture
def radix_cache():
    """Create a fresh radix cache."""
    return SimRadixCache()


@pytest.fixture
def kv_pool():
    """Create a KV pool with 10000 tokens."""
    return SimTokenToKVPool(max_tokens=10000)


@pytest.fixture
def small_kv_pool():
    """Create a small KV pool for testing memory pressure."""
    return SimTokenToKVPool(max_tokens=1000)
