"""Test memory pool implementations."""

import pytest
from memory.token_to_kv_pool import SimTokenToKVPool
from memory.req_to_token_pool import SimReqToTokenPool


class TestTokenToKVPool:
    """Test KV cache token pool."""

    def test_initialization(self):
        """Test pool initialization."""
        pool = SimTokenToKVPool(total_kv_tokens=1000)
        assert pool.total_kv_tokens == 1000
        assert pool.free_kv_tokens == 1000

    def test_allocate_success(self):
        """Test successful allocation."""
        pool = SimTokenToKVPool(total_kv_tokens=1000)
        indices = pool.alloc(100)
        assert len(indices) == 100
        assert pool.free_kv_tokens == 900
        assert all(0 <= i < 1000 for i in indices)

    def test_allocate_failure(self):
        """Test allocation failure when insufficient memory."""
        pool = SimTokenToKVPool(total_kv_tokens=1000)
        pool.alloc(800)
        # Try to allocate more than available
        indices = pool.alloc(300)
        assert indices is None
        assert pool.free_kv_tokens == 200

    def test_can_allocate(self):
        """Test can_allocate check."""
        pool = SimTokenToKVPool(total_kv_tokens=1000)
        assert pool.can_allocate(500) is True
        pool.alloc(800)
        assert pool.can_allocate(300) is False
        assert pool.can_allocate(200) is True

    def test_free_memory(self):
        """Test freeing allocated memory."""
        pool = SimTokenToKVPool(total_kv_tokens=1000)
        indices = pool.alloc(100)
        assert pool.free_kv_tokens == 900

        pool.free(indices)
        assert pool.free_kv_tokens == 1000

    def test_multiple_allocations(self):
        """Test multiple allocations and deallocations."""
        pool = SimTokenToKVPool(total_kv_tokens=1000)

        indices1 = pool.alloc(100)
        indices2 = pool.alloc(200)
        indices3 = pool.alloc(300)

        assert pool.free_kv_tokens == 400

        pool.free(indices2)
        assert pool.free_kv_tokens == 600

        indices4 = pool.alloc(150)
        assert len(indices4) == 150
        assert pool.free_kv_tokens == 450

    def test_allocate_zero(self):
        """Test allocating zero tokens."""
        pool = SimTokenToKVPool(total_kv_tokens=1000)
        indices = pool.alloc(0)
        assert indices == []
        assert pool.free_kv_tokens == 1000


class TestReqToTokenPool:
    """Test request to token mapping pool."""

    def test_initialization(self):
        """Test pool initialization."""
        pool = SimReqToTokenPool(size=100)
        assert pool.size == 100
        assert pool.available_size() == 100

    def test_allocate_request_slot(self):
        """Test allocating request slot."""
        pool = SimReqToTokenPool(size=100)
        idx = pool.alloc()
        assert idx == 0
        assert pool.available_size() == 99

        idx2 = pool.alloc()
        assert idx2 == 1
        assert pool.available_size() == 98

    def test_free_request_slot(self):
        """Test freeing request slot."""
        pool = SimReqToTokenPool(size=100)
        idx1 = pool.alloc()
        idx2 = pool.alloc()
        idx3 = pool.alloc()

        assert pool.available_size() == 97

        pool.free(idx2)
        assert pool.available_size() == 98

        # Next allocation should reuse freed slot
        idx4 = pool.alloc()
        assert idx4 == 1  # Reused idx2
        assert pool.available_size() == 97

    def test_allocation_limit(self):
        """Test allocation limit."""
        pool = SimReqToTokenPool(size=5)

        for i in range(5):
            idx = pool.alloc()
            assert idx == i

        # Should fail to allocate beyond limit
        idx = pool.alloc()
        assert idx is None

    def test_can_allocate(self):
        """Test available slots check."""
        pool = SimReqToTokenPool(size=5)

        assert pool.available_size() > 0

        for _ in range(5):
            pool.alloc()

        assert pool.available_size() == 0
