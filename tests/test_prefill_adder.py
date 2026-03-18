"""Test PrefillAdder admission control."""

import pytest
from scheduling.prefill_adder import SimPrefillAdder, AddReqResult
from memory.radix_cache import SimRadixCache
from memory.token_to_kv_pool import SimTokenToKVPool
from request.batch import SimBatch, ForwardMode
from request.request import SimReq


class TestPrefillAdder:
    """Test PrefillAdder admission control."""

    def test_initialization(self):
        """Test adder initialization."""
        cache = SimRadixCache()
        pool = SimTokenToKVPool(total_kv_tokens=10000)
        batch = SimBatch(reqs=[], forward_mode=ForwardMode.PREFILL)

        adder = SimPrefillAdder(
            tree_cache=cache,
            token_to_kv_pool=pool,
            running_batch=batch,
            max_prefill_tokens=16384
        )

        assert adder.rem_input_tokens == 16384
        assert adder.rem_chunk_tokens is None  # No chunking by default

    def test_add_request_success(self):
        """Test successfully adding a request."""
        cache = SimRadixCache()
        pool = SimTokenToKVPool(total_kv_tokens=10000)
        batch = SimBatch(reqs=[], forward_mode=ForwardMode.PREFILL)

        adder = SimPrefillAdder(
            tree_cache=cache,
            token_to_kv_pool=pool,
            running_batch=batch,
            max_prefill_tokens=1000
        )

        req = SimReq("req_1", 0.0, 100, 50)
        req.extend_input_len = 100

        result = adder.add_one_req(req)

        assert result == AddReqResult.CONTINUE
        assert len(adder.can_run_list) == 1
        assert adder.rem_input_tokens == 900

    def test_add_request_no_token_budget(self):
        """Test first request allowed even if over input budget (matches real SGLang)."""
        cache = SimRadixCache()
        pool = SimTokenToKVPool(total_kv_tokens=10000)
        batch = SimBatch(reqs=[], forward_mode=ForwardMode.PREFILL)

        adder = SimPrefillAdder(
            tree_cache=cache,
            token_to_kv_pool=pool,
            running_batch=batch,
            max_prefill_tokens=500
        )

        req = SimReq("req_1", 0.0, 1000, 50)
        req.extend_input_len = 1000

        result = adder.add_one_req(req)

        # First request is always admitted even if over input budget
        assert len(adder.can_run_list) == 1
        assert result == AddReqResult.OTHER  # Budget exhausted after admission

        # Second request should be rejected
        req2 = SimReq("req_2", 0.1, 200, 30)
        req2.extend_input_len = 200
        result2 = adder.add_one_req(req2)
        assert result2 == AddReqResult.OTHER  # Input budget exhausted
        assert len(adder.can_run_list) == 1

    def test_add_request_no_memory(self):
        """Test rejection due to insufficient KV memory."""
        cache = SimRadixCache()
        pool = SimTokenToKVPool(total_kv_tokens=100)  # Very small
        batch = SimBatch(reqs=[], forward_mode=ForwardMode.PREFILL)

        adder = SimPrefillAdder(
            tree_cache=cache,
            token_to_kv_pool=pool,
            running_batch=batch,
            max_prefill_tokens=1000
        )

        req = SimReq("req_1", 0.0, 500, 50)
        req.extend_input_len = 500

        result = adder.add_one_req(req)

        assert result == AddReqResult.NO_TOKEN
        assert len(adder.can_run_list) == 0

    def test_add_multiple_requests(self):
        """Test adding multiple requests."""
        cache = SimRadixCache()
        pool = SimTokenToKVPool(total_kv_tokens=10000)
        batch = SimBatch(reqs=[], forward_mode=ForwardMode.PREFILL)

        adder = SimPrefillAdder(
            tree_cache=cache,
            token_to_kv_pool=pool,
            running_batch=batch,
            max_prefill_tokens=1000
        )

        req1 = SimReq("req_1", 0.0, 300, 50)
        req1.extend_input_len = 300

        req2 = SimReq("req_2", 0.1, 400, 50)
        req2.extend_input_len = 400

        req3 = SimReq("req_3", 0.2, 500, 50)
        req3.extend_input_len = 500

        # Add first two (total 700 tokens)
        result1 = adder.add_one_req(req1)
        result2 = adder.add_one_req(req2)

        assert result1 == AddReqResult.CONTINUE
        assert result2 == AddReqResult.CONTINUE
        assert len(adder.can_run_list) == 2
        assert adder.rem_input_tokens == 300

        # Third should fail (needs 500, only 300 input budget remaining)
        result3 = adder.add_one_req(req3)

        assert result3 == AddReqResult.OTHER  # Input budget exceeded (not total memory)
        assert len(adder.can_run_list) == 2

    def test_eviction_to_free_memory(self):
        """Test automatic eviction when memory is full."""
        cache = SimRadixCache()
        pool = SimTokenToKVPool(total_kv_tokens=1000)
        batch = SimBatch(reqs=[], forward_mode=ForwardMode.PREFILL)

        # Fill cache with some entries (token_ids and kv_indices must match in length)
        tokens_1 = list(range(300))
        cache.insert(tokens_1, list(range(300)))
        pool.alloc(300)
        tokens_2 = list(range(300, 600))
        cache.insert(tokens_2, list(range(300, 600)))
        pool.alloc(300)

        # Now pool has 400 free tokens, cache has 600 evictable

        adder = SimPrefillAdder(
            tree_cache=cache,
            token_to_kv_pool=pool,
            running_batch=batch,
            max_prefill_tokens=2000
        )

        # Try to add request needing 600 tokens (more than 400 available)
        req = SimReq("req_1", 0.0, 600, 50)
        req.extend_input_len = 600

        result = adder.add_one_req(req)

        # Should succeed after eviction
        assert result == AddReqResult.CONTINUE
        assert len(adder.can_run_list) == 1

    def test_get_can_run_list(self):
        """Test retrieving admitted requests."""
        cache = SimRadixCache()
        pool = SimTokenToKVPool(total_kv_tokens=10000)
        batch = SimBatch(reqs=[], forward_mode=ForwardMode.PREFILL)

        adder = SimPrefillAdder(
            tree_cache=cache,
            token_to_kv_pool=pool,
            running_batch=batch,
            max_prefill_tokens=1000
        )

        req1 = SimReq("req_1", 0.0, 100, 50)
        req1.extend_input_len = 100
        req2 = SimReq("req_2", 0.1, 200, 50)
        req2.extend_input_len = 200

        adder.add_one_req(req1)
        adder.add_one_req(req2)

        can_run = adder.get_can_run_list()

        assert len(can_run) == 2
        assert req1 in can_run
        assert req2 in can_run

    def test_token_budget_tracking(self):
        """Test that token budgets are correctly tracked."""
        cache = SimRadixCache()
        pool = SimTokenToKVPool(total_kv_tokens=10000)
        batch = SimBatch(reqs=[], forward_mode=ForwardMode.PREFILL)

        adder = SimPrefillAdder(
            tree_cache=cache,
            token_to_kv_pool=pool,
            running_batch=batch,
            max_prefill_tokens=1000,
            chunked_prefill_size=1000,
        )

        assert adder.rem_input_tokens == 1000
        assert adder.rem_chunk_tokens == 1000

        req = SimReq("req_1", 0.0, 300, 50)
        req.extend_input_len = 300

        adder.add_one_req(req)

        assert adder.rem_input_tokens == 700
        assert adder.rem_chunk_tokens == 700
