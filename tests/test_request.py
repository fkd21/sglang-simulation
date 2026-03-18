"""Test request lifecycle and state transitions."""

import pytest
from request.request import SimReq, RequestStage


class TestSimReq:
    """Test SimReq class."""

    def test_initialization(self):
        """Test request initialization."""
        req = SimReq(
            rid="req_1",
            arrival_time=0.5,
            context_tokens=100,
            generated_tokens=50
        )

        assert req.rid == "req_1"
        assert req.arrival_time == 0.5
        assert req.context_tokens == 100
        assert req.generated_tokens == 50
        assert req.stage == RequestStage.WAITING
        assert req.finished is False

    def test_auto_generate_token_ids(self):
        """Test automatic token ID generation."""
        req = SimReq(
            rid="req_1",
            arrival_time=0.0,
            context_tokens=100,
            generated_tokens=50
        )

        # Should auto-generate origin_input_ids
        assert len(req.origin_input_ids) == 100
        # Should be unique based on rid hash
        assert req.origin_input_ids[0] != 0  # Not starting from 0

    def test_unique_token_ids_per_request(self):
        """Test that different requests have different token IDs."""
        req1 = SimReq(
            rid="req_1",
            arrival_time=0.0,
            context_tokens=100,
            generated_tokens=50
        )

        req2 = SimReq(
            rid="req_2",
            arrival_time=0.0,
            context_tokens=100,
            generated_tokens=50
        )

        # Different rids should produce different token sequences
        assert req1.origin_input_ids != req2.origin_input_ids

    def test_extend_input_len_initialization(self):
        """Test extend_input_len is initialized to context_tokens."""
        req = SimReq(
            rid="req_1",
            arrival_time=0.0,
            context_tokens=100,
            generated_tokens=50
        )

        assert req.extend_input_len == 100

    def test_stage_transitions(self):
        """Test valid stage transitions."""
        req = SimReq(
            rid="req_1",
            arrival_time=0.0,
            context_tokens=100,
            generated_tokens=50
        )

        # WAITING -> PREFILL_FORWARD
        assert req.stage == RequestStage.WAITING
        req.stage = RequestStage.PREFILL_FORWARD
        assert req.stage == RequestStage.PREFILL_FORWARD

        # PREFILL_FORWARD -> PREFILL_TRANSFER_KV_CACHE
        req.stage = RequestStage.PREFILL_TRANSFER_KV_CACHE
        assert req.stage == RequestStage.PREFILL_TRANSFER_KV_CACHE

        # PREFILL_TRANSFER_KV_CACHE -> DECODE_WAITING
        req.stage = RequestStage.DECODE_WAITING
        assert req.stage == RequestStage.DECODE_WAITING

        # DECODE_WAITING -> DECODE_FORWARD
        req.stage = RequestStage.DECODE_FORWARD
        assert req.stage == RequestStage.DECODE_FORWARD

        # DECODE_FORWARD -> COMPLETED
        req.stage = RequestStage.COMPLETED
        assert req.stage == RequestStage.COMPLETED

    def test_latency_tracking(self):
        """Test latency timestamp tracking."""
        req = SimReq(
            rid="req_1",
            arrival_time=1.0,
            context_tokens=100,
            generated_tokens=50
        )

        req.prefill_start_time = 2.0
        req.prefill_end_time = 3.0
        req.decode_start_time = 4.0
        req.decode_end_time = 10.0

        # TTFT = prefill_end_time - arrival_time
        ttft = req.prefill_end_time - req.arrival_time
        assert ttft == 2.0

        # Decode time
        decode_time = req.decode_end_time - req.decode_start_time
        assert decode_time == 6.0

        # E2E latency
        e2e = req.decode_end_time - req.arrival_time
        assert e2e == 9.0

    def test_output_generation(self):
        """Test output token generation tracking."""
        req = SimReq(
            rid="req_1",
            arrival_time=0.0,
            context_tokens=100,
            generated_tokens=50
        )

        assert req.decode_tokens_generated == 0
        assert len(req.output_ids) == 0

        # Simulate generating tokens
        for i in range(50):
            req.output_ids.append(i)
            req.decode_tokens_generated += 1

        assert req.decode_tokens_generated == 50
        assert len(req.output_ids) == 50

    def test_memory_allocation_tracking(self):
        """Test KV cache memory tracking."""
        req = SimReq(
            rid="req_1",
            arrival_time=0.0,
            context_tokens=100,
            generated_tokens=50
        )

        # Initially no allocation
        assert req.kv_cache_indices == []
        assert req.req_pool_idx is None

        # Simulate allocation
        req.kv_cache_indices = list(range(0, 100))
        req.req_pool_idx = 5

        assert len(req.kv_cache_indices) == 100
        assert req.req_pool_idx == 5

    def test_prefix_matching_tracking(self):
        """Test prefix matching tracking."""
        req = SimReq(
            rid="req_1",
            arrival_time=0.0,
            context_tokens=100,
            generated_tokens=50
        )

        # Simulate prefix match
        req.prefix_indices = [0, 1, 2, 3, 4]
        req.extend_input_len = 95  # 100 - 5 matched

        assert len(req.prefix_indices) == 5
        assert req.extend_input_len == 95

    def test_new_mechanism_tracking(self):
        """Test tracking for new mechanisms."""
        req = SimReq(
            rid="req_1",
            arrival_time=0.0,
            context_tokens=100,
            generated_tokens=50
        )

        assert req.partial_offload_amount == 0
        assert req.continued_on_prefill == 0

        # Simulate partial offload
        req.partial_offload_amount = 20
        assert req.partial_offload_amount == 20

        # Simulate continuation
        req.continued_on_prefill = 10
        assert req.continued_on_prefill == 10
