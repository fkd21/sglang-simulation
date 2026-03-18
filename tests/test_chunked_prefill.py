"""Tests for chunked prefill in PrefillAdder and PrefillScheduler."""

import pytest
from scheduling.prefill_adder import SimPrefillAdder, AddReqResult
from scheduling.prefill_scheduler import PrefillScheduler
from memory.radix_cache import SimRadixCache
from memory.token_to_kv_pool import SimTokenToKVPool
from request.batch import SimBatch, ForwardMode
from request.request import SimReq, RequestStage
from instances.prefill_instance import PrefillInstance


def make_req(rid, context_tokens, generated_tokens=10):
    """Helper to create a SimReq with fill_ids set."""
    req = SimReq(rid, 0.0, context_tokens, generated_tokens)
    req.fill_ids = list(req.origin_input_ids)
    return req


def make_adder(max_prefill_tokens=16384, chunk_size=None, total_kv=100000):
    """Helper to create a SimPrefillAdder."""
    cache = SimRadixCache()
    pool = SimTokenToKVPool(total_kv_tokens=total_kv)
    batch = SimBatch(reqs=[], forward_mode=ForwardMode.PREFILL)
    return SimPrefillAdder(
        tree_cache=cache,
        token_to_kv_pool=pool,
        running_batch=batch,
        max_prefill_tokens=max_prefill_tokens,
        chunked_prefill_size=chunk_size,
    )


class TestPrefillAdderChunking:
    """Test chunked prefill in SimPrefillAdder."""

    def test_no_chunking_by_default(self):
        """When chunk_size is None, requests are never chunked."""
        adder = make_adder(max_prefill_tokens=1000, chunk_size=None)
        req = make_req("r1", 500)
        req.extend_input_len = 500

        result = adder.add_one_req(req)
        assert result == AddReqResult.CONTINUE
        assert adder.new_chunked_req is None
        assert req.is_chunked == 0

    def test_small_request_not_chunked(self):
        """Request smaller than chunk_size is not chunked."""
        adder = make_adder(max_prefill_tokens=16384, chunk_size=4096)
        req = make_req("r1", 2000)
        req.extend_input_len = 2000

        result = adder.add_one_req(req)
        assert result == AddReqResult.CONTINUE
        assert adder.new_chunked_req is None
        assert req.is_chunked == 0
        assert req.extend_input_len == 2000

    def test_request_exceeding_chunk_is_chunked(self):
        """Request larger than chunk_size is truncated and marked chunked."""
        adder = make_adder(max_prefill_tokens=16384, chunk_size=2048)
        req = make_req("r1", 5000)
        req.extend_input_len = 5000

        result = adder.add_one_req(req)
        # After using chunk budget, budget_state returns OTHER (chunk exhausted)
        assert result == AddReqResult.OTHER
        assert adder.new_chunked_req is req
        # is_chunked is now incremented by the scheduler, not the adder
        assert req.extend_input_len == 2048
        assert len(req.fill_ids) == 2048

    def test_chunk_budget_decreases(self):
        """rem_chunk_tokens decreases with each addition."""
        adder = make_adder(max_prefill_tokens=16384, chunk_size=4096)

        req1 = make_req("r1", 1000)
        req1.extend_input_len = 1000
        adder.add_one_req(req1)
        assert adder.rem_chunk_tokens == 3096

        req2 = make_req("r2", 2000)
        req2.extend_input_len = 2000
        adder.add_one_req(req2)
        assert adder.rem_chunk_tokens == 1096

    def test_chunk_budget_exhausted_returns_other(self):
        """When rem_chunk_tokens is 0, returns OTHER (not total memory issue)."""
        adder = make_adder(max_prefill_tokens=16384, chunk_size=1000)

        req1 = make_req("r1", 1000)
        req1.extend_input_len = 1000
        adder.add_one_req(req1)
        assert adder.rem_chunk_tokens == 0

        req2 = make_req("r2", 500)
        req2.extend_input_len = 500
        result = adder.add_one_req(req2)
        assert result == AddReqResult.OTHER

    def test_chunking_with_no_kv_memory(self):
        """Chunking returns NO_TOKEN when KV pool is too small."""
        adder = make_adder(max_prefill_tokens=16384, chunk_size=2048, total_kv=100)
        req = make_req("r1", 5000)
        req.extend_input_len = 5000

        result = adder.add_one_req(req)
        assert result == AddReqResult.NO_TOKEN

    def test_add_chunked_req_continuation(self):
        """add_chunked_req continues processing a chunked request."""
        adder = make_adder(max_prefill_tokens=16384, chunk_size=2048)
        req = make_req("r1", 5000)
        req.extend_input_len = 5000

        # First round: chunk it
        adder.add_one_req(req)
        # is_chunked is now incremented by the scheduler, not the adder
        assert adder.new_chunked_req is req
        assert req.extend_input_len == 2048

        # Simulate next round: create new adder, continue chunked req
        adder2 = make_adder(max_prefill_tokens=16384, chunk_size=2048)
        # Reconstruct fill_ids and extend_input_len for next round
        req.fill_ids = list(req.origin_input_ids)
        req.extend_input_len = 5000 - 2048  # Remaining after prefix
        result = adder2.add_chunked_req(req)

        # Still chunked (2952 > 2048)
        assert result is req
        assert req.extend_input_len == 2048
        assert len(adder2.can_run_list) == 1

    def test_add_chunked_req_last_chunk(self):
        """add_chunked_req completes when remaining fits in chunk."""
        adder = make_adder(max_prefill_tokens=16384, chunk_size=2048)
        req = make_req("r1", 3000)
        req.extend_input_len = 3000

        # First round: chunk it
        adder.add_one_req(req)
        assert adder.new_chunked_req is req
        assert req.extend_input_len == 2048

        # Second round: remaining 952 fits
        adder2 = make_adder(max_prefill_tokens=16384, chunk_size=2048)
        req.fill_ids = list(req.origin_input_ids)
        req.extend_input_len = 952  # 3000 - 2048
        result = adder2.add_chunked_req(req)

        # Should be None (complete, not chunked anymore)
        assert result is None
        assert len(adder2.can_run_list) == 1

    def test_multi_chunk_request(self):
        """Request requiring 3 chunks is chunked correctly across rounds."""
        chunk_size = 1000
        context = 2500  # Needs 3 chunks: 1000, 1000, 500

        # Round 1
        adder1 = make_adder(max_prefill_tokens=16384, chunk_size=chunk_size)
        req = make_req("r1", context)
        req.extend_input_len = context
        adder1.add_one_req(req)
        assert adder1.new_chunked_req is req
        assert req.extend_input_len == 1000

        # Round 2
        adder2 = make_adder(max_prefill_tokens=16384, chunk_size=chunk_size)
        req.fill_ids = list(req.origin_input_ids)
        req.extend_input_len = 1500  # 2500 - 1000
        result2 = adder2.add_chunked_req(req)
        assert result2 is req  # Still chunked
        assert req.extend_input_len == 1000

        # Round 3
        adder3 = make_adder(max_prefill_tokens=16384, chunk_size=chunk_size)
        req.fill_ids = list(req.origin_input_ids)
        req.extend_input_len = 500  # 2500 - 2000
        result3 = adder3.add_chunked_req(req)
        assert result3 is None  # Complete
        assert req.extend_input_len == 500

    def test_chunked_plus_normal_requests(self):
        """After chunking one request, another can fit in remaining budget."""
        adder = make_adder(max_prefill_tokens=16384, chunk_size=4096)

        # Large request gets chunked
        req1 = make_req("r1", 8000)
        req1.extend_input_len = 8000
        adder.add_one_req(req1)
        assert adder.new_chunked_req is req1
        assert adder.rem_chunk_tokens == 0  # Chunk budget used up

        # Small request can't fit (chunk budget exhausted)
        req2 = make_req("r2", 500)
        req2.extend_input_len = 500
        result = adder.add_one_req(req2)
        assert result == AddReqResult.OTHER  # Chunk budget exhausted (not total memory)

    def test_input_budget_limits_chunk(self):
        """First request is allowed even if over input budget, gets chunked."""
        adder = make_adder(max_prefill_tokens=1500, chunk_size=2048)
        req = make_req("r1", 3000)
        req.extend_input_len = 3000

        # rem_input_tokens = 1500 < rem_chunk_tokens = 2048
        # request is 3000 > both
        # First request is allowed even if over input budget (can_run_list empty)
        # Gets chunked to chunk_size (2048), exhausting both budgets
        result = adder.add_one_req(req)
        assert adder.new_chunked_req is req
        assert req.extend_input_len == 2048
        assert result == AddReqResult.OTHER  # Budget exhausted after chunking


class TestPrefillSchedulerChunking:
    """Test chunked prefill in PrefillScheduler."""

    def _make_instance(self, total_kv=100000):
        """Create a PrefillInstance."""
        inst = PrefillInstance("prefill_0")
        inst.token_to_kv_pool = SimTokenToKVPool(total_kv_tokens=total_kv)
        return inst

    def test_scheduler_no_chunking(self):
        """Without chunking, scheduler processes normally."""
        inst = self._make_instance()
        scheduler = PrefillScheduler(
            inst, max_prefill_tokens=16384, chunked_prefill_size=None
        )

        req = make_req("r1", 500)
        inst.add_request(req)

        batch = scheduler.get_next_batch()
        assert batch is not None
        assert len(batch.reqs) == 1
        assert scheduler.chunked_req is None

    def test_scheduler_chunks_large_request(self):
        """Scheduler chunks request exceeding chunk size."""
        inst = self._make_instance()
        scheduler = PrefillScheduler(
            inst, max_prefill_tokens=16384, chunked_prefill_size=2048
        )

        req = make_req("r1", 5000)
        inst.add_request(req)

        batch = scheduler.get_next_batch()
        assert batch is not None
        assert len(batch.reqs) == 1
        # Scheduler should track chunked request
        assert scheduler.chunked_req is req

    def test_scheduler_continuation_across_rounds(self):
        """Scheduler continues chunked request in next round."""
        inst = self._make_instance()
        scheduler = PrefillScheduler(
            inst, max_prefill_tokens=16384, chunked_prefill_size=2048
        )

        req = make_req("r1", 5000)
        inst.add_request(req)

        # Round 1: chunk
        batch1 = scheduler.get_next_batch()
        assert batch1 is not None
        assert scheduler.chunked_req is req

        # Simulate engine freeing memory after batch complete
        inst.free_memory(req)
        req.prefill_tokens_done += batch1.reqs[0].extend_input_len
        inst.busy = False

        # Round 2: continue
        batch2 = scheduler.get_next_batch()
        assert batch2 is not None
        assert len(batch2.reqs) == 1

    def test_scheduler_init_next_round_input_with_prefill_done(self):
        """_init_next_round_input accounts for prefill_tokens_done."""
        inst = self._make_instance()
        scheduler = PrefillScheduler(
            inst, max_prefill_tokens=16384, chunked_prefill_size=2048
        )

        req = make_req("r1", 5000)
        req.prefill_tokens_done = 2000

        scheduler._init_next_round_input(req)

        # extend_input_len should be 5000 - 2000 = 3000
        assert req.extend_input_len == 3000

    def test_scheduler_init_next_round_input_fresh(self):
        """_init_next_round_input with no progress computes full length."""
        inst = self._make_instance()
        scheduler = PrefillScheduler(
            inst, max_prefill_tokens=16384
        )

        req = make_req("r1", 500)
        scheduler._init_next_round_input(req)

        # No prefix match, full length
        assert req.extend_input_len == 500

    def test_scheduler_multiple_requests_with_chunking(self):
        """Multiple requests with chunking: small ones fit, large gets chunked."""
        inst = self._make_instance()
        scheduler = PrefillScheduler(
            inst, max_prefill_tokens=16384, chunked_prefill_size=4096
        )

        req1 = make_req("r1", 1000)
        req2 = make_req("r2", 6000)  # Will be chunked
        req3 = make_req("r3", 500)

        inst.add_request(req1)
        inst.add_request(req2)
        inst.add_request(req3)

        batch = scheduler.get_next_batch()
        assert batch is not None
        # req1 should fit (1000 < 4096)
        # req2 should be chunked (6000 > remaining chunk budget)
        assert len(batch.reqs) >= 1

    def test_scheduler_decay_token_ratio(self):
        """decay_token_ratio reduces new_token_ratio."""
        inst = self._make_instance()
        scheduler = PrefillScheduler(
            inst, max_prefill_tokens=16384, new_token_ratio=0.5
        )

        initial = scheduler.new_token_ratio
        scheduler.decay_token_ratio()
        assert scheduler.new_token_ratio < initial

    def test_scheduler_empty_queue_returns_none(self):
        """Empty queue returns None."""
        inst = self._make_instance()
        scheduler = PrefillScheduler(inst, max_prefill_tokens=16384)

        batch = scheduler.get_next_batch()
        assert batch is None

    def test_scheduler_batch_mode_is_prefill(self):
        """Returned batch has PREFILL forward mode."""
        inst = self._make_instance()
        scheduler = PrefillScheduler(inst, max_prefill_tokens=16384)

        req = make_req("r1", 100)
        inst.add_request(req)

        batch = scheduler.get_next_batch()
        assert batch.forward_mode == ForwardMode.PREFILL

    def test_scheduler_request_removed_from_waiting(self):
        """Completed (non-chunked) request is removed from waiting queue."""
        inst = self._make_instance()
        scheduler = PrefillScheduler(inst, max_prefill_tokens=16384)

        req = make_req("r1", 100)
        inst.add_request(req)
        assert len(inst.waiting_queue) == 1

        scheduler.get_next_batch()
        assert len(inst.waiting_queue) == 0

    def test_scheduler_request_stage_updated(self):
        """Request stage is set to PREFILL_FORWARD after scheduling."""
        inst = self._make_instance()
        scheduler = PrefillScheduler(inst, max_prefill_tokens=16384)

        req = make_req("r1", 100)
        inst.add_request(req)

        scheduler.get_next_batch()
        assert req.stage == RequestStage.PREFILL_FORWARD
