"""Prefill scheduler with chunked prefill and mixed-chunk support.

Mirrors sglang.srt.managers.scheduler.Scheduler::get_new_batch_prefill
"""

from __future__ import annotations

from typing import Optional

from simulation.instances.base_instance import SimInstance
from simulation.request.batch import ForwardMode, SimBatch
from simulation.request.request import RequestStage, SimReq
from simulation.scheduling.prefill_adder import AddReqResult, SimPrefillAdder
from simulation.scheduling.schedule_policy import SimSchedulePolicy


class PrefillScheduler:
    """Prefill scheduler mirroring SGLang's logic with chunked prefill."""

    def __init__(
        self,
        instance: SimInstance,
        max_prefill_tokens: int = 16384,
        chunked_prefill_size: Optional[int] = None,
        new_token_ratio: float = 0.0,
        schedule_policy: str = "fcfs",
        enable_mixed_chunk: bool = False,
        max_running_requests: int = 4096,
    ):
        """Initialize prefill scheduler.

        Args:
            instance: Instance to schedule for
            max_prefill_tokens: Maximum prefill tokens per batch
            chunked_prefill_size: Max tokens per chunk (None = no chunking)
            new_token_ratio: Schedule conservativeness (0 = no conservativeness)
            schedule_policy: Scheduling policy ("fcfs", "lpm")
            enable_mixed_chunk: Enable mixed-chunk scheduling (prefill + decode)
            max_running_requests: Maximum concurrent running requests
        """
        self.instance = instance
        self.max_prefill_tokens = max_prefill_tokens
        self.chunked_prefill_size = chunked_prefill_size
        self.new_token_ratio = new_token_ratio
        self.enable_mixed_chunk = enable_mixed_chunk
        self.max_running_requests = max_running_requests

        self.policy = SimSchedulePolicy(
            policy=schedule_policy,
            tree_cache=instance.tree_cache,
        )

        # Chunked request state: tracks a request being processed across rounds
        self.chunked_req: Optional[SimReq] = None

        # batch_is_full flag: prevents unnecessary scheduling attempts
        # Mirrors scheduler.py running_batch.batch_is_full
        self.batch_is_full: bool = False

        # Adaptive token ratio (for schedule_conservativeness)
        self.init_new_token_ratio = new_token_ratio
        self.min_new_token_ratio = new_token_ratio * 0.6
        self.new_token_ratio_decay = (
            (new_token_ratio - self.min_new_token_ratio) / 50
            if new_token_ratio > 0
            else 0.0
        )

    def get_next_batch(self) -> Optional[SimBatch]:
        """Get next prefill batch, with chunked prefill support.

        Mirrors scheduler.py::get_new_batch_prefill

        Returns:
            Prefill batch or None if no requests can be scheduled
        """
        # Check if there's anything to schedule
        has_waiting = bool(self.instance.waiting_queue)
        has_chunked = self.chunked_req is not None

        # Handle cases where prefill is not allowed
        if (self.batch_is_full or not has_waiting) and not has_chunked:
            return None

        # Check request count limits (mirrors get_num_allocatable_reqs)
        running_bs = len(self.instance.running_batch.reqs)
        num_allocatable = self.max_running_requests - running_bs
        if num_allocatable <= 0 and not has_chunked:
            self.batch_is_full = True
            return None

        # Calculate priorities using scheduling policy
        if has_waiting:
            self.policy.calc_priority(self.instance.waiting_queue)

        # Determine mixed-chunk decode tokens
        mixed_with_decode_tokens = 0
        if self.enable_mixed_chunk and not self.instance.running_batch.is_empty():
            mixed_with_decode_tokens = self.instance.running_batch.batch_size()

        # Create PrefillAdder for admission control
        adder = SimPrefillAdder(
            tree_cache=self.instance.tree_cache,
            token_to_kv_pool=self.instance.token_to_kv_pool,
            running_batch=self.instance.running_batch,
            max_prefill_tokens=self.max_prefill_tokens,
            new_token_ratio=self.new_token_ratio,
            chunked_prefill_size=self.chunked_prefill_size,
            mixed_with_decode_tokens=mixed_with_decode_tokens,
        )

        # Step 1: Handle continuation of a chunked request from previous round
        if self.chunked_req is not None:
            self._init_next_round_input(self.chunked_req)
            self.chunked_req = adder.add_chunked_req(self.chunked_req)

        # Step 2: Try to add new requests from waiting queue
        for req in self.instance.waiting_queue:
            # Check request count limit
            if len(adder.can_run_list) >= num_allocatable:
                self.batch_is_full = True
                break

            if self.batch_is_full:
                break

            # Initialize next round input (compute extend_input_len)
            self._init_next_round_input(req)

            if req.extend_input_len == 0:
                # Fully cached, skip
                continue

            # Try to add request
            result = adder.add_one_req(req)

            if result == AddReqResult.NO_TOKEN:
                # Out of total memory budget, stop trying
                self.batch_is_full = True
                break
            elif result == AddReqResult.OTHER:
                # Out of per-batch budget, stop trying
                break

        # Update chunked_req from adder (may have been set during add_one_req)
        if adder.new_chunked_req is not None:
            assert self.chunked_req is None
            self.chunked_req = adder.new_chunked_req

        # Increment is_chunked counter in scheduler (mirrors scheduler.py:1803)
        if self.chunked_req:
            self.chunked_req.is_chunked += 1

        can_run_list = adder.get_can_run_list()
        if not can_run_list:
            return None

        # Remove scheduled requests from waiting queue
        can_run_set = set(id(r) for r in can_run_list)
        self.instance.waiting_queue = [
            x for x in self.instance.waiting_queue if id(x) not in can_run_set
        ]

        # Allocate memory and update state for admitted requests
        final_list = []
        for req in can_run_list:
            if not self.instance.allocate_memory(req):
                # Allocation failed
                continue

            req.stage = RequestStage.PREFILL_FORWARD
            final_list.append(req)

        if not final_list:
            return None

        # Create batch
        batch = SimBatch(
            reqs=final_list,
            forward_mode=ForwardMode.PREFILL,
        )
        batch.update_metrics()

        return batch

    def decay_token_ratio(self):
        """Decay new_token_ratio when decode succeeds without memory pressure.

        Called by the engine after successful decode batches.
        """
        if self.new_token_ratio > self.min_new_token_ratio:
            self.new_token_ratio = max(
                self.new_token_ratio - self.new_token_ratio_decay,
                self.min_new_token_ratio,
            )

    def reset_token_ratio(self):
        """Reset new_token_ratio to initial value after retraction.

        Called when decode retraction happens due to memory pressure.
        """
        self.new_token_ratio = self.init_new_token_ratio

    def _init_next_round_input(self, req: SimReq):
        """Initialize next round input for request.

        Mirrors Req::init_next_round_input - reconstructs fill_ids,
        performs prefix matching, and computes extend_input_len.

        For chunked requests with prefill_tokens_done > 0, the already-processed
        tokens are accounted for directly (since they may not be in radix cache).

        Args:
            req: Request to initialize
        """
        # Reconstruct full token sequence
        req.fill_ids = req.origin_input_ids + req.output_ids
        full_token_ids = req.fill_ids

        # Match prefix in radix cache
        prefix_indices, matched_len, last_node = self.instance.tree_cache.match_prefix(
            full_token_ids
        )

        req.prefix_indices = prefix_indices

        # Account for both cache hits and already-processed chunks
        effective_prefix = max(matched_len, req.prefill_tokens_done)

        # Compute extend length (tokens to actually process)
        req.extend_input_len = max(0, len(full_token_ids) - effective_prefix)

        # Respect partial offload: LP solver already reduced extend_input_len
        if req.partial_offload_amount > 0:
            req.extend_input_len = max(1, req.extend_input_len - req.partial_offload_amount)
