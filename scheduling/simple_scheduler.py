"""Decode scheduler with per-token KV allocation and retraction support.

Mirrors sglang.srt.managers.scheduler.Scheduler decode scheduling
including update_running_batch, check_decode_mem, and retract_decode.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from instances.base_instance import SimInstance
from request.batch import ForwardMode, SimBatch
from request.request import RequestStage, SimReq

logger = logging.getLogger(__name__)


class SimpleScheduler:
    """Decode scheduler with per-token KV allocation and retraction."""

    def __init__(self, instance: SimInstance, max_prefill_tokens: int = 16384):
        """Initialize scheduler.

        Args:
            instance: Instance to schedule for
            max_prefill_tokens: Maximum prefill tokens per batch
        """
        self.instance = instance
        self.max_prefill_tokens = max_prefill_tokens
        self.last_retracted_reqs: List[SimReq] = []

    def get_next_prefill_batch(self) -> Optional[SimBatch]:
        """Get next prefill batch from waiting queue.

        Returns:
            Prefill batch or None if no requests can be scheduled
        """
        if not self.instance.waiting_queue:
            return None

        # Simple FCFS: take requests until token limit
        batch_reqs = []
        total_tokens = 0

        for req in self.instance.waiting_queue[:]:
            # Check if we can fit this request
            if total_tokens + req.extend_input_len > self.max_prefill_tokens:
                if batch_reqs:
                    break  # Batch is full
                # First request is too large, skip for now
                continue

            # Try to allocate memory
            if self.instance.allocate_memory(req):
                batch_reqs.append(req)
                total_tokens += req.extend_input_len
                self.instance.waiting_queue.remove(req)
                req.stage = RequestStage.PREFILL_FORWARD
            else:
                # Out of memory
                break

        if not batch_reqs:
            return None

        # Create batch
        batch = SimBatch(
            reqs=batch_reqs,
            forward_mode=ForwardMode.PREFILL
        )
        batch.update_metrics()

        return batch

    def get_next_decode_batch(self) -> Optional[SimBatch]:
        """Get next decode batch from running requests.

        Mirrors update_running_batch: checks decode memory, performs
        retraction if needed, allocates 1 KV token per request.

        Returns:
            Decode batch or None if no running requests
        """
        if self.instance.running_batch.is_empty():
            return None

        batch = self.instance.running_batch

        # Check decode memory: each request needs 1 new KV token
        num_tokens_needed = len(batch.reqs)
        self.last_retracted_reqs = self._check_and_handle_decode_mem(batch, num_tokens_needed)

        if batch.is_empty():
            return None

        # Allocate 1 KV token per request for the next decode iteration
        for req in batch.reqs:
            kv_indices = self.instance.token_to_kv_pool.alloc(1)
            if kv_indices is not None:
                req.kv_cache_indices.extend(kv_indices)
            # If allocation fails after retraction, we still proceed
            # (the real system would also proceed after retraction)

        # Detect offloaded requests that need first-pass prefill on decode
        extend_reqs = [r for r in batch.reqs if r.extend_input_len > 0]

        if extend_reqs:
            batch.forward_mode = ForwardMode.MIXED
        else:
            batch.forward_mode = ForwardMode.DECODE

        batch.update_metrics()

        # Reset extend_input_len after first MIXED iteration
        # (offloaded prefill only happens once per request)
        for r in extend_reqs:
            r.extend_input_len = 0

        return batch

    def _check_and_handle_decode_mem(
        self, batch: SimBatch, num_tokens_needed: int
    ) -> List[SimReq]:
        """Check decode memory and retract requests if needed.

        Mirrors ScheduleBatch.check_decode_mem and retract_decode.

        Returns:
            List of retracted requests (empty if no retraction needed)
        """
        # Check if we can allocate the needed tokens (with eviction)
        available = self.instance.token_to_kv_pool.available_size()
        evictable = self.instance.tree_cache.evictable_size

        if available + evictable >= num_tokens_needed:
            # Try eviction if needed
            if available < num_tokens_needed:
                self.instance.tree_cache.evict(
                    num_tokens_needed - available,
                    evict_callback=lambda kv_indices: self.instance.token_to_kv_pool.free(kv_indices)
                )
            return []

        # Not enough memory even after eviction - retract requests
        return self._retract_decode(batch)

    def _retract_decode(self, batch: SimBatch) -> List[SimReq]:
        """Retract decode requests when memory is insufficient.

        Mirrors ScheduleBatch.retract_decode: removes requests with the
        most output tokens first (they've made the most progress and
        occupy the most KV cache).

        Returns:
            List of retracted requests
        """
        retracted_reqs = []

        # Sort by output length descending (retract longest first)
        sorted_indices = sorted(
            range(len(batch.reqs)),
            key=lambda i: (len(batch.reqs[i].output_ids), -batch.reqs[i].context_tokens),
            reverse=True,
        )

        for idx in sorted_indices:
            if len(batch.reqs) <= 1:
                break  # Keep at least one request

            # Check if we have enough memory now
            if self.instance.token_to_kv_pool.available_size() >= len(batch.reqs):
                break

            req = batch.reqs[idx]
            # Free this request's KV cache
            self.instance.free_memory(req)
            retracted_reqs.append(req)

        # Remove retracted requests from batch
        retracted_set = set(id(r) for r in retracted_reqs)
        batch.reqs = [r for r in batch.reqs if id(r) not in retracted_set]

        if retracted_reqs:
            logger.info(
                f"Retracted {len(retracted_reqs)} decode requests due to memory pressure"
            )

        return retracted_reqs

    def add_to_running_batch(self, reqs: list):
        """Add requests to running batch.

        Args:
            reqs: Requests to add
        """
        self.instance.running_batch.reqs.extend(reqs)
