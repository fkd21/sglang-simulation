"""PrefillAdder for admission control with chunked prefill support.

Mirrors sglang.srt.managers.schedule_policy.PrefillAdder
"""

from __future__ import annotations

from enum import Enum, auto
from typing import List, Optional

from simulation.memory.radix_cache import SimRadixCache
from simulation.memory.token_to_kv_pool import SimTokenToKVPool
from simulation.request.batch import SimBatch
from simulation.request.request import SimReq

# Cap the estimated max_new_tokens to prevent over-conservative budgeting.
# Mirrors CLIP_MAX_NEW_TOKENS in sglang.srt.managers.schedule_policy.
CLIP_MAX_NEW_TOKENS = 4096

# Default page size (tokens per page). Page size 1 means no alignment overhead.
DEFAULT_PAGE_SIZE = 1


class AddReqResult(Enum):
    """Result of adding a request to batch."""

    CONTINUE = auto()  # Successfully added, can continue
    NO_TOKEN = auto()  # Out of total memory budget
    OTHER = auto()  # Out of per-batch budget (input tokens or chunk tokens)


class SimPrefillAdder:
    """Admission control for prefill requests with chunked prefill support.

    Mirrors sglang.srt.managers.schedule_policy.PrefillAdder with:
    - rem_total_tokens: Conservative memory budget accounting for running decode
    - rem_input_tokens: Per-batch prefill input token budget
    - rem_chunk_tokens: Per-chunk token limit (None if chunking disabled)
    - Page-aligned token allocation
    - mixed_with_decode_tokens budget deduction
    """

    def __init__(
        self,
        tree_cache: SimRadixCache,
        token_to_kv_pool: SimTokenToKVPool,
        running_batch: SimBatch,
        max_prefill_tokens: int,
        new_token_ratio: float = 0.0,
        chunked_prefill_size: Optional[int] = None,
        mixed_with_decode_tokens: int = 0,
        page_size: int = DEFAULT_PAGE_SIZE,
    ):
        self.tree_cache = tree_cache
        self.token_to_kv_pool = token_to_kv_pool
        self.running_batch = running_batch
        self.new_token_ratio = new_token_ratio
        self.page_size = page_size

        # Track added requests
        self.can_run_list: List[SimReq] = []

        # Chunked request state: set when a request is partially scheduled
        self.new_chunked_req: Optional[SimReq] = None

        # Token budgets (mirrors real PrefillAdder)
        self.rem_input_tokens = max_prefill_tokens - mixed_with_decode_tokens
        self.rem_chunk_tokens = chunked_prefill_size  # None = no chunking
        if self.rem_chunk_tokens is not None:
            self.rem_chunk_tokens -= mixed_with_decode_tokens

        # Conservative memory budgeting offsets
        # rem_total_token_offset tracks tokens that are "spoken for" by:
        # 1. Running decode requests' estimated future tokens
        # 2. Newly added prefill requests' extend + estimated decode tokens
        self.rem_total_token_offset = mixed_with_decode_tokens
        self.cur_rem_token_offset = mixed_with_decode_tokens

        # Account for running batch's estimated future decode tokens
        if running_batch is not None and not running_batch.is_empty():
            for req in running_batch.reqs:
                self.rem_total_token_offset += self._get_running_request_total_token_offset(req)

    def _get_running_request_total_token_offset(self, req: SimReq) -> int:
        """Estimate tokens a running decode request still needs.

        Mirrors PrefillAdder._get_running_request_total_token_offset.
        """
        remaining = max(req.generated_tokens - len(req.output_ids), 0)
        return min(remaining, CLIP_MAX_NEW_TOKENS) * self.new_token_ratio

    @property
    def rem_total_tokens(self) -> float:
        """Available memory budget accounting for running decode tokens.

        Mirrors PrefillAdder.rem_total_tokens property.
        """
        available_and_evictable = (
            self.token_to_kv_pool.available_size()
            + self.tree_cache.evictable_size
        )
        return available_and_evictable - self.rem_total_token_offset

    @property
    def cur_rem_tokens(self) -> float:
        """Current remaining tokens for this batch.

        Mirrors PrefillAdder.cur_rem_tokens property.
        """
        available_and_evictable = (
            self.token_to_kv_pool.available_size()
            + self.tree_cache.evictable_size
        )
        return available_and_evictable - self.cur_rem_token_offset

    def ceil_paged_tokens(self, tokens: int) -> int:
        """Round up tokens to page boundary.

        Mirrors PrefillAdder.ceil_paged_tokens.
        """
        return -(-tokens // self.page_size) * self.page_size

    def budget_state(self) -> AddReqResult:
        """Check current budget state.

        Mirrors PrefillAdder.budget_state.
        """
        if self.rem_total_tokens <= 0 or self.cur_rem_tokens <= 0:
            return AddReqResult.NO_TOKEN

        if self.rem_input_tokens <= 0 or (
            self.rem_chunk_tokens is not None and self.rem_chunk_tokens <= 0
        ):
            return AddReqResult.OTHER

        return AddReqResult.CONTINUE

    def _update_prefill_budget(
        self, prefix_len: int, extend_input_len: int, max_new_tokens: int
    ):
        """Update budget after adding a request.

        Mirrors PrefillAdder._update_prefill_budget.
        """
        extend_input_len_paged = self.ceil_paged_tokens(extend_input_len)

        self.rem_total_token_offset += extend_input_len_paged + max_new_tokens
        self.cur_rem_token_offset += extend_input_len_paged
        self.rem_input_tokens -= extend_input_len_paged
        if self.rem_chunk_tokens is not None:
            self.rem_chunk_tokens -= extend_input_len_paged

    def add_one_req(self, req: SimReq) -> AddReqResult:
        """Try to add one request to the batch.

        Mirrors PrefillAdder.add_one_req. Checks total memory budget,
        input token budget, and chunk budget before adding.
        """
        input_tokens = self.ceil_paged_tokens(req.extend_input_len)
        prefix_len = len(req.prefix_indices)
        max_new_tokens = min(
            max(req.generated_tokens - len(req.output_ids), 0),
            CLIP_MAX_NEW_TOKENS,
        )
        total_tokens = input_tokens + max_new_tokens

        # Check total memory budget (conservative estimate)
        if total_tokens >= self.rem_total_tokens:
            return AddReqResult.NO_TOKEN

        # Check input token budget (allow first request even if over budget)
        if input_tokens >= self.rem_input_tokens and len(self.can_run_list) != 0:
            return AddReqResult.OTHER

        # Check chunk token budget and decide chunking
        if self.rem_chunk_tokens is None or input_tokens <= self.rem_chunk_tokens:
            # Non-chunked path: request fits entirely
            # Verify KV cache has space (with eviction)
            if not self._try_allocate_kv_cache(req.extend_input_len):
                return AddReqResult.NO_TOKEN

            self.can_run_list.append(req)
            self._update_prefill_budget(
                prefix_len,
                req.extend_input_len,
                min(req.generated_tokens, CLIP_MAX_NEW_TOKENS),
            )
        else:
            # Request exceeds chunk size - CHUNK IT
            trunc_len = self.rem_chunk_tokens
            if trunc_len <= 0:
                return AddReqResult.OTHER

            # Page-align the truncation
            trunc_len = (trunc_len // self.page_size) * self.page_size
            if trunc_len <= 0:
                return AddReqResult.OTHER

            # Verify KV cache has space for the chunk
            if not self._try_allocate_kv_cache(trunc_len):
                return AddReqResult.NO_TOKEN

            # Modify request for this chunk only
            req.extend_input_len = trunc_len
            req.fill_ids = req.fill_ids[:prefix_len + trunc_len]

            # Mark as chunked for continuation next round
            self.new_chunked_req = req
            self.can_run_list.append(req)
            self._update_prefill_budget(prefix_len, trunc_len, 0)

        return self.budget_state()

    def add_chunked_req(self, req: SimReq) -> Optional[SimReq]:
        """Continue processing a chunked request from the previous round.

        Mirrors PrefillAdder.add_chunked_req.

        Returns:
            The request if still chunked (needs more rounds), None if complete
        """
        prefix_len = len(req.prefix_indices)

        # How many tokens can we process this round?
        _rem_tokens = min(self.rem_chunk_tokens, int(self.rem_total_tokens))
        truncated = req.extend_input_len > _rem_tokens
        req.extend_input_len = min(req.extend_input_len, _rem_tokens)
        req.fill_ids = req.fill_ids[:prefix_len + req.extend_input_len]

        self.can_run_list.append(req)
        self._update_prefill_budget(
            0,
            req.extend_input_len,
            (
                min(req.generated_tokens, CLIP_MAX_NEW_TOKENS)
                if not truncated
                else 0
            ),
        )

        # Return if chunked prefill not finished
        return req if truncated else None

    def _try_allocate_kv_cache(self, num_tokens: int) -> bool:
        """Check if KV cache has space, with eviction if needed."""
        if self.token_to_kv_pool.can_allocate(num_tokens):
            return True

        # Try evicting from cache to free memory
        self.tree_cache.evict(
            num_tokens,
            evict_callback=lambda kv_indices: self.token_to_kv_pool.free(kv_indices)
        )

        return self.token_to_kv_pool.can_allocate(num_tokens)

    def get_can_run_list(self) -> List[SimReq]:
        """Get list of requests that can run."""
        return self.can_run_list
