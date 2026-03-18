"""Request data structure for simulation.

Mirrors sglang.srt.managers.schedule_batch.Req
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional


class RequestStage(Enum):
    """Stages of request processing."""

    WAITING = auto()
    PREFILL_FORWARD = auto()
    PREFILL_TRANSFER_KV_CACHE = auto()
    DECODE_WAITING = auto()
    DECODE_FORWARD = auto()
    COMPLETED = auto()


@dataclass
class SimReq:
    """Simulation request mirroring SGLang's Req class.

    Attributes:
        # Input (from trace)
        rid: Request ID
        arrival_time: When request arrives in the system
        context_tokens: Number of prefill tokens (ContextTokens from trace)
        generated_tokens: Target number of tokens to generate (GeneratedTokens from trace)

        # Tracking
        origin_input_ids: Original input token IDs (simulated)
        output_ids: Generated output token IDs
        decode_tokens_generated: Number of tokens generated so far

        # Memory (mirrors SGLang)
        kv_cache_indices: KV cache allocation indices
        req_pool_idx: Index in request-to-token pool
        prefix_indices: Cached prefix indices from radix cache
        extend_input_len: Actual tokens to compute (total - prefix)

        # Latency tracking
        queue_entry_time: When request entered waiting queue
        prefill_start_time: When prefill started
        prefill_end_time: When prefill completed
        decode_start_time: When decode started
        decode_end_time: When decode completed

        # State
        stage: Current processing stage
        finished: Whether request has completed

        # New mechanisms
        partial_offload_amount: Number of tokens offloaded (β * context_tokens)
        continued_on_prefill: Number of tokens decoded on prefill instance
        lp_beta: LP-assigned beta value (for metrics/debugging)
    """

    # Input from trace
    rid: str
    arrival_time: float
    context_tokens: int
    generated_tokens: int

    # Tracking
    origin_input_ids: List[int] = field(default_factory=list)
    output_ids: List[int] = field(default_factory=list)
    fill_ids: List[int] = field(default_factory=list)  # Truncated input for current chunk
    decode_tokens_generated: int = 0
    is_chunked: int = 0  # Counter: how many times this request has been chunked
    prefill_tokens_done: int = 0  # Tokens already prefilled (for chunked continuation)

    # Memory
    kv_cache_indices: List[int] = field(default_factory=list)
    req_pool_idx: Optional[int] = None
    prefix_indices: List[int] = field(default_factory=list)
    extend_input_len: int = 0

    # Latency tracking
    queue_entry_time: float = 0.0
    prefill_start_time: float = 0.0
    prefill_end_time: float = 0.0
    decode_start_time: float = 0.0
    decode_end_time: float = 0.0

    # Per-request lifecycle tracking
    bootstrap_exit_time: float = 0.0
    continuation_start_time: float = 0.0
    continuation_end_time: float = 0.0
    kv_transfer_start_time: float = 0.0
    kv_transfer_end_time: float = 0.0
    offload_start_time: float = 0.0
    offload_end_time: float = 0.0
    decode_queue_entry_time: float = 0.0
    completion_time: float = 0.0

    # State
    stage: RequestStage = RequestStage.WAITING
    finished: bool = False

    # New mechanisms
    partial_offload_amount: int = 0
    continued_on_prefill: int = 0
    lp_beta: float = 0.0  # LP-assigned beta value (for metrics/debugging)

    # Instance assignment
    assigned_prefill_instance: Optional[int] = None
    assigned_decode_instance: Optional[int] = None

    # Role switching migration tracking
    migrated_from_switch: bool = False
    migration_timestamp: float = 0.0

    def __post_init__(self):
        """Initialize derived fields."""
        if not self.origin_input_ids:
            # Simulate token IDs (unique per request using rid hash)
            # Use a large offset based on request ID to ensure uniqueness
            offset = hash(self.rid) % 1000000000
            self.origin_input_ids = list(range(offset, offset + self.context_tokens))
        self.extend_input_len = self.context_tokens

    def tokens_remaining(self) -> int:
        """Get number of tokens left to generate."""
        return self.generated_tokens - self.decode_tokens_generated

    def total_tokens(self) -> int:
        """Get total tokens (context + generated so far)."""
        return self.context_tokens + self.decode_tokens_generated

    def is_prefill_complete(self) -> bool:
        """Check if prefill is complete."""
        return self.stage.value >= RequestStage.PREFILL_TRANSFER_KV_CACHE.value

    def is_decode_started(self) -> bool:
        """Check if decode has started."""
        return self.stage.value >= RequestStage.DECODE_FORWARD.value

    def __repr__(self) -> str:
        return (
            f"SimReq(rid={self.rid}, ctx={self.context_tokens}, "
            f"gen={self.decode_tokens_generated}/{self.generated_tokens}, "
            f"stage={self.stage.name})"
        )
