"""Base instance class for prefill and decode instances."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Deque, List, Optional

from memory.radix_cache import SimRadixCache
from memory.req_to_token_pool import SimReqToTokenPool
from memory.token_to_kv_pool import SimTokenToKVPool
from request.batch import ForwardMode, SimBatch
from request.request import SimReq
from utils.constants import (
    AVAILABLE_KV_MEMORY_BYTES,
    MODEL_SIZE_BYTES,
    TOTAL_KV_CACHE_TOKENS,
    TOTAL_VRAM_BYTES,
)


class InstanceType(Enum):
    """Type of instance."""

    PREFILL = auto()
    DECODE = auto()


@dataclass
class SimInstance:
    """Base class for simulation instances.

    Attributes:
        instance_id: Unique instance identifier
        instance_type: PREFILL or DECODE

        # Queues
        waiting_queue: Requests waiting to be scheduled
        running_batch: Currently running batch

        # Memory configuration
        total_memory: Total VRAM in bytes
        model_weights: Model weight size in bytes
        available_kv_memory: Available memory for KV cache
        max_kv_tokens: Maximum KV cache tokens

        # Memory pools
        req_to_token_pool: Request-to-token pool
        token_to_kv_pool: KV cache token pool
        tree_cache: Radix cache for prefix matching

        # State
        busy: Whether instance is currently processing
        busy_until: Time when instance will be free
        current_batch: Currently executing batch
    """

    instance_id: str
    instance_type: InstanceType

    # Queues
    waiting_queue: List[SimReq] = field(default_factory=list)
    running_batch: SimBatch = field(default_factory=SimBatch)

    # Prefill-specific queues (used when instance_type == PREFILL)
    bootstrap_queue: Deque[SimReq] = field(default_factory=deque)
    inflight_queue: List[SimReq] = field(default_factory=list)

    # Decode-specific queues (used when instance_type == DECODE)
    prealloc_queue: List[SimReq] = field(default_factory=list)
    transfer_queue: List[SimReq] = field(default_factory=list)
    prealloc_reserved: List[SimReq] = field(default_factory=list)

    # Memory configuration
    total_memory: int = TOTAL_VRAM_BYTES
    model_weights: int = MODEL_SIZE_BYTES
    available_kv_memory: int = AVAILABLE_KV_MEMORY_BYTES
    max_kv_tokens: int = TOTAL_KV_CACHE_TOKENS

    # Memory pools
    req_to_token_pool: SimReqToTokenPool = field(default_factory=SimReqToTokenPool)
    token_to_kv_pool: SimTokenToKVPool = field(default_factory=SimTokenToKVPool)
    tree_cache: SimRadixCache = field(default_factory=SimRadixCache)

    # State
    busy: bool = False
    busy_until: float = 0.0
    current_batch: Optional[SimBatch] = None

    # Role switching state
    draining: bool = False
    switch_target_role: Optional[InstanceType] = None
    switch_initiated_time: float = 0.0
    drain_duration: float = 0.0
    blocked_until: float = 0.0
    accepting_requests: bool = True

    def is_idle(self) -> bool:
        """Check if instance is idle.

        Returns:
            True if instance is not busy
        """
        return not self.busy

    def has_waiting_requests(self) -> bool:
        """Check if instance has waiting requests.

        Returns:
            True if waiting queue is not empty
        """
        return len(self.waiting_queue) > 0

    def has_running_requests(self) -> bool:
        """Check if instance has running requests.

        Returns:
            True if running batch is not empty
        """
        return not self.running_batch.is_empty()

    def get_queue_length(self) -> int:
        """Get total queue length (waiting + running).

        Returns:
            Total number of requests
        """
        return len(self.waiting_queue) + len(self.running_batch.reqs)

    def add_request(self, req: SimReq):
        """Add request to waiting queue.

        Args:
            req: Request to add
        """
        self.waiting_queue.append(req)

    def allocate_memory(self, req: SimReq) -> bool:
        """Allocate memory for a request.

        Args:
            req: Request to allocate memory for

        Returns:
            True if allocation succeeded
        """
        # Allocate req_pool slot
        req_pool_idx = self.req_to_token_pool.alloc()
        if req_pool_idx is None:
            return False

        req.req_pool_idx = req_pool_idx

        # Allocate KV cache
        kv_indices = self.token_to_kv_pool.alloc(req.extend_input_len)
        if kv_indices is None:
            # Free req pool slot
            self.req_to_token_pool.free(req_pool_idx)
            return False

        req.kv_cache_indices = kv_indices
        self.req_to_token_pool.set_tokens(req_pool_idx, kv_indices)

        return True

    def free_memory(self, req: SimReq):
        """Free memory for a request.

        Args:
            req: Request to free memory for
        """
        if req.req_pool_idx is not None:
            self.req_to_token_pool.free(req.req_pool_idx)
            req.req_pool_idx = None

        if req.kv_cache_indices:
            self.token_to_kv_pool.free(req.kv_cache_indices)
            req.kv_cache_indices = []

    def __repr__(self) -> str:
        return (
            f"SimInstance(id={self.instance_id}, "
            f"type={self.instance_type.name}, "
            f"waiting={len(self.waiting_queue)}, "
            f"running={self.running_batch.batch_size()}, "
            f"busy={self.busy})"
        )
