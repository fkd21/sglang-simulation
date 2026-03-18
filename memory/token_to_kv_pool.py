"""KV cache token pool allocator.

Simplified from sglang.srt.mem_cache.allocator.BaseTokenToKVPoolAllocator
"""

from __future__ import annotations

from typing import List, Optional

from simulation.utils.constants import TOTAL_KV_CACHE_TOKENS


class SimTokenToKVPool:
    """Manages KV cache token allocation.

    Simplified allocator that tracks free KV cache tokens and allocates
    contiguous blocks for requests.
    """

    def __init__(self, total_kv_tokens: int = TOTAL_KV_CACHE_TOKENS):
        """Initialize KV pool.

        Args:
            total_kv_tokens: Total number of KV cache tokens available
        """
        self.total_kv_tokens = total_kv_tokens
        self.free_kv_tokens = total_kv_tokens
        self.allocated: dict[int, List[int]] = {}  # req_pool_idx -> indices
        self._next_idx = 0

    def alloc(self, num_tokens: int) -> Optional[List[int]]:
        """Allocate KV cache tokens.

        Args:
            num_tokens: Number of tokens to allocate

        Returns:
            List of allocated indices, or None if insufficient memory
        """
        if num_tokens > self.free_kv_tokens:
            return None

        # Allocate contiguous block
        start_idx = self._next_idx
        indices = list(range(start_idx, start_idx + num_tokens))
        self._next_idx += num_tokens
        self.free_kv_tokens -= num_tokens

        return indices

    def free(self, indices: List[int]):
        """Free KV cache tokens.

        Args:
            indices: Token indices to free
        """
        self.free_kv_tokens += len(indices)

    def available_size(self) -> int:
        """Get number of free KV tokens.

        Returns:
            Number of free tokens
        """
        return self.free_kv_tokens

    def can_allocate(self, num_tokens: int) -> bool:
        """Check if allocation is possible.

        Args:
            num_tokens: Number of tokens needed

        Returns:
            True if allocation is possible
        """
        return num_tokens <= self.free_kv_tokens

    def reset(self):
        """Reset allocator to initial state."""
        self.free_kv_tokens = self.total_kv_tokens
        self.allocated.clear()
        self._next_idx = 0
