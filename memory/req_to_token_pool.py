"""Request-to-token pool for managing request slots.

Simplified from sglang.srt.mem_cache.common.ReqToTokenPool
"""

from __future__ import annotations

from typing import Dict, List, Optional

from utils.constants import DEFAULT_MAX_RUNNING_REQUESTS


class SimReqToTokenPool:
    """Manages request slots in the token pool.

    Tracks allocation of request pool indices and token mappings.
    """

    def __init__(
        self,
        size: int = DEFAULT_MAX_RUNNING_REQUESTS,
        max_context_len: int = 32768
    ):
        """Initialize request-to-token pool.

        Args:
            size: Maximum number of concurrent requests
            max_context_len: Maximum context length per request
        """
        self.size = size
        self.max_context_len = max_context_len
        self.req_to_token: Dict[int, List[int]] = {}
        self.free_slots: List[int] = list(range(size))

    def alloc(self, need_size: int = 1) -> Optional[int]:
        """Allocate a request pool index.

        Args:
            need_size: Size needed (always 1 for simplified version)

        Returns:
            Allocated req_pool_idx, or None if no slots available
        """
        if not self.free_slots:
            return None

        req_pool_idx = self.free_slots.pop(0)
        self.req_to_token[req_pool_idx] = []
        return req_pool_idx

    def free(self, req_pool_idx: int):
        """Free a request pool index.

        Args:
            req_pool_idx: Request pool index to free
        """
        if req_pool_idx in self.req_to_token:
            del self.req_to_token[req_pool_idx]
            self.free_slots.append(req_pool_idx)
            self.free_slots.sort()

    def available_size(self) -> int:
        """Get number of free slots.

        Returns:
            Number of free request slots
        """
        return len(self.free_slots)

    def set_tokens(self, req_pool_idx: int, token_indices: List[int]):
        """Set token indices for a request.

        Args:
            req_pool_idx: Request pool index
            token_indices: Token indices to store
        """
        if req_pool_idx in self.req_to_token:
            self.req_to_token[req_pool_idx] = token_indices

    def get_tokens(self, req_pool_idx: int) -> List[int]:
        """Get token indices for a request.

        Args:
            req_pool_idx: Request pool index

        Returns:
            Token indices for the request
        """
        return self.req_to_token.get(req_pool_idx, [])

    def reset(self):
        """Reset pool to initial state."""
        self.req_to_token.clear()
        self.free_slots = list(range(self.size))
