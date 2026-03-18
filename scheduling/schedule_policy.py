"""Scheduling policies for request prioritization.

Mirrors sglang.srt.managers.schedule_policy.SchedulePolicy
"""

from __future__ import annotations

from enum import Enum
from typing import List

from memory.radix_cache import SimRadixCache
from request.request import SimReq


class PolicyType(Enum):
    """Scheduling policy types."""

    FCFS = "fcfs"  # First Come First Serve
    LPM = "lpm"  # Longest Prefix Match
    LRU = "lru"  # Least Recently Used
    RANDOM = "random"


class SimSchedulePolicy:
    """Scheduling policy for prioritizing requests."""

    def __init__(
        self,
        policy: str = "fcfs",
        tree_cache: SimRadixCache | None = None
    ):
        """Initialize scheduling policy.

        Args:
            policy: Policy type ("fcfs", "lpm", etc.)
            tree_cache: Radix cache for LPM policy
        """
        try:
            self.policy = PolicyType(policy.lower())
        except ValueError:
            self.policy = PolicyType.FCFS

        self.tree_cache = tree_cache

    def calc_priority(self, waiting_queue: List[SimReq]):
        """Calculate and assign priorities to requests.

        Modifies requests in-place by sorting the waiting queue.

        Args:
            waiting_queue: List of waiting requests to prioritize
        """
        if self.policy == PolicyType.FCFS:
            # Already in FIFO order, no change needed
            pass
        elif self.policy == PolicyType.LPM:
            # Sort by longest prefix match
            self._sort_by_prefix_match(waiting_queue)
        elif self.policy == PolicyType.RANDOM:
            # Random shuffle
            import random
            random.shuffle(waiting_queue)

    def _sort_by_prefix_match(self, waiting_queue: List[SimReq]):
        """Sort requests by longest prefix match.

        Args:
            waiting_queue: List of waiting requests
        """
        if not self.tree_cache:
            return

        # Calculate prefix match length for each request
        match_lengths = []
        for req in waiting_queue:
            prefix_indices, matched_len, _ = self.tree_cache.match_prefix(
                req.origin_input_ids + req.output_ids
            )
            match_lengths.append(matched_len)

        # Sort by descending match length (longest prefix first)
        sorted_pairs = sorted(
            zip(waiting_queue, match_lengths),
            key=lambda x: x[1],
            reverse=True
        )

        # Update waiting queue in-place
        waiting_queue.clear()
        waiting_queue.extend([req for req, _ in sorted_pairs])
