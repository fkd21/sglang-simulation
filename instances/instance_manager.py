"""Instance manager for routing requests to instances."""

from __future__ import annotations

from typing import List, Optional

from instances.base_instance import InstanceType, SimInstance
from instances.decode_instance import DecodeInstance
from instances.prefill_instance import PrefillInstance
from utils.constants import TOTAL_KV_CACHE_TOKENS


class InstanceManager:
    """Manages prefill and decode instances."""

    def __init__(self, num_prefill: int, num_decode: int, total_kv_tokens: int = TOTAL_KV_CACHE_TOKENS):
        """Initialize instance manager.

        Args:
            num_prefill: Number of prefill instances
            num_decode: Number of decode instances
            total_kv_tokens: KV cache capacity per instance in tokens
        """
        self.prefill_instances: List[PrefillInstance] = [
            PrefillInstance(instance_id=f"prefill_{i}", total_kv_tokens=total_kv_tokens)
            for i in range(num_prefill)
        ]
        self.decode_instances: List[DecodeInstance] = [
            DecodeInstance(instance_id=f"decode_{i}", total_kv_tokens=total_kv_tokens)
            for i in range(num_decode)
        ]

        self._next_prefill_idx = 0
        self._next_decode_idx = 0

    def select_prefill_instance(self) -> PrefillInstance:
        """Select a prefill instance for request assignment.

        Uses round-robin selection.

        Returns:
            Selected prefill instance
        """
        # Clamp index after role switches may have shrunk the list
        self._next_prefill_idx %= len(self.prefill_instances)
        instance = self.prefill_instances[self._next_prefill_idx]
        self._next_prefill_idx = (self._next_prefill_idx + 1) % len(
            self.prefill_instances
        )
        return instance

    def select_decode_instance(self) -> DecodeInstance:
        """Select a decode instance for request assignment.

        Uses round-robin selection.

        Returns:
            Selected decode instance
        """
        # Clamp index after role switches may have shrunk the list
        self._next_decode_idx %= len(self.decode_instances)
        instance = self.decode_instances[self._next_decode_idx]
        self._next_decode_idx = (self._next_decode_idx + 1) % len(
            self.decode_instances
        )
        return instance

    def select_least_loaded_prefill(self) -> Optional[PrefillInstance]:
        """Select prefill instance with smallest queue.

        Returns:
            Prefill instance with minimum queue length
        """
        if not self.prefill_instances:
            return None

        return min(self.prefill_instances, key=lambda i: i.get_queue_length())

    def select_least_loaded_decode(self) -> Optional[DecodeInstance]:
        """Select decode instance with smallest queue.

        Returns:
            Decode instance with minimum queue length
        """
        if not self.decode_instances:
            return None

        return min(self.decode_instances, key=lambda i: i.get_queue_length())

    def select_instance_for_migrated(
        self,
        target_type: InstanceType,
        exclude_instance_id: str
    ) -> Optional[SimInstance]:
        """Select least-loaded instance for migrated request.

        Excludes:
        - The draining instance
        - Instances not accepting requests

        Args:
            target_type: Type of instance to select (PREFILL or DECODE)
            exclude_instance_id: Instance ID to exclude (the draining one)

        Returns:
            Least-loaded instance, or None if no available instance
        """
        if target_type == InstanceType.PREFILL:
            candidates = [
                i for i in self.prefill_instances
                if i.instance_id != exclude_instance_id and i.accepting_requests
            ]
        else:
            candidates = [
                i for i in self.decode_instances
                if i.instance_id != exclude_instance_id and i.accepting_requests
            ]

        if not candidates:
            return None

        # Select least loaded (shortest queue)
        return min(candidates, key=lambda i: i.get_queue_length())

    def get_system_state(self) -> dict:
        """Get system-wide state for policy decisions.

        Returns:
            Dictionary with system metrics
        """
        total_prefill_queue = sum(i.get_queue_length() for i in self.prefill_instances)
        total_decode_queue = sum(i.get_queue_length() for i in self.decode_instances)

        return {
            'prefill_queue_length': total_prefill_queue,
            'decode_queue_length': total_decode_queue,
            'num_prefill_instances': len(self.prefill_instances),
            'num_decode_instances': len(self.decode_instances),
            'prefill_instances': self.prefill_instances,
            'decode_instances': self.decode_instances,
        }
