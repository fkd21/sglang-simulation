"""Decode instance for handling decode requests."""

from __future__ import annotations

from typing import List

from instances.base_instance import InstanceType, SimInstance
from request.request import SimReq


class DecodeInstance(SimInstance):
    """Decode instance for processing decode batches.

    Additional queues for P/D disaggregation:
    - prealloc_queue: Requests pre-allocating memory
    - transfer_queue: Requests receiving KV transfer
    """

    def __init__(self, instance_id: str):
        """Initialize decode instance.

        Args:
            instance_id: Instance identifier
        """
        super().__init__(instance_id=instance_id, instance_type=InstanceType.DECODE)

        # Phase 4v2: Running sum of tokens in prealloc_reserved for O(1) capacity calculation
        # Eliminates O(n) loop over prealloc_reserved list (n ~150 items, called 11M times)
        self._prealloc_reserved_tokens = 0
