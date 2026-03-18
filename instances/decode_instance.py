"""Decode instance for handling decode requests."""

from __future__ import annotations

from typing import List

from simulation.instances.base_instance import InstanceType, SimInstance
from simulation.request.request import SimReq


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
