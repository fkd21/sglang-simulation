"""Prefill instance for handling prefill requests."""

from __future__ import annotations

from typing import List

from simulation.instances.base_instance import InstanceType, SimInstance
from simulation.request.request import SimReq


class PrefillInstance(SimInstance):
    """Prefill instance for processing prefill batches.

    Additional queues for P/D disaggregation:
    - bootstrap_queue: Requests bootstrapping KV transfer
    - inflight_queue: Requests with KV transfer in progress
    """

    def __init__(self, instance_id: str):
        """Initialize prefill instance.

        Args:
            instance_id: Instance identifier
        """
        super().__init__(instance_id=instance_id, instance_type=InstanceType.PREFILL)
