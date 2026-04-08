"""Decode instance for handling decode requests."""

from __future__ import annotations

from typing import List

from instances.base_instance import InstanceType, SimInstance
from request.request import SimReq
from utils.constants import TOTAL_KV_CACHE_TOKENS


class DecodeInstance(SimInstance):
    """Decode instance for processing decode batches.

    Additional queues for P/D disaggregation:
    - prealloc_queue: Requests pre-allocating memory
    - transfer_queue: Requests receiving KV transfer
    """

    def __init__(self, instance_id: str, total_kv_tokens: int = TOTAL_KV_CACHE_TOKENS):
        """Initialize decode instance.

        Args:
            instance_id: Instance identifier
            total_kv_tokens: KV cache capacity in tokens (default: A100 40GB profile)
        """
        super().__init__(instance_id=instance_id, instance_type=InstanceType.DECODE, max_kv_tokens=total_kv_tokens)
        # Note: _prealloc_reserved_tokens counter is inherited from SimInstance base class
