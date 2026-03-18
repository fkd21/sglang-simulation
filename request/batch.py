"""Batch data structure for simulation.

Mirrors sglang.srt.managers.schedule_batch.ScheduleBatch
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List

from request.request import SimReq
from utils.profiling_formulas import compute_inference_time


class ForwardMode(Enum):
    """Forward pass mode."""

    PREFILL = auto()
    DECODE = auto()
    MIXED = auto()
    EXTEND = auto()
    IDLE = auto()


@dataclass
class SimBatch:
    """Simulation batch mirroring SGLang's ScheduleBatch.

    Attributes:
        reqs: List of requests in the batch
        forward_mode: Batch processing mode (PREFILL, DECODE, MIXED, EXTEND)

        # Metrics for profiling formula
        total_prefill_tokens: Total number of prefill tokens
        decode_batch_size: Number of decode requests
        decode_computed_token_sum: Total computed decode-context tokens

        # Timing
        start_time: When batch execution started
        estimated_duration: Estimated execution time from profiling formula
    """

    reqs: List[SimReq] = field(default_factory=list)
    forward_mode: ForwardMode = ForwardMode.IDLE

    # Metrics for profiling formula
    total_prefill_tokens: int = 0
    decode_batch_size: int = 0
    decode_computed_token_sum: int = 0

    # Timing
    start_time: float = 0.0
    estimated_duration: float = 0.0

    def compute_inference_time(self) -> float:
        """Compute inference time using profiling formula.

        Uses regression model:
        t_inference = 0.014654
                + 6.944e-5 * total_prefill_tokens
                + 6.024e-5 * decode_batch_size
                + 8.463e-8 * decode_computed_token_sum

        Returns:
            Estimated inference time in seconds
        """
        return compute_inference_time(
            self.total_prefill_tokens,
            self.decode_batch_size,
            self.decode_computed_token_sum
        )

    def update_metrics(self):
        """Update batch metrics from requests."""
        if self.forward_mode == ForwardMode.PREFILL or self.forward_mode == ForwardMode.EXTEND:
            self.total_prefill_tokens = sum(req.extend_input_len for req in self.reqs)
            self.decode_batch_size = 0
            self.decode_computed_token_sum = 0
        elif self.forward_mode == ForwardMode.DECODE:
            self.total_prefill_tokens = 0
            self.decode_batch_size = len(self.reqs)
            self.decode_computed_token_sum = sum(
                req.context_tokens + len(req.output_ids) for req in self.reqs
            )
        elif self.forward_mode == ForwardMode.MIXED:
            # Mixed batch has both prefill and decode
            prefill_reqs = [r for r in self.reqs if not r.is_decode_started()]
            decode_reqs = [r for r in self.reqs if r.is_decode_started()]

            self.total_prefill_tokens = sum(r.extend_input_len for r in prefill_reqs)
            self.decode_batch_size = len(decode_reqs)
            self.decode_computed_token_sum = sum(
                r.context_tokens + len(r.output_ids) for r in decode_reqs
            )

        self.estimated_duration = self.compute_inference_time() * random.uniform(0.95, 1.05)

    def is_empty(self) -> bool:
        """Check if batch is empty."""
        return len(self.reqs) == 0

    def batch_size(self) -> int:
        """Get batch size."""
        return len(self.reqs)

    def __repr__(self) -> str:
        return (
            f"SimBatch(mode={self.forward_mode.name}, size={len(self.reqs)}, "
            f"prefill_tokens={self.total_prefill_tokens}, "
            f"decode_bs={self.decode_batch_size}, "
            f"duration={self.estimated_duration:.6f}s)"
        )
