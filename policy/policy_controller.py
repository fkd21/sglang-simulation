"""PolicyController coordinates the optimization mechanisms.

Provides a unified interface for the SimulationEngine to query decisions
about partial offloading (dynamic LP), decode continuation, and role switching.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from config import SimConfig
from instances.base_instance import InstanceType
from mechanisms.decode_continuation import DecodeContinuation
from mechanisms.partial_offload import DynamicBetaSolver
from mechanisms.role_switching import RoleSwitcher
from request.request import SimReq


class PolicyController:
    """Coordinates all optimization mechanisms."""

    def __init__(self, config: SimConfig):
        """Initialize from simulation config.

        Args:
            config: Simulation configuration
        """
        self.dynamic_lp_enabled = config.enable_dynamic_lp
        self.lp_solver: Optional[DynamicBetaSolver] = None
        if config.enable_dynamic_lp:
            self.lp_solver = DynamicBetaSolver(
                slo_target=config.slo_target,
                max_window_size=config.lp_max_window_size,
            )

        self.continuation = DecodeContinuation(
            M=config.M if config.enable_continuation else 0
        )
        self.switcher = RoleSwitcher(
            policy=config.switch_policy if config.enable_switching else "never"
        )

    def solve_dynamic_betas(
        self,
        requests: Sequence[SimReq],
        prefill_ready_time: float,
        decode_bs: int,
        decode_token_sum: int,
    ) -> Tuple[List[float], bool, float]:
        """Solve for dynamic betas using LP solver.

        Args:
            requests: Requests from the waiting queue (up to max_window_size used).
            prefill_ready_time: When the prefill instance is available.
            decode_bs: Current decode batch size.
            decode_token_sum: Current decode computed token sum.

        Returns:
            (betas, feasible, solve_time) tuple
        """
        if self.lp_solver is None:
            return [0.0] * len(requests), True, 0.0
        return self.lp_solver.solve(
            requests, prefill_ready_time, decode_bs, decode_token_sum
        )

    def tokens_to_continue_on_prefill(self, req: SimReq) -> int:
        """Compute decode continuation length on prefill instance.

        Args:
            req: Request that just completed prefill

        Returns:
            Number of decode tokens to generate on prefill instance
        """
        return self.continuation.tokens_to_continue(req)

    def should_switch_role(
        self,
        instance_type: InstanceType,
        prefill_queue_len: int,
        decode_queue_len: int,
        current_time: float,
        instance_is_idle: bool,
    ) -> Optional[InstanceType]:
        """Check if an instance should switch roles.

        Returns:
            New role if should switch, None otherwise
        """
        return self.switcher.should_switch(
            instance_type=instance_type,
            prefill_queue_len=prefill_queue_len,
            decode_queue_len=decode_queue_len,
            current_time=current_time,
            instance_is_idle=instance_is_idle,
        )

    def record_switch(self, current_time: float):
        """Record that a switch occurred."""
        self.switcher.record_switch(current_time)

    @property
    def switch_cost(self) -> float:
        """Time cost for role switching."""
        return self.switcher.switch_cost

    @property
    def offloading_enabled(self) -> bool:
        return self.dynamic_lp_enabled

    @property
    def continuation_enabled(self) -> bool:
        return self.continuation.enabled

    @property
    def switching_enabled(self) -> bool:
        return self.switcher.enabled

    def __repr__(self) -> str:
        return (
            f"PolicyController("
            f"dynamic_lp={self.dynamic_lp_enabled}, "
            f"M={self.continuation.M}, "
            f"switch_policy={self.switcher.policy})"
        )
