"""Role Switching mechanism.

Allows instances to dynamically switch between prefill and decode roles
based on system load conditions.
"""

from __future__ import annotations

from typing import Optional

from simulation.instances.base_instance import InstanceType


class RoleSwitcher:
    """Provides role switching execution mechanism.

    Switch decisions are made by PolicyMonitor, which creates ROLE_SWITCH events.
    This class now only provides switch cost and the execution mechanism.

    The manual switch API (_handle_role_switch, _initiate_role_switch in engine.py)
    is used by:
    1. Manual switch_schedule from config (for testing/debugging)
    2. PolicyMonitor (for policy-driven switching)

    Switching has a cost (flush KV cache, reconfigure scheduler) modeled
    as a time delay.
    """

    def __init__(
        self,
        policy: str = "never",
        prefill_queue_threshold: int = 50,
        decode_queue_threshold: int = 100,
        switch_cooldown: float = 5.0,
        switch_cost: float = 0.1,
    ):
        """Initialize role switcher.

        Args:
            policy: Switching policy ("never", "load_based", "adaptive")
            prefill_queue_threshold: Switch decode→prefill when prefill queue exceeds this
            decode_queue_threshold: Switch prefill→decode when decode queue exceeds this
            switch_cooldown: Minimum time between switches (seconds)
            switch_cost: Time cost per switch (seconds)
        """
        self.policy = policy
        self.prefill_queue_threshold = prefill_queue_threshold
        self.decode_queue_threshold = decode_queue_threshold
        self.switch_cooldown = switch_cooldown
        self.switch_cost = switch_cost

        self._last_switch_time: float = -float('inf')

    @property
    def enabled(self) -> bool:
        return self.policy != "never"

    def should_switch(
        self,
        instance_type: InstanceType,
        prefill_queue_len: int,
        decode_queue_len: int,
        current_time: float,
        instance_is_idle: bool,
    ) -> Optional[InstanceType]:
        """Decide if an instance should switch roles.

        NOTE: With PolicyMonitor, this method is only used for manual switching.
        Policy-driven switching goes through MONITOR_EVAL → ROLE_SWITCH events.

        Args:
            instance_type: Current role of the instance
            prefill_queue_len: Total system prefill queue length
            decode_queue_len: Total system decode queue length
            current_time: Current simulation time
            instance_is_idle: Whether the instance is currently idle

        Returns:
            Always returns None (policy-driven switching uses PolicyMonitor)
        """
        # Always return None - policy-driven switching uses monitor
        return None

    def record_switch(self, current_time: float):
        """Record that a switch occurred."""
        self._last_switch_time = current_time
