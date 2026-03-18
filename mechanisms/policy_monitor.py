"""Policy Monitor for periodic role switching decisions.

The monitor:
1. Runs at regular intervals (configured via monitor_interval_s)
2. Collects metrics from all instances
3. Calls policy.evaluate() to get switching decisions
4. Creates ROLE_SWITCH events when policy proposes switches
"""

from __future__ import annotations
import math
from typing import List, Optional, Dict, Any, TYPE_CHECKING
import argparse

from simulation.mechanisms.worker_state import WorkerState
from simulation.mechanisms.policy_alpha_sim import PolicyAlphaSim
from simulation.mechanisms.policy_v1_sim import PolicyV1Sim
from simulation.mechanisms.policy_throughput_sim import PolicyThroughputSim
from simulation.core.event import Event, EventType

if TYPE_CHECKING:
    from simulation.instances.base_instance import SimInstance


class PolicyMonitor:
    """Monitors system state and proposes role switches based on policy."""

    def __init__(
        self,
        policy: str = "never",
        monitor_interval_s: float = 5.0,
        # Alpha policy params
        alpha_threshold: float = 1.0,
        alpha_threshold_down: float = 0.5,
        # V1 policy params
        prefill_high: float = 10.0,
        prefill_low: float = 2.0,
        decode_high: float = 10.0,
        decode_low: float = 2.0,
        prefill_wait_weight: float = 1.0,
        prefill_active_weight: float = 0.5,
        prefill_inflight_weight: float = 0.3,
        decode_prealloc_weight: float = 1.0,
        decode_transfer_weight: float = 0.8,
        decode_active_weight: float = 0.5,
        decode_prefill_prealloc_weight: float = 0.3,
        # Common params
        stable_evals: int = 3,
        global_cooldown_s: float = 30.0,
        per_worker_cooldown_s: float = 60.0,
        min_prefill: int = 1,
        min_decode: int = 1,
        idle_scrapes: int = 2,
        # Alpha metric params
        slo_target: float = 1.0,
        # Throughput policy params
        max_prefill_tokens: int = 8192,
        max_running_requests: int = 1000,
    ):
        """Initialize policy monitor.

        Args:
            policy: Policy to use ("never", "alpha", "v1", "throughput")
            monitor_interval_s: How often to evaluate policy (seconds)
            slo_target: TTFT SLO target in seconds (for alpha computation)
            max_prefill_tokens: Max prefill tokens per batch (for throughput policy)
            max_running_requests: Max concurrent requests per instance (for throughput policy)
            ...: Policy-specific parameters
        """
        self.policy = policy
        self.monitor_interval_s = monitor_interval_s
        self.enabled = policy in ["alpha", "v1", "throughput"]
        self.slo_target = slo_target

        # Initialize policy implementation
        if policy == "alpha":
            args = argparse.Namespace(
                alpha_threshold=alpha_threshold,
                alpha_threshold_down=alpha_threshold_down,
                stable_evals=stable_evals,
                global_cooldown_s=global_cooldown_s,
                per_worker_cooldown_s=per_worker_cooldown_s,
                min_decode=min_decode,
                min_prefill=min_prefill,
                idle_scrapes=idle_scrapes,
            )
            self._policy_impl = PolicyAlphaSim(args)
        elif policy == "v1":
            args = argparse.Namespace(
                stable_evals=stable_evals,
                global_cooldown_s=global_cooldown_s,
                per_worker_cooldown_s=per_worker_cooldown_s,
                min_decode=min_decode,
                min_prefill=min_prefill,
                idle_scrapes=idle_scrapes,
                prefill_high=prefill_high,
                prefill_low=prefill_low,
                decode_high=decode_high,
                decode_low=decode_low,
                prefill_wait_weight=prefill_wait_weight,
                prefill_active_weight=prefill_active_weight,
                prefill_inflight_weight=prefill_inflight_weight,
                decode_prealloc_weight=decode_prealloc_weight,
                decode_transfer_weight=decode_transfer_weight,
                decode_active_weight=decode_active_weight,
                decode_prefill_prealloc_weight=decode_prefill_prealloc_weight,
            )
            self._policy_impl = PolicyV1Sim(args)
        elif policy == "throughput":
            args = argparse.Namespace(
                global_cooldown_s=global_cooldown_s,
                min_decode=min_decode,
                min_prefill=min_prefill,
                idle_scrapes=idle_scrapes,
                max_prefill_tokens=max_prefill_tokens,
                max_running_requests=max_running_requests,
            )
            self._policy_impl = PolicyThroughputSim(args)
        else:
            self._policy_impl = None

    def evaluate(
        self,
        all_instances: List[SimInstance],
        current_time: float,
        interval_metrics: Optional[Dict[str, float]] = None,
    ) -> List[Event]:
        """Evaluate policy and create ROLE_SWITCH events if needed.

        Args:
            all_instances: All instances (prefill + decode)
            current_time: Current simulation time
            interval_metrics: Throughput metrics for throughput-based policy

        Returns:
            List of ROLE_SWITCH events (empty if no switches proposed)
        """
        if not self.enabled or self._policy_impl is None:
            return []

        # Convert instances to WorkerState format
        states = {}
        port_to_instance = {}
        for inst in all_instances:
            worker_state = self._instance_to_worker_state(inst, current_time)
            states[worker_state.port] = worker_state
            port_to_instance[worker_state.port] = inst

        # Call policy
        if self.policy == "throughput":
            result = self._policy_impl.evaluate(
                current_time, states, interval_metrics=interval_metrics
            )
        else:
            result = self._policy_impl.evaluate(current_time, states)

        action = result.get("action", {})
        events = []

        # Handle single switch proposal (alpha, v1)
        if action.get("kind") == "switch_proposed":
            event = self._create_switch_event(action, port_to_instance, current_time)
            if event:
                events.append(event)

        # Handle multi-switch proposal (throughput)
        elif action.get("kind") == "multi_switch_proposed":
            for proposal in action.get("proposals", []):
                event = self._create_switch_event(proposal, port_to_instance, current_time)
                if event:
                    events.append(event)

        return events

    def _create_switch_event(
        self,
        action: Dict[str, Any],
        port_to_instance: Dict[int, SimInstance],
        current_time: float,
    ) -> Optional[Event]:
        """Create a ROLE_SWITCH event from a policy action dict.

        Args:
            action: Dict with 'port' and 'to_role' keys
            port_to_instance: Mapping from port to instance
            current_time: Current simulation time

        Returns:
            ROLE_SWITCH event, or None if instance not found
        """
        chosen_port = action.get("port")
        target_role_str = action.get("to_role")  # "prefill" or "decode"

        if chosen_port not in port_to_instance:
            return None

        instance = port_to_instance[chosen_port]

        from simulation.instances.base_instance import InstanceType
        target_role = (
            InstanceType.PREFILL if target_role_str == "prefill"
            else InstanceType.DECODE
        )

        return Event(
            timestamp=current_time,
            event_type=EventType.ROLE_SWITCH,
            data={
                "instance_id": instance.instance_id,
                "target_role": target_role,
            },
        )

    def _instance_to_port(self, instance: SimInstance) -> int:
        """Map instance ID to stable port number."""
        return hash(instance.instance_id) % 100000

    def _instance_to_worker_state(
        self, instance: SimInstance, current_time: float
    ) -> WorkerState:
        """Convert SimInstance to WorkerState."""
        from simulation.instances.base_instance import InstanceType

        port = self._instance_to_port(instance)
        role = "prefill" if instance.instance_type == InstanceType.PREFILL else "decode"

        # Build metrics dict
        metrics = {
            "ok": True,
            "num_running_reqs": len(instance.running_batch.reqs),
            "num_queue_reqs": len(instance.waiting_queue),
        }

        # Add role-specific metrics
        if role == "prefill":
            # PrefillInstance specific
            metrics["num_prefill_inflight_queue_reqs"] = len(instance.inflight_queue)
            # Compute max_alpha from waiting queue using real SGLang definition:
            # alpha_i = cumulative_vit / (SLO - waiting_time - transfer_time)
            max_alpha = 0.0
            cumulative_vit = 0.0
            for req in instance.waiting_queue:
                num_prefill_tokens = req.context_tokens - req.prefill_tokens_done
                vit = 6.944e-5 * num_prefill_tokens
                transfer_time = 1.86683e-6 * num_prefill_tokens
                waiting_time = max(0.0, current_time - req.arrival_time)
                budget = self.slo_target - waiting_time - transfer_time

                cumulative_vit += vit
                if budget > 0:
                    alpha = cumulative_vit / budget
                    max_alpha = max(max_alpha, alpha)
                else:
                    # Already past SLO deadline
                    max_alpha = float("inf")
                    break
            metrics["max_alpha"] = max_alpha
        else:
            # DecodeInstance specific
            metrics["num_decode_transfer_queue_reqs"] = len(instance.transfer_queue)
            metrics["num_decode_prealloc_queue_reqs"] = len(instance.prealloc_queue)

        return WorkerState(
            port=port,
            role=role,
            last_parsed=metrics,
        )
