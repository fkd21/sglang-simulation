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

from mechanisms.worker_state import WorkerState
from mechanisms.policy_alpha_sim import PolicyAlphaSim
from mechanisms.policy_alpha_sim_v2 import PolicyAlphaSimV2
from mechanisms.policy_v1_sim import PolicyV1Sim
from mechanisms.policy_throughput_sim import PolicyThroughputSim
from core.event import Event, EventType

if TYPE_CHECKING:
    from instances.base_instance import SimInstance


class PolicyMonitor:
    """Monitors system state and proposes role switches based on policy."""

    def __init__(
        self,
        policy: str = "never",
        monitor_interval_s: float = 5.0,
        # Alpha policy params
        alpha_threshold: float = 1.0,
        alpha_threshold_down: float = 0.5,
        alpha_allow_decode_to_prefill: bool = True,
        alpha_allow_prefill_to_decode: bool = True,
        # Alpha V2 policy params (with decode pressure)
        alpha_v2_threshold: float = 1.0,
        alpha_v2_allow_decode_to_prefill: bool = True,
        alpha_v2_allow_prefill_to_decode: bool = True,
        # Alpha V3 policy params (with decode memory)
        alpha_v3_threshold_low: float = 0.6,
        alpha_v3_threshold_high: float = 1.0,
        alpha_v3_allow_decode_to_prefill: bool = True,
        alpha_v3_allow_prefill_to_decode: bool = True,
        decode_memory_low: float = 0.4,
        decode_memory_high: float = 0.8,
        # Alpha V4 policy params (with average decode memory and p95 alpha)
        alpha_v4_threshold_low: float = 0.6,
        alpha_v4_threshold_high: float = 1.0,
        alpha_v4_allow_decode_to_prefill: bool = True,
        alpha_v4_allow_prefill_to_decode: bool = True,
        # Alpha V5 policy params (with Kalman Filter prediction)
        alpha_v5_threshold_low: float = 0.6,
        alpha_v5_threshold_high: float = 1.0,
        alpha_v5_allow_decode_to_prefill: bool = True,
        alpha_v5_allow_prefill_to_decode: bool = True,
        # Alpha V6 policy params (allocatable-ratio-based memory guard)
        alpha_v6_threshold_low: float = 0.6,
        alpha_v6_threshold_high: float = 1.0,
        alpha_v6_allow_decode_to_prefill: bool = True,
        alpha_v6_allow_prefill_to_decode: bool = True,
        decode_allocatable_low: float = 0.2,
        decode_allocatable_high: float = 0.6,
        num_reserved_decode_tokens: int = 512,
        # Kalman Filter params (for Alpha V5)
        kf_process_noise: float = 0.01,
        kf_measurement_noise: float = 0.1,
        kf_prediction_steps: int = 3,
        kf_velocity_threshold: float = 0.05,
        kf_min_stable_evals: int = 2,
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
        # EMA smoothing params for throughput policy
        ema_base_alpha: float = 0.3,
        ema_max_alpha: float = 0.7,
        ema_sensitivity_threshold: float = 0.2,
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
        self.enabled = policy in ["alpha", "alpha_v2", "alpha_v3", "alpha_v4", "alpha_v5", "alpha_v6", "v1", "throughput"]
        self.num_reserved_decode_tokens = num_reserved_decode_tokens
        self.slo_target = slo_target

        # Initialize policy implementation
        if policy == "alpha":
            args = argparse.Namespace(
                alpha_threshold=alpha_threshold,
                alpha_threshold_down=alpha_threshold_down,
                alpha_allow_decode_to_prefill=alpha_allow_decode_to_prefill,
                alpha_allow_prefill_to_decode=alpha_allow_prefill_to_decode,
                stable_evals=stable_evals,
                global_cooldown_s=global_cooldown_s,
                per_worker_cooldown_s=per_worker_cooldown_s,
                min_decode=min_decode,
                min_prefill=min_prefill,
                idle_scrapes=idle_scrapes,
            )
            self._policy_impl = PolicyAlphaSim(args)
        elif policy == "alpha_v2":
            args = argparse.Namespace(
                alpha_threshold=alpha_v2_threshold,
                alpha_allow_decode_to_prefill=alpha_v2_allow_decode_to_prefill,
                alpha_allow_prefill_to_decode=alpha_v2_allow_prefill_to_decode,
                decode_low=decode_low,
                decode_high=decode_high,
                decode_prealloc_weight=decode_prealloc_weight,
                decode_transfer_weight=decode_transfer_weight,
                decode_active_weight=decode_active_weight,
                decode_prefill_prealloc_weight=decode_prefill_prealloc_weight,
                stable_evals=stable_evals,
                global_cooldown_s=global_cooldown_s,
                per_worker_cooldown_s=per_worker_cooldown_s,
                min_decode=min_decode,
                min_prefill=min_prefill,
                idle_scrapes=idle_scrapes,
            )
            self._policy_impl = PolicyAlphaSimV2(args)
        elif policy == "alpha_v3":
            args = argparse.Namespace(
                alpha_threshold_low=alpha_v3_threshold_low,
                alpha_threshold_high=alpha_v3_threshold_high,
                alpha_allow_decode_to_prefill=alpha_v3_allow_decode_to_prefill,
                alpha_allow_prefill_to_decode=alpha_v3_allow_prefill_to_decode,
                decode_memory_low=decode_memory_low,
                decode_memory_high=decode_memory_high,
                stable_evals=stable_evals,
                global_cooldown_s=global_cooldown_s,
                per_worker_cooldown_s=per_worker_cooldown_s,
                min_decode=min_decode,
                min_prefill=min_prefill,
                idle_scrapes=idle_scrapes,
            )
            from mechanisms.policy_alpha_sim_v3 import PolicyAlphaSimV3
            self._policy_impl = PolicyAlphaSimV3(args)
        elif policy == "alpha_v4":
            args = argparse.Namespace(
                alpha_threshold_low=alpha_v4_threshold_low,
                alpha_threshold_high=alpha_v4_threshold_high,
                alpha_allow_decode_to_prefill=alpha_v4_allow_decode_to_prefill,
                alpha_allow_prefill_to_decode=alpha_v4_allow_prefill_to_decode,
                decode_memory_low=decode_memory_low,
                decode_memory_high=decode_memory_high,
                stable_evals=stable_evals,
                global_cooldown_s=global_cooldown_s,
                per_worker_cooldown_s=per_worker_cooldown_s,
                min_decode=min_decode,
                min_prefill=min_prefill,
                idle_scrapes=idle_scrapes,
            )
            from mechanisms.policy_alpha_sim_v4 import PolicyAlphaSimV4
            self._policy_impl = PolicyAlphaSimV4(args)
        elif policy == "alpha_v5":
            args = argparse.Namespace(
                alpha_threshold_low=alpha_v5_threshold_low,
                alpha_threshold_high=alpha_v5_threshold_high,
                alpha_allow_decode_to_prefill=alpha_v5_allow_decode_to_prefill,
                alpha_allow_prefill_to_decode=alpha_v5_allow_prefill_to_decode,
                decode_memory_low=decode_memory_low,
                decode_memory_high=decode_memory_high,
                stable_evals=stable_evals,
                global_cooldown_s=global_cooldown_s,
                per_worker_cooldown_s=per_worker_cooldown_s,
                min_decode=min_decode,
                min_prefill=min_prefill,
                idle_scrapes=idle_scrapes,
                # Kalman Filter params
                monitor_interval_s=monitor_interval_s,
                kf_process_noise=kf_process_noise,
                kf_measurement_noise=kf_measurement_noise,
                kf_prediction_steps=kf_prediction_steps,
                kf_velocity_threshold=kf_velocity_threshold,
                kf_min_stable_evals=kf_min_stable_evals,
            )
            from mechanisms.policy_alpha_sim_v5 import PolicyAlphaSimV5
            self._policy_impl = PolicyAlphaSimV5(args)
        elif policy == "alpha_v6":
            args = argparse.Namespace(
                alpha_threshold_low=alpha_v6_threshold_low,
                alpha_threshold_high=alpha_v6_threshold_high,
                alpha_allow_decode_to_prefill=alpha_v6_allow_decode_to_prefill,
                alpha_allow_prefill_to_decode=alpha_v6_allow_prefill_to_decode,
                decode_memory_low=decode_memory_low,
                decode_allocatable_low=decode_allocatable_low,
                stable_evals=stable_evals,
                global_cooldown_s=global_cooldown_s,
                per_worker_cooldown_s=per_worker_cooldown_s,
                min_decode=min_decode,
                min_prefill=min_prefill,
                idle_scrapes=idle_scrapes,
            )
            from mechanisms.policy_alpha_sim_v6 import PolicyAlphaSimV6
            self._policy_impl = PolicyAlphaSimV6(args)
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
                ema_base_alpha=ema_base_alpha,
                ema_max_alpha=ema_max_alpha,
                ema_sensitivity_threshold=ema_sensitivity_threshold,
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

        from instances.base_instance import InstanceType
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
        from instances.base_instance import InstanceType

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
            metrics["num_prefill_prealloc_queue_reqs"] = 0  # Prefill doesn't have prealloc queue yet
            # One pass: collect per-request cumulative alphas, derive max_alpha and p95_alpha.
            # alpha_i = cumulative_vit_i / (SLO - waiting_time_i - transfer_time_i)
            alpha_list = []
            cumulative_vit = 0.0
            for req in instance.waiting_queue:
                num_prefill_tokens = req.context_tokens - req.prefill_tokens_done
                vit = 6.944e-5 * num_prefill_tokens
                transfer_time = 1.86683e-6 * num_prefill_tokens
                waiting_time = max(0.0, current_time - req.arrival_time)
                budget = self.slo_target - waiting_time - transfer_time
                cumulative_vit += vit
                if budget > 0:
                    alpha_list.append(cumulative_vit / budget)
                else:
                    alpha_list.append(float("inf"))
                    break  # subsequent requests also inf
            if not alpha_list:
                max_alpha = 0.0
                p95_alpha = 0.0
            else:
                max_alpha = max(alpha_list)
                alpha_list.sort()  # inf floats to end naturally
                idx = 0.95 * (len(alpha_list) - 1)
                lower = int(idx)
                upper = min(lower + 1, len(alpha_list) - 1)
                weight = idx - lower
                # weight==0.0 guard: avoids inf * 0.0 = nan (IEEE 754)
                p95_alpha = (alpha_list[lower] if weight == 0.0
                             else alpha_list[lower] * (1 - weight) + alpha_list[upper] * weight)
            metrics["max_alpha"] = max_alpha
            metrics["p95_alpha"] = p95_alpha
        else:
            # DecodeInstance specific
            metrics["num_decode_transfer_queue_reqs"] = len(instance.transfer_queue)
            metrics["num_decode_prealloc_queue_reqs"] = len(instance.prealloc_queue)
            # Add memory metrics for v3/v4 policy
            metrics["total_kv_tokens"] = instance.token_to_kv_pool.total_kv_tokens
            metrics["free_kv_tokens"] = instance.token_to_kv_pool.available_size()
            # Allocatable tokens: mirrors _decode_allocatable_tokens() in engine.py
            # Subtracts virtual prealloc reservations and per-inflight-req reserve
            num_active_inflight = len(instance.running_batch.reqs) + len(instance.transfer_queue)
            reserved = self.num_reserved_decode_tokens * num_active_inflight
            metrics["allocatable_kv_tokens"] = max(
                0,
                instance.token_to_kv_pool.available_size()
                - instance._prealloc_reserved_tokens
                - reserved,
            )

        return WorkerState(
            port=port,
            role=role,
            last_parsed=metrics,
        )
