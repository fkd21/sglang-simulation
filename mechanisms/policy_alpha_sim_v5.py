"""Policy Alpha V5 (Simulation Version with Kalman Filter Prediction).

Alpha-based role switching policy with predictive decision making.
Extends V4 with:
1. Kalman Filter prediction for p95_alpha and avg_memory (3 steps ahead = 15s)
2. OR logic: trigger if current OR predicted values meet condition
3. Adaptive stable_evals: reduce from 5→2 when velocity is high (rapid change)
4. Enhanced diagnostics for prediction analysis
"""

import argparse
from typing import Any, Dict, List

from mechanisms.worker_state import WorkerState, _get_metric, _idle_for_k_scrapes
from mechanisms.kalman_filter_1d import KalmanFilter1D


class PolicyAlphaSimV5(object):
    """Policy alpha v5: alpha-led switching with Kalman Filter prediction.

    Decision Logic (OR-based):
      - Decode→Prefill: (current_alpha > 1.0 OR predicted_alpha > 1.0) AND
                        (current_memory < 0.4 OR predicted_memory < 0.4)
      - Prefill→Decode: (current_alpha < 0.6 OR predicted_alpha < 0.6) AND
                        (current_memory > 0.8 OR predicted_memory > 0.8)

    Adaptive Stability:
      - High velocity (rapid change) → lower stable_evals (faster response)
      - Low velocity (stable) → normal stable_evals (conservative)

    Prediction Horizon:
      - 3 steps ahead (15 seconds with 5s monitor_interval)

    Notes:
      - Uses same p95 alpha and avg decode memory as V4
      - Kalman Filter tracks global metrics, not per-instance
      - Candidate selection and cooldown guardrails mirror V4
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self._stable_dp = 0  # decode -> prefill
        self._stable_pd = 0  # prefill -> decode
        self._last_proposal_unix = 0.0
        self._worker_cooldown_until = {}  # type: Dict[int, float]

        # Kalman Filters for predictive decision making
        dt = float(getattr(args, "monitor_interval_s", 5.0))
        process_noise = float(getattr(args, "kf_process_noise", 0.01))
        measurement_noise = float(getattr(args, "kf_measurement_noise", 0.1))

        self._kf_alpha = KalmanFilter1D(
            initial_value=0.0,
            process_noise=process_noise,
            measurement_noise=measurement_noise,
            dt=dt
        )

        self._kf_memory = KalmanFilter1D(
            initial_value=0.0,
            process_noise=process_noise,
            measurement_noise=measurement_noise,
            dt=dt
        )

        # Prediction parameters
        self._prediction_steps = int(getattr(args, "kf_prediction_steps", 3))
        self._velocity_threshold = float(getattr(args, "kf_velocity_threshold", 0.05))
        self._min_stable_evals = int(getattr(args, "kf_min_stable_evals", 2))

    def _in_cooldown(self, now_unix: float, port: int) -> bool:
        return now_unix < self._worker_cooldown_until.get(port, 0.0)

    def _set_cooldown(self, now_unix: float, port: int) -> None:
        self._worker_cooldown_until[port] = now_unix + float(self.args.per_worker_cooldown_s)

    def _calculate_required_stable_evals(self) -> int:
        """Calculate adaptive stable_evals based on velocity.

        High velocity (rapid change) → reduce stable_evals for faster response
        Low velocity (stable) → use normal stable_evals for conservative switching

        Returns:
            Required number of stable evaluations (2 or base value)
        """
        base_stable_evals = int(self.args.stable_evals)

        # Get velocity magnitudes (rate of change per time step)
        alpha_velocity = self._kf_alpha.get_velocity_magnitude()
        memory_velocity = self._kf_memory.get_velocity_magnitude()

        # Combine velocities (max indicates highest rate of change)
        max_velocity = max(alpha_velocity, memory_velocity)

        # Adaptive logic:
        # - If velocity > threshold: use minimum stable_evals (system changing rapidly)
        # - Else: use base stable_evals (system stable)
        if max_velocity > self._velocity_threshold:
            return self._min_stable_evals  # Fast response during rapid changes
        else:
            return base_stable_evals  # Conservative during stable periods

    def evaluate(self, now_unix: float, states: Dict[int, WorkerState]) -> Dict[str, Any]:
        eligible = []  # type: List[WorkerState]
        for _port, s in states.items():
            if not isinstance(s.last_parsed, dict) or not s.last_parsed.get("ok"):
                continue
            eligible.append(s)

        prefill = [s for s in eligible if s.role == "prefill"]
        decode = [s for s in eligible if s.role == "decode"]

        def sum_role(role_workers: List[WorkerState], metric: str) -> float:
            tot = 0.0
            for w in role_workers:
                v = _get_metric(w.last_parsed, metric)
                if v is not None:
                    tot += v
            return tot

        def percentile_role(role_workers: List[WorkerState], metric: str, pct: float) -> float:
            """Calculate percentile of metric across workers."""
            if not role_workers:
                return 0.0
            values = []
            for w in role_workers:
                v = _get_metric(w.last_parsed, metric)
                if isinstance(v, (int, float)):
                    values.append(float(v))
            if not values:
                return 0.0
            values.sort()
            # Linear interpolation for percentile
            idx = pct * (len(values) - 1)
            lower_idx = int(idx)
            upper_idx = min(lower_idx + 1, len(values) - 1)
            weight = idx - lower_idx
            return values[lower_idx] * (1 - weight) + values[upper_idx] * weight

        def avg_role(role_workers: List[WorkerState], metric: str) -> float:
            if not role_workers:
                return 0.0
            tot = 0.0
            for w in role_workers:
                v = _get_metric(w.last_parsed, metric)
                tot += float(v) if isinstance(v, (int, float)) else 0.0
            return tot / float(len(role_workers))

        # ===== CURRENT METRICS (same as V4) =====
        current_p95_alpha = percentile_role(prefill, "max_alpha", 0.95)

        # Calculate average memory usage across decode workers
        current_avg_memory = 0.0
        if decode:
            total_usage = 0.0
            count = 0
            for w in decode:
                total_kv = _get_metric(w.last_parsed, "total_kv_tokens")
                free_kv = _get_metric(w.last_parsed, "free_kv_tokens")
                if total_kv and total_kv > 0:
                    usage = 1.0 - (free_kv / total_kv)
                    total_usage += usage
                    count += 1
            if count > 0:
                current_avg_memory = total_usage / count

        # ===== UPDATE KALMAN FILTERS =====
        self._kf_alpha.update(current_p95_alpha)
        self._kf_memory.update(current_avg_memory)

        # ===== PREDICT FUTURE VALUES =====
        predicted_alpha, alpha_velocity = self._kf_alpha.predict(self._prediction_steps)
        predicted_memory, memory_velocity = self._kf_memory.predict(self._prediction_steps)

        # ===== THRESHOLDS =====
        alpha_threshold_low = float(getattr(self.args, "alpha_threshold_low", 0.6))
        alpha_threshold_high = float(getattr(self.args, "alpha_threshold_high", 1.0))
        decode_memory_low = float(self.args.decode_memory_low)
        decode_memory_high = float(self.args.decode_memory_high)

        # ===== OR-BASED CONDITION LOGIC =====
        # Decode→Prefill: (current OR predicted) alpha high AND (current OR predicted) memory low
        alpha_high_condition = (current_p95_alpha > alpha_threshold_high) or \
                               (predicted_alpha > alpha_threshold_high)
        memory_low_condition = (current_avg_memory < decode_memory_low) or \
                              (predicted_memory < decode_memory_low)
        dp_condition = alpha_high_condition and memory_low_condition

        # Prefill→Decode: (current OR predicted) alpha low AND (current OR predicted) memory high
        alpha_low_condition = (current_p95_alpha < alpha_threshold_low) or \
                             (predicted_alpha < alpha_threshold_low)
        memory_high_condition = (current_avg_memory > decode_memory_high) or \
                               (predicted_memory > decode_memory_high)
        pd_condition = alpha_low_condition and memory_high_condition

        # ===== UPDATE STABILITY COUNTERS =====
        if dp_condition and not pd_condition:
            self._stable_dp += 1
            self._stable_pd = 0
        elif pd_condition and not dp_condition:
            self._stable_pd += 1
            self._stable_dp = 0
        else:
            self._stable_dp = 0
            self._stable_pd = 0

        # ===== CALCULATE ADAPTIVE STABLE_EVALS =====
        required_stable_evals = self._calculate_required_stable_evals()

        # ===== THROUGHPUT METRICS =====
        prefill_throughput = sum_role(prefill, "req_throughput")
        decode_throughput = sum_role(decode, "req_throughput")

        # ===== ENHANCED DIAGNOSTIC CONTEXT =====
        context = {
            # Current metrics
            "current_p95_alpha": current_p95_alpha,
            "current_avg_memory": current_avg_memory,
            # Predicted metrics
            "predicted_alpha": predicted_alpha,
            "predicted_memory": predicted_memory,
            "prediction_horizon_s": self._prediction_steps * self._kf_alpha.dt,
            # Velocities
            "alpha_velocity": alpha_velocity,
            "memory_velocity": memory_velocity,
            "max_velocity": max(abs(alpha_velocity), abs(memory_velocity)),
            # Kalman Filter state
            "kf_alpha_measurements": self._kf_alpha.measurement_count,
            "kf_memory_measurements": self._kf_memory.measurement_count,
            # Thresholds
            "alpha_threshold_low": alpha_threshold_low,
            "alpha_threshold_high": alpha_threshold_high,
            "decode_memory_low": decode_memory_low,
            "decode_memory_high": decode_memory_high,
            # Condition breakdown
            "alpha_high_condition": alpha_high_condition,
            "memory_low_condition": memory_low_condition,
            "alpha_low_condition": alpha_low_condition,
            "memory_high_condition": memory_high_condition,
            "dp_condition": dp_condition,
            "pd_condition": pd_condition,
            # Stability tracking
            "stable_dp": self._stable_dp,
            "stable_pd": self._stable_pd,
            "required_stable_evals": required_stable_evals,
            "base_stable_evals": int(self.args.stable_evals),
            "velocity_threshold": self._velocity_threshold,
            # System state
            "prefill_count": len(prefill),
            "decode_count": len(decode),
            "prefill_throughput": prefill_throughput,
            "decode_throughput": decode_throughput,
        }

        action = {"kind": "noop", "reason": "no_condition_met"}  # type: Dict[str, Any]

        # Global cooldown check
        if self._last_proposal_unix and (now_unix - self._last_proposal_unix) < float(self.args.global_cooldown_s):
            action = {"kind": "noop", "reason": "global_cooldown"}
            return {"policy": "alpha_v5", "context": context, "action": action}

        # Get switching flags
        idle_k = int(self.args.idle_scrapes)
        allow_decode_to_prefill = getattr(self.args, "alpha_allow_decode_to_prefill", True)
        allow_prefill_to_decode = getattr(self.args, "alpha_allow_prefill_to_decode", True)

        # ===== DECODE→PREFILL TRANSITION =====
        if allow_decode_to_prefill and self._stable_dp >= required_stable_evals:
            if len(decode) <= int(self.args.min_decode):
                action = {"kind": "noop", "reason": "min_decode_guard"}
                return {"policy": "alpha_v5", "context": context, "action": action}

            candidates = [w for w in decode if not self._in_cooldown(now_unix, w.port)]
            if not candidates:
                action = {"kind": "noop", "reason": "no_decode_candidate"}
                return {"policy": "alpha_v5", "context": context, "action": action}

            # Selection logic (same as V4)
            def score_decode(w: WorkerState) -> float:
                transfer = _get_metric(w.last_parsed, "num_decode_transfer_queue_reqs") or 0.0
                prealloc = _get_metric(w.last_parsed, "num_decode_prealloc_queue_reqs") or 0.0
                queue = _get_metric(w.last_parsed, "num_queue_reqs") or 0.0
                return transfer + prealloc + queue

            def key_decode(w: WorkerState) -> Any:
                idle_pref = 0 if _idle_for_k_scrapes(w, idle_k) else 1
                running = _get_metric(w.last_parsed, "num_running_reqs")
                running_val = float(running) if running is not None else float("inf")
                return (idle_pref, running_val, score_decode(w), w.port)

            chosen = sorted(candidates, key=key_decode)[0]
            self._last_proposal_unix = now_unix
            self._set_cooldown(now_unix, chosen.port)
            self._stable_dp = 0
            self._stable_pd = 0
            action = {
                "kind": "switch_proposed",
                "reason": "alpha_high_memory_low_predicted",
                "port": chosen.port,
                "from_role": "decode",
                "to_role": "prefill",
                "candidate": {
                    "idle_pref": _idle_for_k_scrapes(chosen, idle_k),
                    "num_running_reqs": _get_metric(chosen.last_parsed, "num_running_reqs"),
                },
            }
            return {"policy": "alpha_v5", "context": context, "action": action}

        # ===== PREFILL→DECODE TRANSITION =====
        if allow_prefill_to_decode and self._stable_pd >= required_stable_evals:
            if len(prefill) <= int(self.args.min_prefill):
                action = {"kind": "noop", "reason": "min_prefill_guard"}
                return {"policy": "alpha_v5", "context": context, "action": action}

            candidates = [w for w in prefill if not self._in_cooldown(now_unix, w.port)]
            if not candidates:
                action = {"kind": "noop", "reason": "no_prefill_candidate"}
                return {"policy": "alpha_v5", "context": context, "action": action}

            # Selection logic (same as V4)
            def score_prefill(w: WorkerState) -> float:
                inflight = _get_metric(w.last_parsed, "num_prefill_inflight_queue_reqs") or 0.0
                prealloc = _get_metric(w.last_parsed, "num_prefill_prealloc_queue_reqs") or 0.0
                queue = _get_metric(w.last_parsed, "num_queue_reqs") or 0.0
                return inflight + prealloc + queue

            def key_prefill(w: WorkerState) -> Any:
                idle_pref = 0 if _idle_for_k_scrapes(w, idle_k) else 1
                running = _get_metric(w.last_parsed, "num_running_reqs")
                running_val = float(running) if running is not None else float("inf")
                return (idle_pref, running_val, score_prefill(w), w.port)

            chosen = sorted(candidates, key=key_prefill)[0]
            self._last_proposal_unix = now_unix
            self._set_cooldown(now_unix, chosen.port)
            self._stable_dp = 0
            self._stable_pd = 0
            action = {
                "kind": "switch_proposed",
                "reason": "alpha_low_memory_high_predicted",
                "port": chosen.port,
                "from_role": "prefill",
                "to_role": "decode",
                "candidate": {
                    "idle_pref": _idle_for_k_scrapes(chosen, idle_k),
                    "num_running_reqs": _get_metric(chosen.last_parsed, "num_running_reqs"),
                },
            }
            return {"policy": "alpha_v5", "context": context, "action": action}

        return {"policy": "alpha_v5", "context": context, "action": action}
