"""Dynamic LP-based partial prefill offloading mechanism.

Uses a sliding-window LP solver (SciPy SLSQP) to determine per-request
beta values that minimize total offloaded tokens while meeting SLO constraints.
Replaces the previous static beta policy.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Tuple

from scipy.optimize import minimize

from request.request import SimReq

MAX_SOLVER_ITER = 64
BETA_THRESHOLD = 0.01  # Betas below this are zeroed out


def prefill_inference_time(length: float) -> float:
    """Estimate prefill time for a given number of tokens (pure prefill, no decode mixing).

    Uses: t = 0.014654 + 6.944e-5 * prefill_tokens
    """
    return 0.014654 + 6.944e-5 * length


def offload_prefill_on_decode_time(
    length: float, decode_bs: int, decode_token_sum: int
) -> float:
    """Estimate time to compute offloaded prefill tokens on a decode instance.

    The offloaded portion runs as a prefill batch on the decode instance,
    potentially mixed with ongoing decode work.

    Uses the full profiling formula:
    t = 0.014654 + 6.944e-5 * length + 6.024e-5 * decode_bs + 8.463e-8 * decode_token_sum
    """
    return (
        0.014654
        + 6.944e-5 * length
        + 6.024e-5 * decode_bs
        + 8.463e-8 * decode_token_sum
    )


class _DESScheduleEvaluator:
    """Schedule evaluator that uses actual profiling formulas and decode state.

    Caches schedule evaluations for the SciPy solver callbacks.

    Uses a smooth formulation (always computes offload_duration) so the
    SLSQP solver can compute correct gradients.
    """

    def __init__(
        self,
        window: Sequence[SimReq],
        prefill_ready: float,
        decode_bs: int,
        decode_token_sum: int,
    ) -> None:
        self.window = window
        self.prefill_ready = prefill_ready
        self.decode_bs = decode_bs
        self.decode_token_sum = decode_token_sum
        self._cache_key: Optional[Tuple[float, ...]] = None
        self._cache_schedule: List[Dict[str, float]] = []

    def schedule(self, betas: Sequence[float]) -> List[Dict[str, float]]:
        key = tuple(float(b) for b in betas)
        if self._cache_key != key:
            self._cache_key = key
            self._cache_schedule = self._compute_schedule(key)
        return self._cache_schedule

    def _compute_schedule(self, betas: Sequence[float]) -> List[Dict[str, float]]:
        """Compute per-request schedule with actual profiling formulas."""
        current_time = self.prefill_ready
        schedule: List[Dict[str, float]] = []
        for req, beta in zip(self.window, betas):
            beta = min(max(beta, 0.0), 1.0)
            start = max(current_time, req.queue_entry_time)

            # Prefill duration: (1-beta) * context_tokens on prefill instance
            effective_prefill_tokens = max((1.0 - beta) * req.context_tokens, 0.0)
            prefill_duration = prefill_inference_time(effective_prefill_tokens)
            prefill_finish = start + prefill_duration

            # KV transfer time for the (1-beta) portion
            kv_transfer = 1.86683e-6 * effective_prefill_tokens

            # Offload duration: beta * context_tokens on decode instance.
            # Always compute (even for beta~0) to keep the function smooth
            # for the SLSQP solver's numerical gradient computation.
            offload_tokens = max(beta * req.context_tokens, 0.0)
            offload_duration = offload_prefill_on_decode_time(
                offload_tokens, self.decode_bs, self.decode_token_sum
            )

            # Total latency from queue entry to first decode token:
            # queue_wait + prefill + kv_transfer + offload
            total_latency = (
                (prefill_finish - req.queue_entry_time)
                + kv_transfer
                + offload_duration
            )

            schedule.append(
                {
                    "start": start,
                    "finish": prefill_finish,
                    "prefill_duration": prefill_duration,
                    "kv_transfer": kv_transfer,
                    "offload_duration": offload_duration,
                    "total_latency": total_latency,
                    "queue_time": start - req.queue_entry_time,
                }
            )
            current_time = prefill_finish  # Serial prefill assumption
        return schedule


class DynamicBetaSolver:
    """LP-based dynamic beta solver for partial prefill offloading.

    Adapts the standalone sliding-window LP solver for integration
    with the online DES simulator. Key differences from standalone:

    1. Uses actual decode instance state (batch size, token sum) for
       offload duration estimation.
    2. Applies betas to ALL requests in the window (not just the first),
       since the next scheduling opportunity is unknown.
    3. Wall-clock solver time is measured and returned for TTFT accounting.
    4. Two-phase approach: first checks if beta=0 meets all SLOs (no
       offloading needed). Only invokes SLSQP when offloading is required.
    """

    def __init__(self, slo_target: float, max_window_size: int = 5):
        self.slo_target = slo_target
        self.max_window_size = max_window_size

    def solve(
        self,
        requests: Sequence[SimReq],
        prefill_ready_time: float,
        decode_bs: int,
        decode_token_sum: int,
    ) -> Tuple[List[float], bool, float]:
        """Solve for optimal betas for a window of requests.

        Uses a two-phase approach:
        1. Check if beta=0 (no offloading) meets all SLO constraints.
           If so, return immediately (offloading is unnecessary).
        2. Otherwise, run smooth SLSQP solver to find minimal offloading.

        Args:
            requests: Up to max_window_size requests from the waiting queue.
            prefill_ready_time: Time when the prefill instance becomes available.
            decode_bs: Current decode batch size on the target decode instance.
            decode_token_sum: Current decode computed token sum.

        Returns:
            (betas, feasible, solve_time) - list of beta values,
            whether solution is feasible, and wall-clock solve time in seconds.
        """
        if not requests:
            return [], True, 0.0

        window = list(requests[: self.max_window_size])
        window_size = len(window)

        t_start = time.perf_counter()

        # Phase 1: Check if beta=0 already satisfies all constraints.
        # Use exact evaluation (no offload overhead when beta=0).
        if self._all_slo_met_at_zero(window, prefill_ready_time):
            solve_time = time.perf_counter() - t_start
            return [0.0] * window_size, True, solve_time

        # Phase 2: Need offloading. Use smooth evaluator for SLSQP.
        evaluator = _DESScheduleEvaluator(
            window, prefill_ready_time, decode_bs, decode_token_sum
        )

        bounds = [(0.0, 1.0) for _ in range(window_size)]
        x0 = [0.5] * window_size

        def objective(beta: Sequence[float]) -> float:
            return float(
                sum(float(b) * req.context_tokens for b, req in zip(beta, window))
            )

        constraints = self._build_constraints(evaluator, window)

        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": MAX_SOLVER_ITER, "ftol": 1e-9, "disp": False},
        )
        solve_time = time.perf_counter() - t_start

        feasible = bool(result.success) and result.x is not None
        if result.x is None:
            betas = [0.0] * window_size
        else:
            # Threshold tiny betas to zero to avoid useless offloads
            betas = [
                float(min(max(val, 0.0), 1.0)) if val > BETA_THRESHOLD else 0.0
                for val in result.x
            ]

        return betas, feasible, solve_time

    def _all_slo_met_at_zero(
        self, window: List[SimReq], prefill_ready_time: float
    ) -> bool:
        """Check if all SLO constraints are met with beta=0 (no offloading).

        Uses exact evaluation: no offload overhead when beta=0.
        """
        current_time = prefill_ready_time
        cumulative_prefill = 0.0
        for req in window:
            start = max(current_time, req.queue_entry_time)
            prefill_duration = prefill_inference_time(req.context_tokens)
            cumulative_prefill += prefill_duration
            kv_transfer = 1.86683e-6 * req.context_tokens
            queue_wait = start - req.queue_entry_time
            # No offload when beta=0
            total = cumulative_prefill + kv_transfer
            rhs = self.slo_target - queue_wait
            if total > rhs:
                return False
            current_time = start + prefill_duration
        return True

    def _build_constraints(
        self, evaluator: _DESScheduleEvaluator, window: List[SimReq]
    ):
        constraints = []
        for idx, req in enumerate(window):

            def constraint(beta, i=idx, request=req):
                schedule = evaluator.schedule(beta)
                cumulative_prefill = sum(
                    schedule[j]["prefill_duration"] for j in range(i + 1)
                )
                kv_transfer = schedule[i]["kv_transfer"]
                offload_duration = schedule[i]["offload_duration"]
                queue_wait = max(evaluator.prefill_ready, request.queue_entry_time) - request.queue_entry_time
                rhs = self.slo_target - queue_wait
                return rhs - (cumulative_prefill + kv_transfer + offload_duration)

            constraints.append({"type": "ineq", "fun": constraint})
        return constraints
