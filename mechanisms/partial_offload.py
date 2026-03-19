"""Dynamic LP-based partial prefill offloading mechanism.

Uses a greedy analytical solver to determine per-request beta values that
minimize total offloaded tokens while meeting SLO constraints.

The underlying LP has a lower-triangular constraint matrix, which allows
exact solution via O(N^2) greedy forward substitution instead of iterative
numerical optimization.
"""

from __future__ import annotations

import time
from typing import List, Sequence, Tuple

from request.request import SimReq

BETA_THRESHOLD = 0.01  # Betas below this are zeroed out

# Profiling model constants
_PREFILL_INTERCEPT = 0.014654
_PREFILL_COEFF = 6.944e-5  # 'a' - prefill time per token
_KV_TRANSFER_COEFF = 1.86683e-6  # 'k' - KV transfer time per token
_DECODE_BS_COEFF = 6.024e-5
_DECODE_DTS_COEFF = 8.463e-8


def prefill_inference_time(length: float) -> float:
    """Estimate prefill time for a given number of tokens (pure prefill, no decode mixing).

    Uses: t = 0.014654 + 6.944e-5 * prefill_tokens
    """
    return _PREFILL_INTERCEPT + _PREFILL_COEFF * length


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
        _PREFILL_INTERCEPT
        + _PREFILL_COEFF * length
        + _DECODE_BS_COEFF * decode_bs
        + _DECODE_DTS_COEFF * decode_token_sum
    )


class DynamicBetaSolver:
    """Analytical solver for partial prefill offloading betas.

    Exploits the fact that the optimization problem is a linear program
    with lower-triangular constraint structure, enabling exact O(N^2)
    greedy forward substitution.

    Mathematical formulation:
        minimize   sum_i  c_i * beta_i        (minimize total offloaded tokens)
        subject to:
            For each request i in window:
                a * sum_{j<i} c_j*beta_j + k * c_i*beta_i >= C_i - R_i
            0 <= beta_i <= 1

    Where:
        c_i = context_tokens of request i
        a   = prefill time coefficient (6.944e-5)
        k   = KV transfer coefficient (1.86683e-6)
        C_i = constant term (prefill + KV + offload overhead at beta=0)
        R_i = slo_target - queue_wait_i

    Key insight: increasing beta_j (j < i) reduces cumulative prefill for
    constraint i at rate a*c_j, while increasing beta_i only helps via KV
    transfer savings at rate k*c_i. Since a >> k (~37x), earlier betas are
    far more cost-effective, and all j < i have equal efficiency (a),
    so greedy earliest-first is optimal.
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
        2. Otherwise, solve analytically via greedy forward substitution.

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
        n = len(window)

        t_start = time.perf_counter()

        # Phase 1: Check if beta=0 already satisfies all constraints.
        if self._all_slo_met_at_zero(window, prefill_ready_time):
            solve_time = time.perf_counter() - t_start
            return [0.0] * n, True, solve_time

        # Phase 2: Analytical greedy forward substitution.
        a = _PREFILL_COEFF
        b = _PREFILL_INTERCEPT
        k = _KV_TRANSFER_COEFF
        D = b + _DECODE_BS_COEFF * decode_bs + _DECODE_DTS_COEFF * decode_token_sum

        c = [req.context_tokens for req in window]
        q = [
            max(prefill_ready_time, req.queue_entry_time) - req.queue_entry_time
            for req in window
        ]

        betas = [0.0] * n
        feasible = True

        # Precompute cumulative sum of context tokens
        cum_c = 0.0
        cum_c_list = []
        for ci in c:
            cum_c += ci
            cum_c_list.append(cum_c)

        for i in range(n):
            # Constant term: C_i = (i+1)*b + a*sum(c[0..i]) + k*c[i] + D
            C_i = (i + 1) * b + a * cum_c_list[i] + k * c[i] + D
            R_i = self.slo_target - q[i]

            # Current LHS reduction from betas set so far
            lhs_reduction = k * betas[i] * c[i]
            for j in range(i):
                lhs_reduction += a * betas[j] * c[j]

            deficit = (C_i - R_i) - lhs_reduction

            if deficit <= 0.0:
                continue  # Constraint already satisfied

            # Greedy: increase betas starting from earliest (j=0)
            # Earlier betas help all future constraints too
            for j in range(i + 1):
                if deficit <= 1e-12:
                    break
                rate = a * c[j] if j < i else k * c[j]
                if rate <= 0.0:
                    continue
                room = 1.0 - betas[j]
                if room <= 1e-12:
                    continue
                needed = deficit / rate
                increase = min(needed, room)
                betas[j] += increase
                deficit -= increase * rate

            if deficit > 1e-9:
                feasible = False

        # Threshold tiny betas to zero
        betas = [
            float(min(max(val, 0.0), 1.0)) if val > BETA_THRESHOLD else 0.0
            for val in betas
        ]

        solve_time = time.perf_counter() - t_start
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
            kv_transfer = _KV_TRANSFER_COEFF * req.context_tokens
            queue_wait = start - req.queue_entry_time
            # No offload when beta=0
            total = cumulative_prefill + kv_transfer
            rhs = self.slo_target - queue_wait
            if total > rhs:
                return False
            current_time = start + prefill_duration
        return True
