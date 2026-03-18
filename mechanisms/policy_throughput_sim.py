"""Policy Throughput (Simulation Version).

Throughput-based role switching policy for simulation.
Computes optimal prefill/decode instance ratio based on observed
input/output token throughput and per-instance capability estimates
derived from the profiling formula.
"""

import argparse
from typing import Any, Dict, List, Optional

from mechanisms.worker_state import WorkerState, _get_metric, _idle_for_k_scrapes
from utils.constants import TOTAL_KV_CACHE_TOKENS

# Profiling formula constants
INTERCEPT = 0.014654
PREFILL_SLOPE = 6.944e-5
DECODE_BS_SLOPE = 6.024e-5
DECODE_TOKEN_SUM_SLOPE = 8.463e-8


class PolicyThroughputSim:
    """Throughput-based policy: balance prefill/decode capacity to match demand.

    Computes per-instance capability using the profiling formula, then derives
    the optimal np:nd ratio from the balance equation:
        np * p_capability / p_throughput = nd * d_capability / d_throughput
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self._last_proposal_unix = 0.0

    def evaluate(
        self,
        now_unix: float,
        states: Dict[int, WorkerState],
        *,
        interval_metrics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Evaluate throughput-based switching policy.

        Args:
            now_unix: Current simulation time
            states: Worker states keyed by port
            interval_metrics: Throughput metrics from the monitoring interval:
                - input_throughput: prefill tokens/sec
                - output_throughput: decode tokens/sec (1 per request per iteration)
                - avg_decode_computed_token_sum: avg per decode batch
                - avg_decode_batch_size: avg decode batch size

        Returns:
            Policy result dict with action (noop or multi_switch_proposed)
        """
        prefill = [s for s in states.values() if s.role == "prefill"]
        decode = [s for s in states.values() if s.role == "decode"]
        current_np = len(prefill)
        current_nd = len(decode)
        N = current_np + current_nd

        context = {
            "current_np": current_np,
            "current_nd": current_nd,
            "total_instances": N,
        }

        # Guard: need at least 2 total instances to do any switching
        if N < 2:
            return self._noop("too_few_instances", context)

        # Guard: no interval metrics available
        if interval_metrics is None:
            return self._noop("no_interval_metrics", context)

        input_throughput = interval_metrics.get("input_throughput", 0.0)
        output_throughput = interval_metrics.get("output_throughput", 0.0)
        avg_dcts = interval_metrics.get("avg_decode_computed_token_sum", 0.0)
        avg_dbs = interval_metrics.get("avg_decode_batch_size", 0.0)

        context.update({
            "input_throughput": input_throughput,
            "output_throughput": output_throughput,
            "avg_decode_computed_token_sum": avg_dcts,
            "avg_decode_batch_size": avg_dbs,
        })

        # Guard: zero throughput means we can't make a meaningful decision
        if input_throughput <= 0 or output_throughput <= 0:
            return self._noop("zero_throughput", context)

        # Guard: insufficient decode data
        if avg_dbs <= 0:
            return self._noop("insufficient_decode_data", context)

        # Guard: global cooldown
        cooldown_s = float(getattr(self.args, "global_cooldown_s", 30.0))
        if self._last_proposal_unix > 0 and (now_unix - self._last_proposal_unix) < cooldown_s:
            return self._noop("global_cooldown", context)

        # Compute prefill capability (per instance)
        max_prefill_tokens = int(getattr(self.args, "max_prefill_tokens", 8192))
        t_prefill = INTERCEPT + PREFILL_SLOPE * max_prefill_tokens
        p_capability = max_prefill_tokens / t_prefill

        # Compute decode capability (per instance)
        avg_kv = avg_dcts / avg_dbs
        max_running_requests = int(getattr(self.args, "max_running_requests", 1000))
        max_bs = min(max_running_requests, TOTAL_KV_CACHE_TOKENS / avg_kv) if avg_kv > 0 else max_running_requests
        t_decode = INTERCEPT + (DECODE_BS_SLOPE + DECODE_TOKEN_SUM_SLOPE * avg_kv) * max_bs
        d_capability = max_bs / t_decode

        context.update({
            "p_capability": p_capability,
            "d_capability": d_capability,
            "avg_kv_per_request": avg_kv,
            "max_decode_batch_size": max_bs,
        })

        # Compute target np:nd ratio
        # np * p_cap / p_thru = nd * d_cap / d_thru
        # => np/nd = (d_cap * p_thru) / (p_cap * d_thru)
        ratio = (d_capability * input_throughput) / (p_capability * output_throughput)
        target_np = round(N * ratio / (1 + ratio))

        # Clamp to ensure min 1 of each
        min_p = int(getattr(self.args, "min_prefill", 1))
        min_d = int(getattr(self.args, "min_decode", 1))
        target_np = max(min_p, min(target_np, N - min_d))
        target_nd = N - target_np

        context.update({
            "ratio": ratio,
            "target_np": target_np,
            "target_nd": target_nd,
        })

        delta = target_np - current_np

        if delta == 0:
            return self._noop("already_balanced", context)

        # Generate switch proposals
        idle_k = int(getattr(self.args, "idle_scrapes", 2))
        proposals = []

        if delta > 0:
            # Need more prefill: switch `delta` decode instances to prefill
            candidates = [w for w in decode if self._is_switchable(w)]
            candidates.sort(key=lambda w: self._score_candidate(w, idle_k))
            for c in candidates[:delta]:
                proposals.append({
                    "port": c.port,
                    "from_role": "decode",
                    "to_role": "prefill",
                })
        else:
            # Need more decode: switch `|delta|` prefill instances to decode
            candidates = [w for w in prefill if self._is_switchable(w)]
            candidates.sort(key=lambda w: self._score_candidate(w, idle_k))
            for c in candidates[:abs(delta)]:
                proposals.append({
                    "port": c.port,
                    "from_role": "prefill",
                    "to_role": "decode",
                })

        if not proposals:
            return self._noop("no_switchable_candidates", context)

        self._last_proposal_unix = now_unix

        return {
            "policy": "throughput",
            "context": context,
            "action": {
                "kind": "multi_switch_proposed",
                "reason": "throughput_balance",
                "proposals": proposals,
            },
        }

    def _is_switchable(self, w: WorkerState) -> bool:
        """Check if a worker is eligible for switching."""
        if not isinstance(w.last_parsed, dict):
            return False
        if not w.last_parsed.get("ok", False):
            return False
        return True

    def _score_candidate(self, w: WorkerState, idle_k: int) -> tuple:
        """Score a candidate for switching (lower is better).

        Prefer idle instances, then least running requests, then least queued work.
        """
        idle_pref = 0 if _idle_for_k_scrapes(w, idle_k) else 1
        running = _get_metric(w.last_parsed, "num_running_reqs")
        running_val = float(running) if running is not None else float("inf")
        queue = _get_metric(w.last_parsed, "num_queue_reqs") or 0.0
        return (idle_pref, running_val, queue, w.port)

    def _noop(self, reason: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Return a noop result."""
        return {
            "policy": "throughput",
            "context": context,
            "action": {"kind": "noop", "reason": reason},
        }
