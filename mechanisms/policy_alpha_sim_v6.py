"""Policy Alpha V6 (Simulation Version with Hybrid Decode Memory Guard).

Alpha-based role switching policy for simulation using different decode memory signals for
each transition direction. This is adapted from policy_alpha_sim_v4.py with:
1. p→d uses avg(allocatable_kv / total_kv) — conservative signal accounting for prealloc reserves,
   guards against OOM when adding a new decode worker.
2. d→p uses avg(1 - free_kv / total_kv) (V4 style) — raw KV cache utilization, determines when
   decode has enough room to spare a worker back to prefill.
Thresholds:
   - decode_memory_low:      d→p trigger (usage < threshold, decode has room to spare)
   - decode_allocatable_low: p→d trigger (allocatable ratio < threshold, decode is tight)
"""

import argparse
from typing import Any, Dict, List

from mechanisms.worker_state import WorkerState, _get_metric, _idle_for_k_scrapes


class PolicyAlphaSimV6(object):
    """Policy alpha v6: alpha-led switching with hybrid decode memory guard.

    Direction:
      - if p95_prefill_alpha > alpha_threshold_high AND avg_decode_memory_usage < decode_memory_low
        for N stable evals => propose decode -> prefill  (uses raw KV utilization like V4)
      - if p95_prefill_alpha < alpha_threshold_low AND avg_allocatable_ratio < decode_allocatable_low
        for N stable evals => propose prefill -> decode  (uses allocatable ratio, conservative)

    Notes:
      - Candidate selection and cooldown guardrails mirror PolicyV4.
      - avg_decode_memory_usage = mean(1 - free_kv/total_kv) across decode workers.
      - avg_allocatable_ratio = mean(allocatable_kv_tokens / total_kv_tokens) across decode workers.
      - allocatable_kv_tokens accounts for virtual prealloc reservations missing from free_kv.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self._stable_dp = 0  # decode -> prefill
        self._stable_pd = 0  # prefill -> decode
        self._last_proposal_unix = 0.0
        self._worker_cooldown_until = {}  # type: Dict[int, float]

    def _in_cooldown(self, now_unix: float, port: int) -> bool:
        return now_unix < self._worker_cooldown_until.get(port, 0.0)

    def _set_cooldown(self, now_unix: float, port: int) -> None:
        self._worker_cooldown_until[port] = now_unix + float(self.args.per_worker_cooldown_s)

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

        def max_role(role_workers: List[WorkerState], metric: str) -> float:
            if not role_workers:
                return 0.0
            best = 0.0
            for w in role_workers:
                v = _get_metric(w.last_parsed, metric)
                candidate = float(v) if isinstance(v, (int, float)) else 0.0
                if candidate > best:
                    best = candidate
            return best

        def avg_role(role_workers: List[WorkerState], metric: str) -> float:
            if not role_workers:
                return 0.0
            tot = 0.0
            for w in role_workers:
                v = _get_metric(w.last_parsed, metric)
                tot += float(v) if isinstance(v, (int, float)) else 0.0
            return tot / float(len(role_workers))

        max_prefill_alpha = max_role(prefill, "max_alpha")
        avg_prefill_alpha = avg_role(prefill, "max_alpha")
        # p95_alpha is computed per-worker in policy_monitor (p95 of per-request cumulative alphas).
        # Taking max across workers: trigger if any worker's queue is 95% past SLO pressure.
        p95_prefill_alpha = max_role(prefill, "p95_alpha")

        alpha_threshold_low = float(getattr(self.args, "alpha_threshold_low", 0.6))
        alpha_threshold_high = float(getattr(self.args, "alpha_threshold_high", 1.0))

        # Calculate average KV cache utilization across decode workers (V4 style, used for d→p).
        # usage = 1 - free_kv/total_kv; higher means decode is more occupied.
        avg_decode_memory_usage = 0.0
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
                avg_decode_memory_usage = total_usage / count

        # Calculate average allocatable ratio across decode workers (used for p→d).
        # allocatable_kv_tokens mirrors _decode_allocatable_tokens():
        #   allocatable = free_kv - _prealloc_reserved_tokens - (num_reserved_decode_tokens * num_active_inflight)
        # Higher ratio means decode has more truly usable memory.
        avg_allocatable_ratio = 0.0
        if decode:
            total_ratio = 0.0
            count = 0
            for w in decode:
                total_kv = _get_metric(w.last_parsed, "total_kv_tokens")
                allocatable_kv = _get_metric(w.last_parsed, "allocatable_kv_tokens")
                if total_kv and total_kv > 0 and allocatable_kv is not None:
                    total_ratio += allocatable_kv / total_kv
                    count += 1
            if count > 0:
                avg_allocatable_ratio = total_ratio / count

        decode_memory_low = float(getattr(self.args, "decode_memory_low", 0.4))
        decode_allocatable_low = float(getattr(self.args, "decode_allocatable_low", 0.2))

        # Decode→Prefill: high alpha (p95), decode KV usage is low (has room to spare)
        dp_condition = (p95_prefill_alpha > alpha_threshold_high) and (avg_decode_memory_usage < decode_memory_low)

        # Prefill→Decode: low alpha (p95), decode allocatable ratio is tight
        pd_condition = (p95_prefill_alpha < alpha_threshold_low) and (avg_allocatable_ratio < decode_allocatable_low)

        if dp_condition and not pd_condition:
            self._stable_dp += 1
            self._stable_pd = 0
        elif pd_condition and not dp_condition:
            self._stable_pd += 1
            self._stable_dp = 0
        else:
            self._stable_dp = 0
            self._stable_pd = 0

        prefill_throughput = sum_role(prefill, "req_throughput")
        decode_throughput = sum_role(decode, "req_throughput")

        context = {
            "avg_decode_memory_usage": avg_decode_memory_usage,
            "decode_memory_low": decode_memory_low,
            "avg_allocatable_ratio": avg_allocatable_ratio,
            "decode_allocatable_low": decode_allocatable_low,
            "max_prefill_alpha": max_prefill_alpha,
            "avg_prefill_alpha": avg_prefill_alpha,
            "p95_prefill_alpha": p95_prefill_alpha,
            "decision_prefill_alpha": p95_prefill_alpha,
            "alpha_threshold_low": alpha_threshold_low,
            "alpha_threshold_high": alpha_threshold_high,
            "dp_condition": dp_condition,
            "pd_condition": pd_condition,
            "stable_dp": self._stable_dp,
            "stable_pd": self._stable_pd,
            "prefill_count": len(prefill),
            "decode_count": len(decode),
            "prefill_throughput": prefill_throughput,
            "decode_throughput": decode_throughput,
        }

        action = {"kind": "noop", "reason": "no_condition_met"}  # type: Dict[str, Any]

        if self._last_proposal_unix and (now_unix - self._last_proposal_unix) < float(self.args.global_cooldown_s):
            action = {"kind": "noop", "reason": "global_cooldown"}
            return {"policy": "alpha_v6", "context": context, "action": action}

        stable_evals = int(self.args.stable_evals)
        idle_k = int(self.args.idle_scrapes)
        allow_decode_to_prefill = getattr(self.args, "alpha_allow_decode_to_prefill", True)
        allow_prefill_to_decode = getattr(self.args, "alpha_allow_prefill_to_decode", True)

        # Decode→Prefill transition (alpha high, allocatable ratio high)
        if allow_decode_to_prefill and self._stable_dp >= stable_evals:
            if len(decode) <= int(self.args.min_decode):
                action = {"kind": "noop", "reason": "min_decode_guard"}
                return {"policy": "alpha_v6", "context": context, "action": action}

            candidates = [w for w in decode if not self._in_cooldown(now_unix, w.port)]
            if not candidates:
                action = {"kind": "noop", "reason": "no_decode_candidate"}
                return {"policy": "alpha_v6", "context": context, "action": action}

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
                "reason": "alpha_high_memory_low",
                "port": chosen.port,
                "from_role": "decode",
                "to_role": "prefill",
                "candidate": {
                    "idle_pref": _idle_for_k_scrapes(chosen, idle_k),
                    "num_running_reqs": _get_metric(chosen.last_parsed, "num_running_reqs"),
                },
            }
            return {"policy": "alpha_v6", "context": context, "action": action}

        # Prefill→Decode transition (alpha low, allocatable ratio low)
        if allow_prefill_to_decode and self._stable_pd >= stable_evals:
            if len(prefill) <= int(self.args.min_prefill):
                action = {"kind": "noop", "reason": "min_prefill_guard"}
                return {"policy": "alpha_v6", "context": context, "action": action}

            candidates = [w for w in prefill if not self._in_cooldown(now_unix, w.port)]
            if not candidates:
                action = {"kind": "noop", "reason": "no_prefill_candidate"}
                return {"policy": "alpha_v6", "context": context, "action": action}

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
                "reason": "alpha_low_allocatable_low",
                "port": chosen.port,
                "from_role": "prefill",
                "to_role": "decode",
                "candidate": {
                    "idle_pref": _idle_for_k_scrapes(chosen, idle_k),
                    "num_running_reqs": _get_metric(chosen.last_parsed, "num_running_reqs"),
                },
            }
            return {"policy": "alpha_v6", "context": context, "action": action}

        return {"policy": "alpha_v6", "context": context, "action": action}
