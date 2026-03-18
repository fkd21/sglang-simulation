"""Policy Alpha (Simulation Version).

Alpha-based role switching policy for simulation.
This is adapted from policy_alpha.py with:
1. Local imports (worker_state module instead of monitor.*)
2. Added prefill→decode transition logic
"""

import argparse
from typing import Any, Dict, List

from mechanisms.worker_state import WorkerState, _get_metric, _idle_for_k_scrapes


class PolicyAlphaSim(object):
    """Policy alpha: propose bidirectional switches using prefill alpha.

    Direction:
      - if max_prefill_alpha > alpha_threshold for N stable evals => propose decode -> prefill
      - if max_prefill_alpha < alpha_threshold_down for N stable evals => propose prefill -> decode
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self._stable_dp = 0  # decode -> prefill counter
        self._stable_pd = 0  # prefill -> decode counter (NEW)
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
        alpha_threshold = float(getattr(self.args, "alpha_threshold", 1.0))
        alpha_threshold_down = float(getattr(self.args, "alpha_threshold_down", 0.5))

        # Decode→Prefill condition (alpha high)
        dp_condition = max_prefill_alpha > alpha_threshold

        # Prefill→Decode condition (alpha low) - NEW
        pd_condition = max_prefill_alpha < alpha_threshold_down

        if dp_condition:
            self._stable_dp += 1
        else:
            self._stable_dp = 0

        # NEW: Track prefill→decode stability
        if pd_condition:
            self._stable_pd += 1
        else:
            self._stable_pd = 0

        prefill_throughput = sum_role(prefill, "req_throughput")
        decode_throughput = sum_role(decode, "req_throughput")

        context = {
            "max_prefill_alpha": max_prefill_alpha,
            "avg_prefill_alpha": avg_prefill_alpha,
            "decision_prefill_alpha": max_prefill_alpha,
            "alpha_threshold": alpha_threshold,
            "alpha_threshold_down": alpha_threshold_down,
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
            return {"policy": "alpha", "context": context, "action": action}

        stable_evals = int(self.args.stable_evals)
        idle_k = int(self.args.idle_scrapes)

        # Decode→Prefill transition (alpha high)
        if self._stable_dp >= stable_evals:
            if len(decode) <= int(self.args.min_decode):
                action = {"kind": "noop", "reason": "min_decode_guard"}
                return {"policy": "alpha", "context": context, "action": action}

            candidates = [w for w in decode if not self._in_cooldown(now_unix, w.port)]
            if not candidates:
                action = {"kind": "noop", "reason": "no_decode_candidate"}
                return {"policy": "alpha", "context": context, "action": action}

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
            self._stable_pd = 0  # Reset other counter
            action = {
                "kind": "switch_proposed",
                "reason": "alpha_high",
                "port": chosen.port,
                "from_role": "decode",
                "to_role": "prefill",
                "candidate": {
                    "idle_pref": _idle_for_k_scrapes(chosen, idle_k),
                    "num_running_reqs": _get_metric(chosen.last_parsed, "num_running_reqs"),
                },
            }
            return {"policy": "alpha", "context": context, "action": action}

        # Prefill→Decode transition (alpha low) - NEW LOGIC
        if self._stable_pd >= stable_evals:
            if len(prefill) <= int(getattr(self.args, "min_prefill", 1)):
                action = {"kind": "noop", "reason": "min_prefill_guard"}
                return {"policy": "alpha", "context": context, "action": action}

            candidates = [w for w in prefill if not self._in_cooldown(now_unix, w.port)]
            if not candidates:
                action = {"kind": "noop", "reason": "no_prefill_candidate"}
                return {"policy": "alpha", "context": context, "action": action}

            def score_prefill(w: WorkerState) -> float:
                inflight = _get_metric(w.last_parsed, "num_prefill_inflight_queue_reqs") or 0.0
                queue = _get_metric(w.last_parsed, "num_queue_reqs") or 0.0
                return inflight + queue

            def key_prefill(w: WorkerState) -> Any:
                idle_pref = 0 if _idle_for_k_scrapes(w, idle_k) else 1
                running = _get_metric(w.last_parsed, "num_running_reqs")
                running_val = float(running) if running is not None else float("inf")
                return (idle_pref, running_val, score_prefill(w), w.port)

            chosen = sorted(candidates, key=key_prefill)[0]
            self._last_proposal_unix = now_unix
            self._set_cooldown(now_unix, chosen.port)
            self._stable_dp = 0  # Reset other counter
            self._stable_pd = 0
            action = {
                "kind": "switch_proposed",
                "reason": "alpha_low",
                "port": chosen.port,
                "from_role": "prefill",
                "to_role": "decode",
                "candidate": {
                    "idle_pref": _idle_for_k_scrapes(chosen, idle_k),
                    "num_running_reqs": _get_metric(chosen.last_parsed, "num_running_reqs"),
                },
            }
            return {"policy": "alpha", "context": context, "action": action}

        return {"policy": "alpha", "context": context, "action": action}
