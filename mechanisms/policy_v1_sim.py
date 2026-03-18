"""Policy V1 (Simulation Version).

Queue-pressure-based bidirectional role switching policy for simulation.
This is adapted from policy_v1.py with local imports (worker_state module instead of monitor.*).
"""

import argparse
from typing import Any, Dict, List

from simulation.mechanisms.worker_state import WorkerState, _get_metric, _idle_for_k_scrapes


class PolicyV1Sim(object):
    """Policy v1: propose a single decode<->prefill role switch based on queue pressure.

    Definitions (aggregated across currently-healthy workers):
      prefill_pressure = Σ(prefill.prealloc) + a*Σ(prefill.inflight) + b*Σ(prefill.queue)
      decode_pressure  = Σ(decode.prealloc) + c*Σ(decode.transfer) + d*Σ(decode.queue)

    Direction:
      - if prefill_pressure is high AND decode_pressure is low => propose decode -> prefill
      - if decode_pressure is high AND prefill_pressure is low => propose prefill -> decode

    Guardrails:
      - stable-evals: require the same direction condition N evals in a row
      - min-prefill/min-decode: never propose violating minimum counts
      - global cooldown + per-worker cooldown
      - (soft preference) prefer candidates idle for K scrapes (num_running_reqs == 0), but do not require it

    Output:
      - returns a structured decision dict; caller decides how to log/actuate
      - decision-only (no switching in this step)
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

        def avg_role(role_workers: List[WorkerState], metric: str) -> float:
            if not role_workers:
                return 0.0
            tot = 0.0
            for w in role_workers:
                v = _get_metric(w.last_parsed, metric)
                tot += float(v) if isinstance(v, (int, float)) else 0.0
            return tot / float(len(role_workers))

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

        # Prefill pressure: waiting queue + active (+ optional inflight).
        prefill_pressure = (
            float(self.args.prefill_wait_weight) * sum_role(prefill, "num_queue_reqs")
            + float(self.args.prefill_active_weight) * sum_role(prefill, "num_running_reqs")
            + float(self.args.prefill_inflight_weight) * sum_role(prefill, "num_prefill_inflight_queue_reqs")
        )
        # Decode pressure: decode prealloc + transfer + active (+ optional cross: prefill prealloc).
        decode_pressure = (
            float(self.args.decode_prealloc_weight) * sum_role(decode, "num_decode_prealloc_queue_reqs")
            + float(self.args.decode_transfer_weight) * sum_role(decode, "num_decode_transfer_queue_reqs")
            + float(self.args.decode_active_weight) * sum_role(decode, "num_running_reqs")
            + float(self.args.decode_prefill_prealloc_weight) * sum_role(prefill, "num_prefill_prealloc_queue_reqs")
        )

        dp_condition = (prefill_pressure > float(self.args.prefill_high)) and (
            decode_pressure < float(self.args.decode_low)
        )
        pd_condition = (decode_pressure > float(self.args.decode_high)) and (
            prefill_pressure < float(self.args.prefill_low)
        )

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
        max_prefill_alpha = max_role(prefill, "max_alpha")
        avg_prefill_alpha = avg_role(prefill, "max_alpha")

        context = {
            "prefill_pressure": prefill_pressure,
            "decode_pressure": decode_pressure,
            "prefill_count": len(prefill),
            "decode_count": len(decode),
            "max_prefill_alpha": max_prefill_alpha,
            "avg_prefill_alpha": avg_prefill_alpha,
            "dp_condition": dp_condition,
            "pd_condition": pd_condition,
            "stable_dp": self._stable_dp,
            "stable_pd": self._stable_pd,
            "prefill_throughput": prefill_throughput,
            "decode_throughput": decode_throughput,
        }

        action = {"kind": "noop", "reason": "no_condition_met"}  # type: Dict[str, Any]

        # global cooldown
        if self._last_proposal_unix and (now_unix - self._last_proposal_unix) < float(self.args.global_cooldown_s):
            action = {"kind": "noop", "reason": "global_cooldown"}
            return {"policy": "v1", "context": context, "action": action}

        stable_evals = int(self.args.stable_evals)
        idle_k = int(self.args.idle_scrapes)

        if self._stable_dp >= stable_evals:
            if len(decode) <= int(self.args.min_decode):
                action = {"kind": "noop", "reason": "min_decode_guard"}
                return {"policy": "v1", "context": context, "action": action}

            candidates = [w for w in decode if not self._in_cooldown(now_unix, w.port)]
            if not candidates:
                action = {"kind": "noop", "reason": "no_decode_candidate"}
                return {"policy": "v1", "context": context, "action": action}

            def score_decode(w: WorkerState) -> float:
                transfer = _get_metric(w.last_parsed, "num_decode_transfer_queue_reqs") or 0.0
                prealloc = _get_metric(w.last_parsed, "num_decode_prealloc_queue_reqs") or 0.0
                queue = _get_metric(w.last_parsed, "num_queue_reqs") or 0.0
                return transfer + prealloc + queue

            def key_decode(w: WorkerState) -> Any:
                # Prefer idle-for-K if available, then lower running, then lower local queues.
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
                "reason": "prefill_high_decode_low",
                "port": chosen.port,
                "from_role": "decode",
                "to_role": "prefill",
                "candidate": {
                    "idle_pref": _idle_for_k_scrapes(chosen, idle_k),
                    "num_running_reqs": _get_metric(chosen.last_parsed, "num_running_reqs"),
                },
            }
            return {"policy": "v1", "context": context, "action": action}

        if self._stable_pd >= stable_evals:
            if len(prefill) <= int(self.args.min_prefill):
                action = {"kind": "noop", "reason": "min_prefill_guard"}
                return {"policy": "v1", "context": context, "action": action}

            candidates = [w for w in prefill if not self._in_cooldown(now_unix, w.port)]
            if not candidates:
                action = {"kind": "noop", "reason": "no_prefill_candidate"}
                return {"policy": "v1", "context": context, "action": action}

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
                "reason": "decode_high_prefill_low",
                "port": chosen.port,
                "from_role": "prefill",
                "to_role": "decode",
                "candidate": {
                    "idle_pref": _idle_for_k_scrapes(chosen, idle_k),
                    "num_running_reqs": _get_metric(chosen.last_parsed, "num_running_reqs"),
                },
            }
            return {"policy": "v1", "context": context, "action": action}

        return {"policy": "v1", "context": context, "action": action}
