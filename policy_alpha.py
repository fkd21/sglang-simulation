import argparse
from typing import Any, Dict, List

from monitor.metrics import _get_metric, _idle_for_k_scrapes
from monitor.state import WorkerState


class PolicyAlpha(object):
    """Policy alpha: propose decode->prefill switches using prefill alpha only.

    Direction:
      - if max_prefill_alpha > alpha_threshold_up for N stable evals => propose decode -> prefill
        if max_prefill_alpha < alpha_threshold_down for N stable evals => propose prefill -> decode
    Notes:
      - This policy currently does not propose prefill -> decode switches. #todo relealize this. 
      - Candidate selection and cooldown guardrails mirror PolicyV1.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self._stable_dp = 0  # decode -> prefill
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
        dp_condition = max_prefill_alpha > alpha_threshold

        if dp_condition:
            self._stable_dp += 1
        else:
            self._stable_dp = 0

        prefill_throughput = sum_role(prefill, "req_throughput")
        decode_throughput = sum_role(decode, "req_throughput")

        context = {
            "max_prefill_alpha": max_prefill_alpha,
            "avg_prefill_alpha": avg_prefill_alpha,
            "decision_prefill_alpha": max_prefill_alpha,
            "alpha_threshold": alpha_threshold,
            "dp_condition": dp_condition,
            "stable_dp": self._stable_dp,
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

        return {"policy": "alpha", "context": context, "action": action}
