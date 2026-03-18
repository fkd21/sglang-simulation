"""Per-request lifecycle trace logger.

Outputs one JSONL line per completed request with all timestamps
and derived durations for every stage and queue in the pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

from request.request import SimReq


class RequestTraceLogger:
    """Logs per-request lifecycle data to a JSONL file."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = self.output_dir / "request_traces.jsonl"
        self.output_path.write_text("")
        self.count = 0

    @staticmethod
    def _duration(end: float, start: float) -> float:
        """Compute duration, returning 0.0 if the end timestamp is unset (0.0)."""
        if end > 0:
            return end - start
        return 0.0

    def log_request(self, req: SimReq):
        """Log a completed request's lifecycle data."""
        record = {
            # Identity
            "rid": req.rid,
            "context_tokens": req.context_tokens,
            "generated_tokens": req.generated_tokens,
            # Instance assignment
            "prefill_instance": req.assigned_prefill_instance,
            "decode_instance": req.assigned_decode_instance,
            # Mechanism parameters (beta & M)
            "lp_beta": req.lp_beta,
            "partial_offload_amount": req.partial_offload_amount,
            "continued_on_prefill": req.continued_on_prefill,
            "is_chunked": req.is_chunked,
            # Timestamps
            "arrival_time": req.arrival_time,
            "queue_entry_time": req.queue_entry_time,
            "bootstrap_exit_time": req.bootstrap_exit_time,
            "prefill_start_time": req.prefill_start_time,
            "prefill_end_time": req.prefill_end_time,
            "continuation_start_time": req.continuation_start_time,
            "continuation_end_time": req.continuation_end_time,
            "kv_transfer_start_time": req.kv_transfer_start_time,
            "kv_transfer_end_time": req.kv_transfer_end_time,
            "offload_start_time": req.offload_start_time,
            "offload_end_time": req.offload_end_time,
            "decode_queue_entry_time": req.decode_queue_entry_time,
            "decode_start_time": req.decode_start_time,
            "decode_end_time": req.decode_end_time,
            "completion_time": req.completion_time,
            # Derived durations
            "bootstrap_queue_duration": self._duration(
                req.bootstrap_exit_time, req.arrival_time
            ),
            "prefill_wait_duration": self._duration(
                req.prefill_start_time, req.bootstrap_exit_time
            ),
            "prefill_duration": self._duration(
                req.prefill_end_time, req.prefill_start_time
            ),
            "continuation_duration": self._duration(
                req.continuation_end_time, req.continuation_start_time
            ),
            "kv_transfer_duration": self._duration(
                req.kv_transfer_end_time, req.kv_transfer_start_time
            ),
            "offload_duration": self._duration(
                req.offload_end_time, req.offload_start_time
            ),
            "decode_wait_duration": self._duration(
                req.decode_start_time, req.decode_queue_entry_time
            ),
            "decode_duration": self._duration(
                req.decode_end_time, req.decode_start_time
            ),
            "e2e_latency": self._duration(req.completion_time, req.arrival_time),
            "ttft": self._duration(req.prefill_end_time, req.arrival_time),
        }

        with open(self.output_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        self.count += 1

    def finalize(self):
        """Print summary."""
        size_kb = (
            self.output_path.stat().st_size / 1024
            if self.output_path.exists()
            else 0
        )
        print(
            f"  request_traces.jsonl: {self.count} requests ({size_kb:.1f} KB)"
        )
