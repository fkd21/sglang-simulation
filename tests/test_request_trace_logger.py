"""Tests for per-request lifecycle trace logging."""

import json
import os
import shutil
import tempfile

import pytest

from config import SimConfig
from core.engine import SimulationEngine


def _create_trace_file(requests):
    """Create a temp JSONL trace file from list of (input_len, output_len) tuples."""
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w") as f:
        for input_len, output_len in requests:
            f.write(
                json.dumps({"input_len": input_len, "output_len": output_len}) + "\n"
            )
    return path


class TestRequestTraceLogger:
    """Test per-request lifecycle trace logging."""

    def _run_and_get_traces(self, requests, **config_kwargs):
        """Run simulation and return (results, list of trace dicts, output_dir)."""
        trace_path = _create_trace_file(requests)
        try:
            config = SimConfig(trace_path=trace_path, **config_kwargs)
            engine = SimulationEngine(config, enable_iteration_logging=True)
            results = engine.run()

            trace_file = engine.iteration_logger.output_dir / "request_traces.jsonl"
            assert trace_file.exists(), "request_traces.jsonl should be created"

            traces = []
            with open(trace_file) as f:
                for line in f:
                    if line.strip():
                        traces.append(json.loads(line))

            output_dir = engine.iteration_logger.output_dir
            return results, traces, output_dir
        finally:
            os.unlink(trace_path)

    def test_trace_file_created(self):
        """Running simulation creates request_traces.jsonl."""
        results, traces, output_dir = self._run_and_get_traces([(100, 10)])
        assert len(traces) == 1
        shutil.rmtree(output_dir, ignore_errors=True)

    def test_trace_count_matches_completions(self):
        """Number of JSONL lines equals number of completed requests."""
        reqs = [(50, 5), (100, 10), (200, 3)]
        results, traces, output_dir = self._run_and_get_traces(reqs)
        assert len(traces) == results.total_requests
        shutil.rmtree(output_dir, ignore_errors=True)

    def test_timestamps_monotonic(self):
        """For each request, key timestamps are in lifecycle order."""
        results, traces, output_dir = self._run_and_get_traces([(256, 20)])
        t = traces[0]

        # arrival <= bootstrap_exit <= prefill_start <= prefill_end
        assert t["arrival_time"] <= t["bootstrap_exit_time"]
        assert t["bootstrap_exit_time"] <= t["prefill_start_time"]
        assert t["prefill_start_time"] <= t["prefill_end_time"]

        # prefill_end <= kv_transfer_start <= kv_transfer_end
        assert t["prefill_end_time"] <= t["kv_transfer_start_time"]
        assert t["kv_transfer_start_time"] <= t["kv_transfer_end_time"]

        # kv_transfer_end <= decode_queue_entry <= decode_start <= decode_end
        assert t["kv_transfer_end_time"] <= t["decode_queue_entry_time"]
        assert t["decode_queue_entry_time"] <= t["decode_start_time"]
        assert t["decode_start_time"] <= t["decode_end_time"]

        # decode_end <= completion
        assert t["decode_end_time"] <= t["completion_time"]

        shutil.rmtree(output_dir, ignore_errors=True)

    def test_derived_durations_non_negative(self):
        """All derived durations are >= 0."""
        reqs = [(100, 10), (200, 5)]
        results, traces, output_dir = self._run_and_get_traces(reqs)

        duration_keys = [
            "bootstrap_queue_duration",
            "prefill_wait_duration",
            "prefill_duration",
            "kv_transfer_duration",
            "decode_wait_duration",
            "decode_duration",
            "e2e_latency",
            "ttft",
        ]

        for t in traces:
            for key in duration_keys:
                assert t[key] >= 0, f"{key} should be >= 0 for {t['rid']}"

        shutil.rmtree(output_dir, ignore_errors=True)

    def test_e2e_latency_positive(self):
        """E2E latency is positive for completed requests."""
        results, traces, output_dir = self._run_and_get_traces([(100, 10)])
        assert traces[0]["e2e_latency"] > 0
        shutil.rmtree(output_dir, ignore_errors=True)

    def test_mechanism_fields_baseline(self):
        """Baseline requests have zero offload/continuation values."""
        results, traces, output_dir = self._run_and_get_traces([(100, 10)])
        t = traces[0]
        assert t["lp_beta"] == 0.0
        assert t["partial_offload_amount"] == 0
        assert t["continued_on_prefill"] == 0
        assert t["continuation_duration"] == 0.0
        assert t["offload_duration"] == 0.0
        shutil.rmtree(output_dir, ignore_errors=True)

    def test_continuation_fields_with_M(self):
        """With M>0, continuation timestamps and duration are set."""
        results, traces, output_dir = self._run_and_get_traces(
            [(256, 20)], M=5, enable_continuation=True
        )
        t = traces[0]
        assert t["continued_on_prefill"] == 5
        assert t["continuation_start_time"] > 0
        assert t["continuation_end_time"] > 0
        assert t["continuation_duration"] > 0
        shutil.rmtree(output_dir, ignore_errors=True)

    def test_trace_has_all_expected_fields(self):
        """Each trace record contains all expected fields."""
        results, traces, output_dir = self._run_and_get_traces([(100, 5)])
        t = traces[0]

        expected_fields = [
            "rid", "context_tokens", "generated_tokens",
            "prefill_instance", "decode_instance",
            "lp_beta", "partial_offload_amount", "continued_on_prefill", "is_chunked",
            "arrival_time", "queue_entry_time", "bootstrap_exit_time",
            "prefill_start_time", "prefill_end_time",
            "continuation_start_time", "continuation_end_time",
            "kv_transfer_start_time", "kv_transfer_end_time",
            "offload_start_time", "offload_end_time",
            "decode_queue_entry_time", "decode_start_time", "decode_end_time",
            "completion_time",
            "bootstrap_queue_duration", "prefill_wait_duration", "prefill_duration",
            "continuation_duration", "kv_transfer_duration", "offload_duration",
            "decode_wait_duration", "decode_duration", "e2e_latency", "ttft",
        ]

        for field in expected_fields:
            assert field in t, f"Missing field: {field}"

        shutil.rmtree(output_dir, ignore_errors=True)

    def test_no_trace_without_iteration_logging(self):
        """No trace file when iteration logging is disabled."""
        trace_path = _create_trace_file([(100, 10)])
        try:
            config = SimConfig(trace_path=trace_path)
            engine = SimulationEngine(config, enable_iteration_logging=False)
            results = engine.run()
            assert engine.request_trace_logger is None
        finally:
            os.unlink(trace_path)
