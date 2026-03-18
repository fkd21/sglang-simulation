"""Integration tests for mechanisms in the SimulationEngine.

Uses small synthetic traces to verify end-to-end behavior of:
- Chunked prefill
- Dynamic LP offloading
- Decode continuation (M)
"""

import json
import os
import tempfile

import pytest

from config import SimConfig
from core.engine import SimulationEngine


def _create_trace_file(requests):
    """Create a temp JSONL trace file from list of (input_len, output_len) tuples.

    Returns the path to the temp file. Caller should clean up.
    """
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w") as f:
        for input_len, output_len in requests:
            f.write(json.dumps({"input_len": input_len, "output_len": output_len}) + "\n")
    return path


class TestBaselineEngine:
    """Test baseline engine without any mechanisms."""

    def test_single_request(self):
        """Single request completes through full pipeline."""
        trace_path = _create_trace_file([(100, 10)])
        try:
            config = SimConfig(trace_path=trace_path)
            engine = SimulationEngine(config)
            results = engine.run()

            assert results.total_requests == 1
            assert results.avg_e2e_latency > 0
            assert results.avg_ttft > 0
        finally:
            os.unlink(trace_path)

    def test_multiple_requests(self):
        """Multiple requests all complete."""
        reqs = [(50, 5), (100, 10), (200, 3)]
        trace_path = _create_trace_file(reqs)
        try:
            config = SimConfig(trace_path=trace_path)
            engine = SimulationEngine(config)
            results = engine.run()

            assert results.total_requests == 3
            assert results.throughput > 0
        finally:
            os.unlink(trace_path)

    def test_multiple_instances(self):
        """Works with multiple prefill/decode instances."""
        reqs = [(100, 10)] * 5
        trace_path = _create_trace_file(reqs)
        try:
            config = SimConfig(
                trace_path=trace_path,
                num_prefill_instances=2,
                num_decode_instances=2,
            )
            engine = SimulationEngine(config)
            results = engine.run()

            assert results.total_requests == 5
        finally:
            os.unlink(trace_path)


class TestChunkedPrefillEngine:
    """Integration tests for chunked prefill in the engine."""

    def test_chunked_prefill_completes_all(self):
        """All requests complete with chunked prefill enabled."""
        reqs = [(500, 5), (200, 10), (1000, 3)]
        trace_path = _create_trace_file(reqs)
        try:
            config = SimConfig(
                trace_path=trace_path,
                chunked_prefill_size=256,
            )
            engine = SimulationEngine(config)
            results = engine.run()

            assert results.total_requests == 3
        finally:
            os.unlink(trace_path)

    def test_chunked_vs_unchunked_results_similar(self):
        """Chunked and unchunked should produce similar total requests."""
        reqs = [(100, 5)] * 10
        trace_path = _create_trace_file(reqs)
        try:
            # Without chunking
            config1 = SimConfig(trace_path=trace_path)
            results1 = SimulationEngine(config1).run()

            # With chunking
            config2 = SimConfig(trace_path=trace_path, chunked_prefill_size=64)
            results2 = SimulationEngine(config2).run()

            assert results1.total_requests == results2.total_requests
            # Chunked should take slightly longer due to overhead
            assert results2.total_simulation_time >= results1.total_simulation_time * 0.9
        finally:
            os.unlink(trace_path)

    def test_small_chunk_size_still_completes(self):
        """Even very small chunk sizes complete all requests."""
        reqs = [(500, 3), (1000, 2)]
        trace_path = _create_trace_file(reqs)
        try:
            config = SimConfig(
                trace_path=trace_path,
                chunked_prefill_size=128,
            )
            engine = SimulationEngine(config)
            results = engine.run()

            assert results.total_requests == 2
        finally:
            os.unlink(trace_path)

    def test_chunk_size_larger_than_request(self):
        """Chunk size > request size means no chunking occurs."""
        reqs = [(50, 5)]
        trace_path = _create_trace_file(reqs)
        try:
            config = SimConfig(
                trace_path=trace_path,
                chunked_prefill_size=8192,
            )
            engine = SimulationEngine(config)
            results = engine.run()

            assert results.total_requests == 1
        finally:
            os.unlink(trace_path)


class TestDynamicLPOffloadEngine:
    """Integration tests for dynamic LP-based partial offloading."""

    def test_offload_completes_all(self):
        """All requests complete with dynamic LP offloading enabled."""
        reqs = [(200, 10), (300, 5), (100, 8)]
        trace_path = _create_trace_file(reqs)
        try:
            config = SimConfig(
                trace_path=trace_path,
                enable_dynamic_lp=True,
                slo_target=2.0,
            )
            engine = SimulationEngine(config)
            results = engine.run()

            assert results.total_requests == 3
        finally:
            os.unlink(trace_path)

    def test_offload_disabled_no_offloads(self):
        """With offloading disabled, no offloads happen."""
        reqs = [(200, 10)] * 3
        trace_path = _create_trace_file(reqs)
        try:
            config = SimConfig(
                trace_path=trace_path,
                enable_dynamic_lp=False,
            )
            engine = SimulationEngine(config)
            results = engine.run()

            assert results.total_requests == 3
            assert results.num_offloaded == 0
        finally:
            os.unlink(trace_path)

    def test_offload_generous_slo_no_offloads(self):
        """With very generous SLO, LP should find beta=0 (no offloading needed)."""
        reqs = [(200, 10)]
        trace_path = _create_trace_file(reqs)
        try:
            config = SimConfig(
                trace_path=trace_path,
                enable_dynamic_lp=True,
                slo_target=100.0,  # Very generous
            )
            engine = SimulationEngine(config)
            results = engine.run()

            assert results.total_requests == 1
            assert results.num_offloaded == 0
        finally:
            os.unlink(trace_path)

    def test_offload_various_slo_targets(self):
        """Different SLO targets all produce valid results."""
        reqs = [(300, 5)] * 3
        trace_path = _create_trace_file(reqs)
        try:
            for slo in [0.1, 0.5, 1.0, 2.0, 5.0]:
                config = SimConfig(
                    trace_path=trace_path,
                    enable_dynamic_lp=True,
                    slo_target=slo,
                )
                results = SimulationEngine(config).run()
                assert results.total_requests == 3, f"Failed for slo={slo}"
        finally:
            os.unlink(trace_path)


class TestDecodeContinuationEngine:
    """Integration tests for decode continuation (M parameter)."""

    def test_continuation_completes_all(self):
        """All requests complete with continuation enabled."""
        reqs = [(200, 20), (100, 10)]
        trace_path = _create_trace_file(reqs)
        try:
            config = SimConfig(
                trace_path=trace_path,
                enable_continuation=True,
                M=5,
            )
            engine = SimulationEngine(config)
            results = engine.run()

            assert results.total_requests == 2
        finally:
            os.unlink(trace_path)

    def test_continuation_tracks_mechanism_usage(self):
        """Continued requests are counted in metrics."""
        reqs = [(100, 20)] * 4
        trace_path = _create_trace_file(reqs)
        try:
            config = SimConfig(
                trace_path=trace_path,
                enable_continuation=True,
                M=5,
            )
            engine = SimulationEngine(config)
            results = engine.run()

            assert results.total_requests == 4
            assert results.num_continued == 4
        finally:
            os.unlink(trace_path)

    def test_continuation_disabled_no_continuations(self):
        """With continuation disabled, no continuations happen."""
        reqs = [(100, 20)] * 3
        trace_path = _create_trace_file(reqs)
        try:
            config = SimConfig(
                trace_path=trace_path,
                enable_continuation=False,
                M=5,
            )
            engine = SimulationEngine(config)
            results = engine.run()

            assert results.total_requests == 3
            assert results.num_continued == 0
        finally:
            os.unlink(trace_path)

    def test_continuation_M_larger_than_output(self):
        """M > output_tokens means entire request completes on prefill."""
        reqs = [(100, 3)]  # Only 3 tokens to generate
        trace_path = _create_trace_file(reqs)
        try:
            config = SimConfig(
                trace_path=trace_path,
                enable_continuation=True,
                M=100,  # Much larger than output
            )
            engine = SimulationEngine(config)
            results = engine.run()

            assert results.total_requests == 1
            assert results.num_continued == 1
        finally:
            os.unlink(trace_path)

    def test_continuation_various_M(self):
        """Different M values all produce valid results."""
        reqs = [(100, 20)] * 3
        trace_path = _create_trace_file(reqs)
        try:
            for M in [1, 5, 10, 15, 19, 20, 50]:
                config = SimConfig(
                    trace_path=trace_path,
                    enable_continuation=True,
                    M=M,
                )
                results = SimulationEngine(config).run()
                assert results.total_requests == 3, f"Failed for M={M}"
        finally:
            os.unlink(trace_path)


class TestCombinedMechanisms:
    """Test multiple mechanisms enabled simultaneously."""

    def test_offload_plus_continuation(self):
        """Both dynamic LP offloading and continuation work together."""
        reqs = [(200, 15)] * 3
        trace_path = _create_trace_file(reqs)
        try:
            config = SimConfig(
                trace_path=trace_path,
                enable_dynamic_lp=True,
                slo_target=0.5,
                enable_continuation=True,
                M=5,
            )
            engine = SimulationEngine(config)
            results = engine.run()

            assert results.total_requests == 3
            assert results.num_continued == 3
        finally:
            os.unlink(trace_path)

    def test_chunking_plus_continuation(self):
        """Chunked prefill and continuation work together."""
        reqs = [(500, 10)] * 3
        trace_path = _create_trace_file(reqs)
        try:
            config = SimConfig(
                trace_path=trace_path,
                chunked_prefill_size=256,
                enable_continuation=True,
                M=5,
            )
            engine = SimulationEngine(config)
            results = engine.run()

            assert results.total_requests == 3
            assert results.num_continued == 3
        finally:
            os.unlink(trace_path)

    def test_chunking_plus_offload(self):
        """Chunked prefill and dynamic LP offloading work together."""
        reqs = [(500, 10)] * 3
        trace_path = _create_trace_file(reqs)
        try:
            config = SimConfig(
                trace_path=trace_path,
                chunked_prefill_size=256,
                enable_dynamic_lp=True,
                slo_target=1.0,
            )
            engine = SimulationEngine(config)
            results = engine.run()

            assert results.total_requests == 3
        finally:
            os.unlink(trace_path)

    def test_all_mechanisms_together(self):
        """All mechanisms enabled simultaneously."""
        reqs = [(300, 20)] * 5
        trace_path = _create_trace_file(reqs)
        try:
            config = SimConfig(
                trace_path=trace_path,
                chunked_prefill_size=128,
                enable_dynamic_lp=True,
                slo_target=1.0,
                enable_continuation=True,
                M=5,
            )
            engine = SimulationEngine(config)
            results = engine.run()

            assert results.total_requests == 5
        finally:
            os.unlink(trace_path)


class TestEngineLatencyMetrics:
    """Test that latency metrics are reasonable."""

    def test_ttft_positive(self):
        """TTFT should be positive for all completed requests."""
        reqs = [(100, 10)] * 5
        trace_path = _create_trace_file(reqs)
        try:
            config = SimConfig(trace_path=trace_path)
            engine = SimulationEngine(config)
            results = engine.run()

            assert results.avg_ttft > 0
            assert results.p50_ttft > 0
        finally:
            os.unlink(trace_path)

    def test_e2e_latency_greater_than_ttft(self):
        """E2E latency should be >= TTFT (prefill + decode + transfer >= prefill)."""
        reqs = [(100, 10)] * 5
        trace_path = _create_trace_file(reqs)
        try:
            config = SimConfig(trace_path=trace_path)
            engine = SimulationEngine(config)
            results = engine.run()

            assert results.avg_e2e_latency >= results.avg_ttft
        finally:
            os.unlink(trace_path)

    def test_utilization_between_zero_and_one(self):
        """Instance utilization should be in [0, 1]."""
        reqs = [(100, 10)] * 10
        trace_path = _create_trace_file(reqs)
        try:
            config = SimConfig(trace_path=trace_path)
            engine = SimulationEngine(config)
            results = engine.run()

            assert 0 <= results.prefill_utilization <= 1.0
            assert 0 <= results.decode_utilization <= 1.0
        finally:
            os.unlink(trace_path)
