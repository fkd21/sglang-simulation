#!/usr/bin/env python3
"""Test streaming request loading functionality."""

from __future__ import annotations

from config import SimConfig
from core.engine import SimulationEngine


def test_streaming_mode():
    """Test streaming mode with small trace."""
    print("\n=== Testing Streaming Mode ===\n")

    # Test with streaming enabled
    print("1. Running with streaming ENABLED (window=60s, lookback=10s)...")
    config_streaming = SimConfig(
        trace_path="azure_code_500.csv",
        num_prefill_instances=2,
        num_decode_instances=2,
        enable_streaming_loading=True,
        streaming_window_size=60.0,   # 1 minute windows
        streaming_lookback=10.0,       # 10 second lookback
        chunked_prefill_size=4096,     # Enable chunked prefill to handle more requests
    )
    engine_streaming = SimulationEngine(config_streaming)
    results_streaming = engine_streaming.run()

    print(f"\nStreaming Results:")
    print(f"  Total requests: {results_streaming.total_requests}")
    print(f"  Avg TTFT: {results_streaming.avg_ttft:.3f}s")
    print(f"  Avg E2E: {results_streaming.avg_e2e_latency:.3f}s")
    print(f"  Throughput: {results_streaming.throughput:.2f} req/s")

    # Test with streaming disabled (traditional mode)
    print("\n2. Running with streaming DISABLED (traditional mode)...")
    config_traditional = SimConfig(
        trace_path="azure_code_500.csv",
        num_prefill_instances=2,
        num_decode_instances=2,
        enable_streaming_loading=False,
        chunked_prefill_size=4096,     # Enable chunked prefill to handle more requests
    )
    engine_traditional = SimulationEngine(config_traditional)
    results_traditional = engine_traditional.run()

    print(f"\nTraditional Results:")
    print(f"  Total requests: {results_traditional.total_requests}")
    print(f"  Avg TTFT: {results_traditional.avg_ttft:.3f}s")
    print(f"  Avg E2E: {results_traditional.avg_e2e_latency:.3f}s")
    print(f"  Throughput: {results_traditional.throughput:.2f} req/s")

    # Compare results
    print("\n=== Comparison ===")
    print(f"Request counts match: {results_streaming.total_requests == results_traditional.total_requests}")

    # Allow small differences due to floating point
    ttft_diff = abs(results_streaming.avg_ttft - results_traditional.avg_ttft)
    e2e_diff = abs(results_streaming.avg_e2e_latency - results_traditional.avg_e2e_latency)
    print(f"TTFT difference: {ttft_diff:.6f}s (should be ~0)")
    print(f"E2E difference: {e2e_diff:.6f}s (should be ~0)")

    if ttft_diff < 0.001 and e2e_diff < 0.001:
        print("\n✓ PASS: Streaming mode produces identical results!")
    else:
        print("\n✗ FAIL: Results differ between streaming and traditional mode")
        print(f"  TTFT diff: {ttft_diff:.6f}s, E2E diff: {e2e_diff:.6f}s")

    return results_streaming, results_traditional


if __name__ == "__main__":
    test_streaming_mode()
