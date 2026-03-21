"""Main entry point for running simulations."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from config import SimConfig
from core.engine import SimulationEngine


def run_baseline_simulation(trace_path: str, num_prefill: int = 1, num_decode: int = 1, enable_streaming: bool = False):
    """Run baseline P/D disaggregation simulation.

    Args:
        trace_path: Path to workload trace CSV
        num_prefill: Number of prefill instances
        num_decode: Number of decode instances
        enable_streaming: Enable streaming workload loading

    Returns:
        Tuple of (results, engine)
    """
    config = SimConfig(
        trace_path=trace_path,
        num_prefill_instances=num_prefill,
        num_decode_instances=num_decode,
        M=0,  # No continuation
        enable_dynamic_lp=False,
        enable_continuation=False,
        enable_switching=False,
        enable_streaming_loading=enable_streaming,
    )

    engine = SimulationEngine(config)
    results = engine.run()

    return results, engine


def print_results(results):
    """Print simulation results.

    Args:
        results: SimulationResults object
    """
    print("\n" + "=" * 80)
    print("SIMULATION RESULTS")
    print("=" * 80)
    print(f"Total Requests: {results.total_requests}")
    print(f"Simulation Time: {results.total_simulation_time:.2f}s")
    print(f"Throughput: {results.throughput:.2f} req/s")
    print()
    print("SLA Attainment (TTFT ≤ 1s, ITL ≤ 100ms):")
    print(f"  Overall SLA: {results.sla_attainment_rate:.1f}%")
    print(f"  TTFT SLA: {results.ttft_sla_attainment:.1f}%")
    print(f"  ITL SLA: {results.itl_sla_attainment:.1f}%")
    print()
    print("Latency Metrics:")
    print(f"  E2E Latency - Avg: {results.avg_e2e_latency:.3f}s, "
          f"P50: {results.p50_e2e_latency:.3f}s, "
          f"P95: {results.p95_e2e_latency:.3f}s, "
          f"P99: {results.p99_e2e_latency:.3f}s")
    print(f"  TTFT - Avg: {results.avg_ttft:.3f}s, "
          f"P50: {results.p50_ttft:.3f}s, "
          f"P99: {results.p99_ttft:.3f}s")
    print(f"  ITL - Avg: {results.avg_itl:.3f}s ({results.avg_itl*1000:.1f}ms), "
          f"P50: {results.p50_itl:.3f}s, "
          f"P99: {results.p99_itl:.3f}s")
    print()
    print("Utilization:")
    print(f"  Prefill: {results.prefill_utilization:.1%}")
    print(f"  Decode: {results.decode_utilization:.1%}")
    print()

    # Print dropped requests statistics (if any)
    if results.num_dropped > 0:
        print("⚠️  Dropped Requests:")
        print(f"  Total Dropped: {results.num_dropped}")
        print(f"  Breakdown:")
        for reason, count in results.dropped_breakdown.items():
            print(f"    {reason}: {count}")
        print()

    print("=" * 80)


def main():
    """Main entry point."""
    # Default trace path
    trace_path = Path(__file__).parent / "azure_code_500.csv"

    if not trace_path.exists():
        print(f"Error: Trace file not found at {trace_path}")
        print("Please ensure azure_code_500.csv is in the project root directory")
        sys.exit(1)

    print("Running baseline P/D disaggregation simulation...")
    print(f"Trace: {trace_path.name}")
    print()

    results, engine = run_baseline_simulation(
        trace_path=str(trace_path),
        num_prefill=1,
        num_decode=7,
        enable_streaming=True  # Enable streaming loading
    )

    print_results(results)

    # Save results to the same run directory as iteration logs
    if engine.iteration_logger:
        output_path = engine.iteration_logger.output_dir / "results.json"
    else:
        result_dir = Path(__file__).parent / "result"
        result_dir.mkdir(parents=True, exist_ok=True)
        output_path = result_dir / "results_baseline.json"
    with open(output_path, 'w') as f:
        json.dump(results.to_dict(), f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
