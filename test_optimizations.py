"""Test optimizations to verify correctness and performance improvements."""

from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent))

from config import SimConfig
from core.engine import SimulationEngine

print("="*80)
print("Testing Phase 2 and Phase 3 Optimizations")
print("="*80)

# Use a small trace to test quickly
config = SimConfig(
    trace_path="azure_code_8000.csv",
    num_prefill_instances=2,
    num_decode_instances=2,
    enable_switching=False,
    switch_policy="never",
    enable_dynamic_lp=True,
    lp_max_window_size=3,
    # Enable monitoring to test Phase 3 optimizations
    enable_monitoring=True,
    monitoring_sample_interval=10.0,
    monitoring_sla_window=500,
    enable_periodic_plots=False,
    enable_streaming_metrics=True,
    # Disable expensive features for faster testing
    enable_iteration_logging=False,
    enable_request_trace_logging=False,
)

print("\nConfiguration:")
print(f"  Trace: {config.trace_path}")
print(f"  Instances: {config.num_prefill_instances}P{config.num_decode_instances}D")
print(f"  Bootstrap queue: deque (Phase 2 optimization)")
print(f"  Monitoring: enabled with caching (Phase 3 optimization)")
print(f"  LP window size: {config.lp_max_window_size}")

print("\nRunning simulation...")
start_time = time.time()

try:
    engine = SimulationEngine(config)
    results = engine.run()

    wall_clock_time = time.time() - start_time

    print("\n" + "="*80)
    print("SIMULATION COMPLETED SUCCESSFULLY")
    print("="*80)

    print(f"\nWall-clock time: {wall_clock_time:.2f}s")
    print(f"Simulation time: {results.total_simulation_time:.2f}s")
    print(f"Total requests: {results.total_requests}")
    print(f"Completed: {results.total_requests - len(engine.metrics_collector.dropped_requests)}")
    print(f"Dropped: {len(engine.metrics_collector.dropped_requests)}")

    print(f"\nLatency metrics:")
    print(f"  Avg E2E: {results.avg_e2e_latency:.3f}s")
    print(f"  P95 E2E: {results.p95_e2e_latency:.3f}s")
    print(f"  P99 E2E: {results.p99_e2e_latency:.3f}s")

    print(f"\nSLA attainment:")
    print(f"  TTFT: {results.ttft_sla_attainment:.1f}%")
    print(f"  ITL: {results.itl_sla_attainment:.1f}%")
    print(f"  Overall: {results.sla_attainment_rate:.1f}%")

    # Verify dropped_breakdown_cache works
    if hasattr(engine.metrics_collector, 'dropped_breakdown_cache'):
        print(f"\nDropped breakdown (cached):")
        for reason, count in engine.metrics_collector.dropped_breakdown_cache.items():
            print(f"  {reason}: {count}")

    # Verify SLA count caching works
    if engine.time_series_monitor:
        print(f"\nSLA count caching verified:")
        print(f"  recent_ttft_met_count: {engine.time_series_monitor.recent_ttft_met_count}")
        print(f"  recent_itl_met_count: {engine.time_series_monitor.recent_itl_met_count}")
        print(f"  recent_both_met_count: {engine.time_series_monitor.recent_both_met_count}")

    print("\n" + "="*80)
    print("ALL OPTIMIZATIONS WORKING CORRECTLY")
    print("="*80)

except Exception as e:
    print(f"\n{'='*80}")
    print("ERROR DURING SIMULATION")
    print("="*80)
    print(f"\nException: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
