"""Test a single configuration to verify streaming works correctly."""

from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).parent))

from config import SimConfig
from core.engine import SimulationEngine

print("="*80)
print("Testing single configuration with 24h trace")
print("="*80)

config = SimConfig(
    trace_path="azure_code_24h_25.csv",
    num_prefill_instances=4,
    num_decode_instances=4,
    enable_switching=False,
    switch_policy="never",
    enable_dynamic_lp=False,
    # Enable streaming
    enable_streaming_loading=True,
    streaming_window_size=300.0,  # 5 minutes
    streaming_lookback=60.0,       # 1 minute
    # Disable expensive features for faster testing
    enable_monitoring=False,
    enable_periodic_plots=False,
)

print("\nConfiguration:")
print(f"  Streaming: {config.enable_streaming_loading}")
print(f"  Window size: {config.streaming_window_size}s")
print(f"  Lookback: {config.streaming_lookback}s")
print(f"  4P4D (no switching, no offload)")

print("\nRunning simulation...")
engine = SimulationEngine(config)
results = engine.run()

print("\n" + "="*80)
print("RESULTS")
print("="*80)
print(f"Total requests: {results.total_requests:,}")
print(f"Simulation time: {results.total_simulation_time:.2f}s ({results.total_simulation_time/3600:.2f}h)")
print(f"Throughput: {results.throughput:.2f} req/s")

# Expected values
expected_requests = 622584
expected_time = 86400

print(f"\nExpected:")
print(f"  Requests: {expected_requests:,}")
print(f"  Time: {expected_time}s (24.00h)")

print(f"\nComparison:")
print(f"  Requests: {results.total_requests / expected_requests * 100:.1f}%")
print(f"  Time: {results.total_simulation_time / expected_time * 100:.1f}%")

if results.total_requests >= expected_requests * 0.99:
    print("\n✓ SUCCESS: Full trace processed!")
else:
    print(f"\n⚠️  WARNING: Only processed {results.total_requests:,} / {expected_requests:,} requests")
    print(f"   Missing: {expected_requests - results.total_requests:,} requests")
