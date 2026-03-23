"""Quick 1-hour test to verify streaming mode works correctly."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from config import SimConfig
from core.engine import SimulationEngine

# Use 1-hour trace for quick test
trace_path = "azure_code_1h.csv"

print("="*80)
print("Quick test: 1-hour trace with streaming mode")
print("="*80)

config = SimConfig(
    trace_path=trace_path,
    num_prefill_instances=2,
    num_decode_instances=2,
    enable_switching=False,
    switch_policy="never",
    enable_dynamic_lp=False,
    # Enable streaming with smaller windows for 1h trace
    enable_streaming_loading=True,
    streaming_window_size=300.0,  # 5 minutes
    streaming_lookback=60.0,       # 1 minute
    # Disable expensive features
    enable_monitoring=False,
    enable_periodic_plots=False,
)

print("\nRunning simulation...")
print(f"Streaming enabled: {config.enable_streaming_loading}")
print(f"Window size: {config.streaming_window_size}s")
print(f"Lookback: {config.streaming_lookback}s")

engine = SimulationEngine(config)
results = engine.run()

print("\n" + "="*80)
print("RESULTS")
print("="*80)
print(f"Total requests: {results.total_requests}")
print(f"Simulation time: {results.total_simulation_time:.2f}s ({results.total_simulation_time/60:.2f}min)")

# Check against actual 1h trace
import csv
from datetime import datetime

count = 0
with open(trace_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        count += 1

print(f"\nExpected requests (from trace file): {count}")
if results.total_requests == count:
    print("✓ All requests processed!")
else:
    print(f"⚠️ Missing {count - results.total_requests} requests")
