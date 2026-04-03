"""Quick test to verify streaming mode runs to completion."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from config import SimConfig
from core.engine import SimulationEngine

# Test with a small trace first
trace_path = "azure_code_24h_25.csv"

print("="*80)
print("Testing streaming mode with idle termination fix")
print("="*80)

config = SimConfig(
    trace_path=trace_path,
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

print("\nRunning simulation...")
engine = SimulationEngine(config)
results = engine.run()

print("\n" + "="*80)
print("RESULTS")
print("="*80)
print(f"Total requests: {results.total_requests}")
print(f"Total simulation time: {results.total_simulation_time:.2f}s ({results.total_simulation_time/3600:.2f}h)")
print(f"Expected ~622,584 requests in 24h trace")
print(f"Expected ~86,400s simulation time")

if results.total_requests < 600000:
    print("\n⚠️  WARNING: Simulation terminated early!")
    print(f"   Only processed {results.total_requests} requests")
    print(f"   Only simulated {results.total_simulation_time/3600:.2f} hours")
else:
    print("\n✓ SUCCESS: Full trace processed!")
