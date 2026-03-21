#!/usr/bin/env python3
"""Test periodic plot generation with wall-clock timer."""

import time
from config import SimConfig
from core.engine import SimulationEngine

# Create minimal config for testing
config = SimConfig(
    trace_path="azure_code_500.csv",
    num_prefill_instances=2,
    num_decode_instances=2,
    max_prefill_tokens=8192,
    enable_monitoring=True,
    enable_periodic_plots=True,
    monitoring_plot_interval_minutes=0.25,  # 15 seconds for testing
    monitoring_sample_interval=1.0,
)

print("=" * 80)
print("Testing Periodic Plot Generation")
print("=" * 80)
print(f"Plot interval: {config.monitoring_plot_interval_minutes} minutes")
print(f"This test will run a short simulation and generate plots every 30 seconds")
print("Watch the plots/ directory for updated files")
print("=" * 80)

# Create and run engine
engine = SimulationEngine(config)

print("\nStarting simulation...")
start_time = time.time()

try:
    results = engine.run()

    elapsed = time.time() - start_time
    print(f"\nSimulation completed in {elapsed:.1f} seconds (wall-clock)")
    print(f"Simulation time: {results.total_simulation_time:.1f}s")
    print(f"Requests completed: {results.total_requests}")
    print(f"Average E2E latency: {results.avg_e2e_latency:.2f}s")
    print(f"Average TTFT: {results.avg_ttft:.2f}s")

    # Check if plots were generated
    import os
    plots_dir = engine.time_series_monitor.output_dir if engine.time_series_monitor else None
    if plots_dir and os.path.exists(plots_dir):
        plot_files = [f for f in os.listdir(plots_dir) if f.endswith('.png')]
        print(f"\nPlot files generated ({len(plot_files)}):")
        for f in sorted(plot_files):
            file_path = os.path.join(plots_dir, f)
            mtime = os.path.getmtime(file_path)
            print(f"  - {f} (modified: {time.strftime('%H:%M:%S', time.localtime(mtime))})")

    print("\n✓ Test completed successfully!")

except KeyboardInterrupt:
    print("\n\nInterrupted by user")
    if engine.time_series_monitor:
        engine.time_series_monitor.finalize()
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    if engine.time_series_monitor:
        engine.time_series_monitor.finalize()
