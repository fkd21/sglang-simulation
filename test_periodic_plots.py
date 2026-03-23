#!/usr/bin/env python3
"""Test periodic plot generation in multiprocessing environment."""

import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics.time_series_monitor import TimeSeriesMonitor


def test_periodic_plotting():
    """Test function that runs in a child process."""
    import os
    print(f"\n[TEST] Starting test in PID {os.getpid()}")

    # Create monitor with short interval for testing (1 minute)
    monitor = TimeSeriesMonitor(
        sample_interval=1.0,
        output_dir="result/test_periodic",
        run_name="test_run",
        enable_periodic_plots=True,
        plot_interval_minutes=1.0,  # 1 minute for testing
    )

    # Start periodic plotting
    monitor.start_periodic_plotting()

    # Simulate some work for 3 minutes
    print(f"[TEST] Simulating work for 3 minutes...")
    start_time = time.time()

    # Add dummy samples
    for i in range(180):  # 3 minutes worth of samples
        sim_time = float(i)
        engine_state = {
            'all_instances': [],
            'metrics_collector': None,
        }
        monitor.maybe_sample(sim_time, engine_state)
        time.sleep(1)  # 1 second between samples

        if i % 30 == 0:
            print(f"[TEST] Progress: {i}/180 samples collected ({time.time() - start_time:.1f}s elapsed)")

    print(f"[TEST] Finalizing...")
    monitor.finalize()
    print(f"[TEST] Test complete!")

    return "success"


if __name__ == "__main__":
    print("="*80)
    print("Testing periodic plot generation in child process")
    print("="*80)

    # Test in child process only (faster)
    print("\n[CHILD] Testing in child process via ProcessPoolExecutor...")

    with ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(test_periodic_plotting)
        result = future.result()
        print(f"[CHILD] Child process test result: {result}")

    print("\n" + "="*80)
    print("Test complete!")
    print("="*80)
