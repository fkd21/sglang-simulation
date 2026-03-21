#!/usr/bin/env python3
"""Test parallel LP solver execution."""

import time
from config import SimConfig
from core.engine import SimulationEngine

print("=" * 80)
print("Testing Parallel LP Solver")
print("=" * 80)

# Test configurations
configs = [
    {
        "name": "Sequential (baseline)",
        "enable_parallel": False,
        "num_prefill": 4,
    },
    {
        "name": "Parallel (4 workers)",
        "enable_parallel": True,
        "num_prefill": 4,
    },
]

results = []

for cfg in configs:
    print(f"\n{'='*80}")
    print(f"Configuration: {cfg['name']}")
    print(f"{'='*80}")

    config = SimConfig(
        trace_path="azure_code_500.csv",
        num_prefill_instances=cfg['num_prefill'],
        num_decode_instances=4,
        max_prefill_tokens=8192,
        enable_dynamic_lp=True,  # Enable LP solver
        enable_parallel_lp_solver=cfg['enable_parallel'],
        lp_solver_max_workers=-1,  # Auto
        enable_monitoring=False,  # Disable for speed
        enable_iteration_logging=False,
    )

    engine = SimulationEngine(config)

    start_time = time.time()
    sim_results = engine.run()
    elapsed = time.time() - start_time

    results.append({
        'config': cfg['name'],
        'wall_clock': elapsed,
        'sim_time': sim_results.total_simulation_time,
        'requests': sim_results.total_requests,
        'avg_ttft': sim_results.avg_ttft,
        'p99_ttft': sim_results.p99_ttft,
    })

    print(f"\nWall-clock time: {elapsed:.2f}s")
    print(f"Simulation time: {sim_results.total_simulation_time:.2f}s")
    print(f"Requests: {sim_results.total_requests}")
    print(f"Avg TTFT: {sim_results.avg_ttft:.4f}s")
    print(f"P99 TTFT: {sim_results.p99_ttft:.4f}s")

print("\n" + "=" * 80)
print("COMPARISON RESULTS")
print("=" * 80)

baseline = results[0]
parallel = results[1]

speedup = baseline['wall_clock'] / parallel['wall_clock']
ttft_diff = abs(parallel['avg_ttft'] - baseline['avg_ttft'])
ttft_diff_pct = (ttft_diff / baseline['avg_ttft']) * 100 if baseline['avg_ttft'] > 0 else 0

print(f"\nWall-clock speedup: {speedup:.2f}x")
print(f"Sequential: {baseline['wall_clock']:.2f}s")
print(f"Parallel:   {parallel['wall_clock']:.2f}s")

print(f"\nSimulation results (should be identical):")
print(f"Sequential TTFT: {baseline['avg_ttft']:.4f}s")
print(f"Parallel TTFT:   {parallel['avg_ttft']:.4f}s")
print(f"Difference: {ttft_diff:.6f}s ({ttft_diff_pct:.3f}%)")

if ttft_diff_pct < 0.01:  # Less than 0.01% difference
    print("\n✓ Results are numerically identical (parallelization correct!)")
else:
    print(f"\n⚠ Warning: Results differ by {ttft_diff_pct:.3f}%")

if speedup > 1.1:
    print(f"\n✓ Parallel LP solver is {speedup:.2f}x faster!")
elif speedup > 0.9:
    print(f"\n○ Similar performance ({speedup:.2f}x) - overhead may dominate for small traces")
else:
    print(f"\n✗ Parallel version slower ({speedup:.2f}x) - unexpected!")
