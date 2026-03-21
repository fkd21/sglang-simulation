"""Debug script to analyze role switching behavior."""

import json
import sys
from pathlib import Path

# Load the result file
result_file = Path("experiments/results/exp7_switching_offload_1p7d.json")
with open(result_file) as f:
    data = json.load(f)

# Find the alpha_offload result
for item in data:
    if item.get("label") == "alpha_offload":
        print("=== Alpha + Offload Configuration ===")
        print(f"Total requests: {item['total_requests']}")
        print(f"Simulation time: {item['total_simulation_time']:.2f}s")
        print(f"Throughput: {item['throughput']:.2f} req/s")
        print(f"Token throughput: {item['token_throughput']:.2f} tokens/s")
        print(f"\nPrefill utilization: {item['prefill_utilization']:.4f}")
        print(f"Decode utilization: {item['decode_utilization']:.4f}")
        print(f"\nNum switches: {item['num_switches']}")

        print("\n=== Role Switches ===")
        for i, switch in enumerate(item['role_switches']):
            print(f"\nSwitch {i+1}:")
            print(f"  Time: {switch['time']:.2f}s")
            print(f"  Instance: {switch['instance_id']}")
            print(f"  {switch['from_role']} -> {switch['to_role']}")
            print(f"  Drain time: {switch['drain_time']:.2f}s")

        # Calculate what instance configuration we should have after all switches
        initial_prefill = 1
        initial_decode = 7

        prefill_count = initial_prefill
        decode_count = initial_decode

        print("\n=== Instance Count After Each Switch ===")
        print(f"Initial: {initial_prefill}P{initial_decode}D")

        for i, switch in enumerate(item['role_switches']):
            if switch['from_role'] == 'DECODE' and switch['to_role'] == 'PREFILL':
                decode_count -= 1
                prefill_count += 1
            elif switch['from_role'] == 'PREFILL' and switch['to_role'] == 'DECODE':
                prefill_count -= 1
                decode_count += 1
            print(f"After switch {i+1}: {prefill_count}P{decode_count}D")

        print(f"\nFinal configuration: {prefill_count}P{decode_count}D")
        print(f"Last switch time: {item['role_switches'][-1]['time']:.2f}s")
        print(f"Remaining simulation time: {item['total_simulation_time'] - item['role_switches'][-1]['time']:.2f}s")

        # Calculate expected throughput improvement
        # With 7P1D vs 1P7D, we should see much higher throughput
        # because prefill is usually the bottleneck
        print("\n=== Expected Behavior ===")
        print("With 7 prefill instances (vs 1 initially), we should see:")
        print("  - Much higher prefill throughput")
        print("  - Much lower TTFT (time to first token)")
        print("  - Overall higher request throughput")

        print("\n=== Actual Metrics ===")
        print(f"Avg TTFT: {item['avg_ttft']:.2f}s")
        print(f"P50 TTFT: {item['p50_ttft']:.2f}s")
        print(f"P99 TTFT: {item['p99_ttft']:.2f}s")

        break

# Compare with baseline
print("\n\n=== Comparison with Baseline (no switching) ===")
for item in data:
    if item.get("label") == "none_no_offload":
        print(f"Baseline throughput: {item['throughput']:.2f} req/s")
        print(f"Baseline TTFT (avg): {item['avg_ttft']:.2f}s")
        print(f"Baseline TTFT (p50): {item['p50_ttft']:.2f}s")
        break

for item in data:
    if item.get("label") == "alpha_offload":
        print(f"\nAlpha+offload throughput: {item['throughput']:.2f} req/s")
        print(f"Alpha+offload TTFT (avg): {item['avg_ttft']:.2f}s")
        print(f"Alpha+offload TTFT (p50): {item['p50_ttft']:.2f}s")

        baseline_tput = 10.04  # from earlier
        alpha_tput = item['throughput']
        print(f"\nThroughput change: {((alpha_tput/baseline_tput - 1) * 100):.1f}%")
        print("⚠️ PROBLEM: Throughput should INCREASE with 7P1D, not decrease!")
        break
