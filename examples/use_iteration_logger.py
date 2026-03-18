"""Example: How to use iteration logging.

This example shows how to enable and analyze per-iteration statistics.
"""

import json
from pathlib import Path
from config import SimConfig
from core.engine import SimulationEngine


def main():
    """Run simulation with iteration logging enabled."""

    # Configuration
    trace_path = Path("AzureLLMInferenceTrace_code.csv")

    config = SimConfig(
        trace_path=str(trace_path),
        num_prefill_instances=2,
        num_decode_instances=2,
        max_prefill_tokens=16384,
        schedule_policy="fcfs"
    )

    print("="*80)
    print("Running simulation with iteration logging...")
    print("="*80)

    # Create engine with logging enabled
    engine = SimulationEngine(config, enable_iteration_logging=True)

    # Run simulation
    results = engine.run()

    print("\n" + "="*80)
    print("SIMULATION COMPLETED")
    print("="*80)
    print(f"Total requests: {results.total_requests}")
    print(f"SLA attainment: {results.sla_attainment_rate:.1f}%")
    print(f"TTFT: {results.avg_ttft:.3f}s")
    print(f"ITL: {results.avg_itl:.3f}s ({results.avg_itl*1000:.1f}ms)")

    # Analyze logged data
    print("\n" + "="*80)
    print("ANALYZING ITERATION LOGS")
    print("="*80)

    # Read prefill log
    prefill_log = Path("result/2P2D_prefill.jsonl")
    if prefill_log.exists():
        analyze_prefill_log(prefill_log)

    # Read decode log
    decode_log = Path("result/2P2D_decode.jsonl")
    if decode_log.exists():
        analyze_decode_log(decode_log)


def analyze_prefill_log(log_file: Path):
    """Analyze prefill iterations."""
    print(f"\nPrefill Log: {log_file}")

    total_batches = 0
    total_requests = 0
    total_time = 0.0
    total_prefill_tokens = 0

    with open(log_file, 'r') as f:
        for line in f:
            record = json.loads(line)
            total_batches += 1
            total_requests += record['batch_size']
            total_time += record['inference_time']
            total_prefill_tokens += record['total_prefill_tokens']

    print(f"  Total prefill batches: {total_batches}")
    print(f"  Total requests processed: {total_requests}")
    print(f"  Average batch size: {total_requests/total_batches:.2f}")
    print(f"  Total prefill time: {total_time:.2f}s")
    print(f"  Average batch time: {total_time/total_batches:.4f}s")
    print(f"  Total prefill tokens: {total_prefill_tokens:,}")

    # Show first 3 iterations
    print(f"\n  First 3 iterations:")
    with open(log_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            record = json.loads(line)
            print(f"    Iter {record['iteration']}: "
                  f"{record['batch_size']} reqs, "
                  f"{record['total_prefill_tokens']} tokens, "
                  f"{record['inference_time']:.4f}s")


def analyze_decode_log(log_file: Path):
    """Analyze decode iterations."""
    print(f"\nDecode Log: {log_file}")

    total_batches = 0
    total_requests = 0
    total_time = 0.0
    total_tokens_generated = 0

    with open(log_file, 'r') as f:
        for line in f:
            record = json.loads(line)
            total_batches += 1
            total_requests += record['decode_batch_size']
            total_time += record['inference_time']
            # Each decode iteration generates 1 token per request
            total_tokens_generated += record['decode_batch_size']

    print(f"  Total decode batches: {total_batches}")
    print(f"  Total requests processed: {total_requests}")
    print(f"  Average batch size: {total_requests/total_batches:.2f}")
    print(f"  Total decode time: {total_time:.2f}s")
    print(f"  Average batch time: {total_time/total_batches:.4f}s")
    print(f"  Total tokens generated: {total_tokens_generated:,}")

    # Show first 3 iterations
    print(f"\n  First 3 iterations:")
    with open(log_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            record = json.loads(line)
            print(f"    Iter {record['iteration']}: "
                  f"{record['decode_batch_size']} reqs, "
                  f"{record['inference_time']:.4f}s")


if __name__ == "__main__":
    main()
