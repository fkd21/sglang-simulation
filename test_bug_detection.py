"""Quick test to verify Bug #1 detection with diagnostic logging."""

from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import SimConfig
from core.engine import SimulationEngine

def main():
    print("\n" + "="*80)
    print("BUG DETECTION TEST")
    print("Testing for orphaned requests in prealloc_reserved (Bug #1)")
    print("="*80 + "\n")

    # Use same config as problematic run
    config = SimConfig(
        trace_path="azure_code_1h.csv",
        num_prefill_instances=4,
        num_decode_instances=4,
        enable_switching=True,
        switch_policy="alpha",
        enable_dynamic_lp=True,
        enable_decode_protection=False,
        slo_target=1.0,
        lp_max_window_size=5,
        enable_streaming_loading=True,
        streaming_window_size=300.0,
        streaming_lookback=60.0,
        enable_monitoring=True,
        monitoring_plot_interval_minutes=60.0,
        enable_iteration_logging=False,  # Keep output clean
    )

    print("[CONFIG] Simulation configuration:")
    print(f"  Instances: {config.num_prefill_instances}P + {config.num_decode_instances}D")
    print(f"  Switching: {config.enable_switching} (policy: {config.switch_policy})")
    print(f"  Dynamic LP: {config.enable_dynamic_lp}")
    print(f"  Trace: {config.trace_path}\n")

    print("[START] Creating simulation engine...")
    engine = SimulationEngine(config)

    print("[RUN] Starting simulation with diagnostic logging enabled...")
    print("      Watch for [BUG #1 DETECTED] and [BUG #3 DETECTED] messages\n")

    results = engine.run()

    print("\n" + "="*80)
    print("SIMULATION COMPLETE")
    print("="*80)
    print(f"\n[RESULTS] Summary:")
    print(f"  Completed requests: {results.num_completed}")
    print(f"  Dropped requests: {results.num_dropped}")
    print(f"  Throughput: {results.throughput:.1f} tokens/s")
    print(f"  TTFT SLA: {results.ttft_sla_attainment:.1f}%")
    print(f"  Overall SLA: {results.overall_sla_attainment:.1f}%")
    print(f"\n[DIAG] Check output above for bug detection messages:")
    print(f"  - [BUG #1 DETECTED]: Orphaned requests in prealloc_reserved")
    print(f"  - [BUG #3 DETECTED]: Counter drift")
    print(f"  - Instance selection logs around t=1320s")
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()
