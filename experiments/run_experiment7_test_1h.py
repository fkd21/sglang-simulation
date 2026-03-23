"""Run only Experiment 7: Switching + Offload Interaction at 4p4d (8 configurations)"""

from __future__ import annotations

import json
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from config import SimConfig
from core.engine import SimulationEngine


# Paths
AZURE_TRACE_1H = Path(__file__).resolve().parent.parent / "azure_code_1h.csv"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _run_one(args: Tuple) -> Dict[str, Any]:
    """Worker function for parallel execution."""
    config, label, extras = args
    print(f"[START] Running simulation: {label}", flush=True)
    print(f"  - Policy: {extras.get('policy', 'N/A')}", flush=True)
    print(f"  - Offload mode: {extras.get('offload_mode', 'N/A')}", flush=True)
    print(f"  - Switching: {config.enable_switching}, Protection: {config.enable_decode_protection}", flush=True)

    engine = SimulationEngine(config)
    print(f"[SIMULATING] {label} - engine created, starting simulation...", flush=True)
    results = engine.run()
    print(f"[DONE] {label} - simulation completed", flush=True)

    d = results.to_dict()
    d["label"] = label
    d["config"] = {
        "trace_path": os.path.basename(config.trace_path),
        "num_prefill_instances": config.num_prefill_instances,
        "num_decode_instances": config.num_decode_instances,
        "enable_dynamic_lp": config.enable_dynamic_lp,
        "enable_decode_protection": config.enable_decode_protection,
        "tpot_sla": config.tpot_sla,
        "slo_target": config.slo_target,
        "lp_max_window_size": config.lp_max_window_size,
        "max_prefill_tokens": config.max_prefill_tokens,
        "enable_switching": config.enable_switching,
        "switch_policy": config.switch_policy,
    }
    d.update(extras)
    return d


def run_parallel(tasks: List[Tuple], max_workers: Optional[int] = None) -> List[Dict]:
    """Run multiple simulations in parallel."""
    if max_workers is None:
        max_workers = min(len(tasks), multiprocessing.cpu_count(), 12)  # Cap at 8 for this experiment
    print(f"\n{'='*60}")
    print(f"Running {len(tasks)} simulations with {max_workers} workers...")
    print(f"{'='*60}\n")

    results = []
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            print(f"[SCHEDULER] Submitting {len(tasks)} tasks to executor...")
            futures = [executor.submit(_run_one, task) for task in tasks]
            print(f"[SCHEDULER] All tasks submitted, waiting for results...\n")

            for i, future in enumerate(futures):
                try:
                    print(f"\n[PROGRESS] Waiting for task {i+1}/{len(tasks)}: {tasks[i][1]}")
                    result = future.result()
                    results.append(result)
                    print(f"[SUCCESS] ✓ Completed {i+1}/{len(tasks)}: {tasks[i][1]}")
                    print(f"[PROGRESS] {len(results)}/{len(tasks)} tasks completed ({100*len(results)/len(tasks):.1f}%)")
                except Exception as e:
                    print(f"[ERROR] ✗ Failed task {i+1}/{len(tasks)} ({tasks[i][1]}): {e}")
                    import traceback
                    traceback.print_exc()
    except Exception as e:
        print(f"[FATAL] ProcessPoolExecutor error: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Parallel execution complete: {len(results)}/{len(tasks)} succeeded")
    print(f"{'='*60}\n")
    return results


def get_timestamp() -> str:
    """Get current timestamp string for filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_experiment(name: str, results: List[Dict]) -> Path:
    """Save experiment results to JSON with timestamp."""
    print(f"\n[SAVE] Saving experiment results...")
    out_dir = RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = get_timestamp()
    path = out_dir / f"{ts}_{name}.json"
    print(f"[SAVE] Writing {len(results)} results to {path}")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[SAVE] ✓ Successfully saved to {path}")
    return path


def _ensure_azure_1h() -> Path:
    """Ensure the 1-hour Azure code trace exists."""
    print(f"[TRACE] Checking for 1h Azure trace at {AZURE_TRACE_1H}...")
    if AZURE_TRACE_1H.exists():
        print(f"[TRACE] ✓ Found trace file")
        return AZURE_TRACE_1H

    raise FileNotFoundError(
        f"1-hour Azure trace not found at {AZURE_TRACE_1H}. "
        "Please run extract_1h_trace.py first to create it."
    )


def experiment_switching_offload_4p4d(max_workers: Optional[int] = None) -> List[Dict]:
    """Test 4 switching policies x 3 offload modes at 4p4d with 1h Azure code requests (TEST).

    Offload modes:
    - no_offload: baseline without offload
    - offload_no_protection: offload enabled, no decode protection
    - offload_with_protection: offload enabled, with decode TPOT protection
    """
    print("\n" + "="*80)
    print("=== Experiment 7 TEST (1h trace): Switching + Offload at 4p4d (12 configurations) ===")
    print("="*80 + "\n")
    trace = str(_ensure_azure_1h())

    policies = [
        ("none", False, "never"),
        ("alpha", True, "alpha"),
        ("v1", True, "v1"),
        ("throughput", True, "throughput"),
    ]
    print(f"[CONFIG] Testing {len(policies)} policies: {[p[0] for p in policies]}")
    print(f"[CONFIG] Each policy will be tested with 3 offload modes")
    print(f"[CONFIG] Total configurations: {len(policies) * 3} = 12\n")

    tasks = []

    for policy_name, enable_sw, sw_policy in policies:
        print(f"\n[SETUP] Configuring tasks for policy: {policy_name}")
        # 1. Without offload (baseline)
        config = SimConfig(
            trace_path=trace,
            num_prefill_instances=4,
            num_decode_instances=4,
            enable_switching=enable_sw,
            switch_policy=sw_policy,
            enable_dynamic_lp=False,
            # Enable streaming loading to avoid OOM
            enable_streaming_loading=True,
            streaming_window_size=300.0,  # 5 minutes
            streaming_lookback=60.0,       # 1 minute safety buffer
            # Enable monitoring and periodic plots
            enable_monitoring=True,
            monitoring_plot_interval_minutes=60.0,
        )
        tasks.append((
            config,
            f"{policy_name}_no_offload",
            {"policy": policy_name, "offload_mode": "no_offload"},
        ))
        print(f"  [1/3] Added: {policy_name}_no_offload")

        # 2. With offload, NO decode protection
        config = SimConfig(
            trace_path=trace,
            num_prefill_instances=4,
            num_decode_instances=4,
            enable_switching=enable_sw,
            switch_policy=sw_policy,
            enable_dynamic_lp=True,
            enable_decode_protection=False,  # Disable decode protection
            slo_target=1.0,
            lp_max_window_size=5,
            # Enable streaming loading to avoid OOM
            enable_streaming_loading=True,
            streaming_window_size=300.0,  # 5 minutes
            streaming_lookback=60.0,       # 1 minute safety buffer
            # Enable monitoring and periodic plots
            enable_monitoring=True,
            monitoring_plot_interval_minutes=60.0,
        )
        tasks.append((
            config,
            f"{policy_name}_offload_no_protection",
            {"policy": policy_name, "offload_mode": "offload_no_protection"},
        ))
        print(f"  [2/3] Added: {policy_name}_offload_no_protection")

        # 3. With offload, WITH decode protection
        config = SimConfig(
            trace_path=trace,
            num_prefill_instances=4,
            num_decode_instances=4,
            enable_switching=enable_sw,
            switch_policy=sw_policy,
            enable_dynamic_lp=True,
            enable_decode_protection=True,  # Enable decode protection
            tpot_sla=0.1,  # 100ms TPOT threshold
            slo_target=1.0,
            lp_max_window_size=5,
            # Enable streaming loading to avoid OOM
            enable_streaming_loading=True,
            streaming_window_size=300.0,  # 5 minutes
            streaming_lookback=60.0,       # 1 minute safety buffer
            # Enable monitoring and periodic plots
            enable_monitoring=True,
            monitoring_plot_interval_minutes=60.0,
        )
        tasks.append((
            config,
            f"{policy_name}_offload_with_protection",
            {"policy": policy_name, "offload_mode": "offload_with_protection"},
        ))
        print(f"  [3/3] Added: {policy_name}_offload_with_protection")

    print(f"\n[SETUP] Total tasks configured: {len(tasks)}")
    print(f"[SETUP] Task list:")
    for i, (_, label, _) in enumerate(tasks, 1):
        print(f"  {i:2d}. {label}")

    return run_parallel(tasks, max_workers=max_workers)


def main():
    import sys
    print("\n" + "="*80)
    print("EXPERIMENT 7 TEST (1H TRACE): SWITCHING + OFFLOAD INTERACTION")
    print("="*80)
    print(f"[INIT] Results will be saved to {RESULTS_DIR}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Check if test mode (single worker)
    test_mode = "--test" in sys.argv
    workers = 1 if test_mode else None
    if test_mode:
        print("\n" + "!"*80)
        print("!!! TEST MODE: Using 1 worker !!!")
        print("!"*80 + "\n")

    print(f"[INIT] Starting experiment 7 (1h trace)...")
    r7 = experiment_switching_offload_4p4d(max_workers=workers)
    save_experiment("exp7_test_1h_switching_offload_4p4d", r7)

    print("\n" + "="*80)
    print("✓ ALL EXPERIMENTS COMPLETE!")
    print(f"✓ {len(r7)} configurations tested")
    print(f"✓ Results saved to {RESULTS_DIR}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
