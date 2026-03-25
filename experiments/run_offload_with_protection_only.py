"""Run Offload with Protection Budget Sweep: Test budget_scaling_factor from 1.0 to 2.0"""

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
    print(f"  - Budget scaling factor: {extras.get('budget_scaling_factor', 'N/A')}", flush=True)
    print(f"  - Switching: {config.enable_switching}, Protection: {config.enable_decode_protection}", flush=True)
    print(f"  - TPOT SLA: {config.tpot_sla}s", flush=True)

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
        "budget_scaling_factor": config.budget_scaling_factor,
        "slo_target": config.slo_target,
        "lp_max_window_size": config.lp_max_window_size,
        "max_prefill_tokens": config.max_prefill_tokens,
        "enable_switching": config.enable_switching,
        "switch_policy": config.switch_policy,
        "alpha_allow_decode_to_prefill": config.alpha_allow_decode_to_prefill,
        "alpha_allow_prefill_to_decode": config.alpha_allow_prefill_to_decode,
    }
    d.update(extras)
    return d


def run_parallel(tasks: List[Tuple], max_workers: Optional[int] = None) -> List[Dict]:
    """Run multiple simulations in parallel."""
    if max_workers is None:
        max_workers = min(len(tasks), multiprocessing.cpu_count(), 12)
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


def experiment_offload_with_protection_budget_sweep_4p4d(max_workers: Optional[int] = None) -> List[Dict]:
    """Sweep budget_scaling_factor from 1.0 to 2.0 for offload with protection.

    Tests alpha policy with decode→prefill only switching, offload WITH decode protection enabled.
    Sweeps budget_scaling_factor in steps of 0.2 to understand impact on offload aggressiveness.

    Configuration:
    - 4 prefill instances, 4 decode instances
    - Alpha policy with decode→prefill only (prefill→decode disabled)
    - Dynamic LP enabled with decode protection
    - TPOT SLA: 100ms (0.1s)
    - Budget scaling factor: 1.0, 1.2, 1.4, 1.6, 1.8, 2.0

    Returns:
        List of result dictionaries for each configuration
    """
    print("\n" + "="*80)
    print("=== Budget Scaling Factor Sweep (1h trace): Alpha + Offload with Protection ===")
    print("=== Sweeping budget_scaling_factor from 1.0 to 2.0 in steps of 0.2 ===")
    print("="*80 + "\n")
    trace = str(_ensure_azure_1h())

    print(f"[CONFIG] Policy: alpha (decode→prefill only)")
    print(f"[CONFIG] Topology: 4 prefill instances, 4 decode instances")
    print(f"[CONFIG] Offload: enabled with decode protection")
    print(f"[CONFIG] TPOT SLA: 0.1s (100ms)")
    print(f"[CONFIG] Budget scaling factors: 1.0, 1.2, 1.4, 1.6, 1.8, 2.0")
    print(f"[CONFIG] Total configurations: 6\n")

    tasks = []

    # Sweep from 1.0 to 2.0 in steps of 0.2
    for scaling_factor in [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]:
        config = SimConfig(
            trace_path=trace,
            num_prefill_instances=4,
            num_decode_instances=4,
            enable_switching=True,
            switch_policy="alpha",
            alpha_allow_decode_to_prefill=True,  # Enable decode→prefill
            alpha_allow_prefill_to_decode=False,  # Disable prefill→decode
            enable_dynamic_lp=True,
            enable_decode_protection=True,  # Enable decode protection
            tpot_sla=0.1,  # 100ms TPOT threshold
            budget_scaling_factor=scaling_factor,  # KEY: Sweep this parameter
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
            f"alpha_offload_with_protection_budget_{scaling_factor:.1f}",
            {
                "policy": "alpha",
                "offload_mode": "offload_with_protection",
                "budget_scaling_factor": scaling_factor
            }
        ))
        print(f"  [{len(tasks)}/6] Added: budget_scaling_factor={scaling_factor:.1f}")

    print(f"\n[SETUP] Total tasks configured: {len(tasks)}")
    print(f"[SETUP] Task list:")
    for i, (_, label, extras) in enumerate(tasks, 1):
        print(f"  {i:2d}. {label} (scaling={extras['budget_scaling_factor']:.1f})")

    return run_parallel(tasks, max_workers=max_workers)


def main():
    import sys
    print("\n" + "="*80)
    print("BUDGET SCALING FACTOR SWEEP (1H TRACE): ALPHA + OFFLOAD WITH PROTECTION")
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

    print(f"[INIT] Starting budget scaling factor sweep experiment (1h trace)...")
    results = experiment_offload_with_protection_budget_sweep_4p4d(max_workers=workers)
    save_experiment("offload_with_protection_budget_sweep_4p4d", results)

    print("\n" + "="*80)
    print("✓ EXPERIMENT COMPLETE!")
    print(f"✓ {len(results)} configurations tested")
    print(f"✓ Results saved to {RESULTS_DIR}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
