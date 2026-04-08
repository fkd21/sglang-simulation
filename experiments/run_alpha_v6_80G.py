"""Run Alpha V6-Only Experiment on A100 80GB GPUs.

Same as run_alpha_v6_only.py but with gpu_type="A100_80GB":
  - KV cache per instance: 670,484 tokens (vs 164,458 on 40GB)
  - Profiling formulas unchanged (BYTES_PER_TOKEN = 84,876 held constant)

Alpha V6 uses avg(allocatable_kv / total_kv) as the decode health signal.
  decode_allocatable_low  = 0.2  (pd trigger when ratio < 0.2, decode is tight)
  decode_allocatable_high = 0.6  (dp trigger when ratio > 0.6, decode has room)
"""

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
AZURE_TRACE_1H = Path(__file__).resolve().parent.parent / "kartik_syn.csv"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

GPU_TYPE = "A100_80GB"


def _run_one(args: Tuple) -> Dict[str, Any]:
    """Worker function for parallel execution."""
    config, label, extras = args
    print(f"[START] Running simulation: {label}", flush=True)
    print(f"  - Policy: {extras.get('policy', 'N/A')}", flush=True)
    print(f"  - Offload mode: {extras.get('offload_mode', 'N/A')}", flush=True)
    print(f"  - GPU type: {config.gpu_type}", flush=True)
    print(f"  - Switching: {config.enable_switching}, Protection: {config.enable_decode_protection}", flush=True)
    print(f"  - Alpha V6 decode→prefill: {config.alpha_v6_allow_decode_to_prefill}", flush=True)
    print(f"  - Alpha V6 prefill→decode: {config.alpha_v6_allow_prefill_to_decode}", flush=True)

    engine = SimulationEngine(config)
    print(f"[SIMULATING] {label} - engine created, starting simulation...", flush=True)
    results = engine.run()
    print(f"[DONE] {label} - simulation completed", flush=True)

    d = results.to_dict()
    d["label"] = label
    d["config"] = {
        "trace_path": os.path.basename(config.trace_path),
        "gpu_type": config.gpu_type,
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
        "alpha_v6_allow_decode_to_prefill": config.alpha_v6_allow_decode_to_prefill,
        "alpha_v6_allow_prefill_to_decode": config.alpha_v6_allow_prefill_to_decode,
        "decode_allocatable_low": config.decode_allocatable_low,
        "decode_allocatable_high": config.decode_allocatable_high,
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


def experiment_alpha_v6_80g(max_workers: Optional[int] = None) -> List[Dict]:
    """Test alpha_v6 policy on A100 80GB with 1h Azure code requests.

    Identical topology and policy sweep as run_alpha_v6_only.py;
    only gpu_type="A100_80GB" is added to every SimConfig.

    Policies tested:
    - baselines: 6p2d, 5p3d, 4p4d, 1p7d (no switching)
    - alpha_v6: no_offload / offload_no_protection / offload_with_protection
    """
    print("\n" + "="*80)
    print("=== Alpha V6 Experiment — A100 80GB (670,484 KV tokens/instance) ===")
    print("=== Allocatable-Ratio Decode Memory Guard ===")
    print("=== decode_allocatable_high=0.6 (d→p), decode_allocatable_low=0.2 (p→d) ===")
    print("="*80 + "\n")
    trace = str(_ensure_azure_1h())

    tasks = []

    # Baseline 1: 6p2d
    config = SimConfig(
        trace_path=trace,
        num_prefill_instances=6,
        num_decode_instances=2,
        gpu_type=GPU_TYPE,
        enable_switching=False,
        switch_policy="none",
        enable_dynamic_lp=False,
        enable_streaming_loading=True,
        streaming_window_size=300.0,
        streaming_lookback=60.0,
        enable_monitoring=True,
        monitoring_plot_interval_minutes=60.0,
    )
    tasks.append((
        config,
        "baseline_6p2d",
        {"policy": "none", "offload_mode": "no_offload"},
    ))
    print(f"  [1/4] Added: baseline_6p2d")

    # Baseline 2: 5p3d
    config = SimConfig(
        trace_path=trace,
        num_prefill_instances=5,
        num_decode_instances=3,
        gpu_type=GPU_TYPE,
        enable_switching=False,
        switch_policy="none",
        enable_dynamic_lp=False,
        enable_streaming_loading=True,
        streaming_window_size=300.0,
        streaming_lookback=60.0,
        enable_monitoring=True,
        monitoring_plot_interval_minutes=60.0,
    )
    tasks.append((
        config,
        "baseline_5p3d",
        {"policy": "none", "offload_mode": "no_offload"},
    ))
    print(f"  [2/4] Added: baseline_5p3d")

    # Baseline 3: 4p4d
    config = SimConfig(
        trace_path=trace,
        num_prefill_instances=4,
        num_decode_instances=4,
        gpu_type=GPU_TYPE,
        enable_switching=False,
        switch_policy="none",
        enable_dynamic_lp=False,
        enable_streaming_loading=True,
        streaming_window_size=300.0,
        streaming_lookback=60.0,
        enable_monitoring=True,
        monitoring_plot_interval_minutes=60.0,
    )
    tasks.append((
        config,
        "baseline_4p4d",
        {"policy": "none", "offload_mode": "no_offload"},
    ))
    print(f"  [3/4] Added: baseline_4p4d")

    # Baseline 4: 1p7d
    config = SimConfig(
        trace_path=trace,
        num_prefill_instances=1,
        num_decode_instances=7,
        gpu_type=GPU_TYPE,
        enable_switching=False,
        switch_policy="none",
        enable_dynamic_lp=False,
        enable_streaming_loading=True,
        streaming_window_size=300.0,
        streaming_lookback=60.0,
        enable_monitoring=True,
        monitoring_plot_interval_minutes=60.0,
    )
    tasks.append((
        config,
        "baseline_1p7d",
        {"policy": "none", "offload_mode": "no_offload"},
    ))
    print(f"  [4/4] Added: baseline_1p7d")

    policies = [
        ("alpha_v6", True, "alpha_v6"),
    ]
    print(f"\n[CONFIG] Testing alpha_v6 policy")
    print(f"[CONFIG] Each policy tested with 3 offload modes")
    print(f"[CONFIG] Total configurations: {len(policies) * 3}\n")

    for policy_name, enable_sw, sw_policy in policies:
        print(f"\n[SETUP] Configuring tasks for policy: {policy_name}")

        # 1. Without offload
        config = SimConfig(
            trace_path=trace,
            num_prefill_instances=4,
            num_decode_instances=4,
            gpu_type=GPU_TYPE,
            enable_switching=enable_sw,
            switch_policy=sw_policy,
            alpha_v6_allow_decode_to_prefill=True,
            alpha_v6_allow_prefill_to_decode=True,
            enable_dynamic_lp=False,
            enable_streaming_loading=True,
            streaming_window_size=300.0,
            streaming_lookback=60.0,
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
            gpu_type=GPU_TYPE,
            enable_switching=enable_sw,
            switch_policy=sw_policy,
            alpha_v6_allow_decode_to_prefill=True,
            alpha_v6_allow_prefill_to_decode=True,
            enable_dynamic_lp=True,
            enable_decode_protection=False,
            slo_target=1.0,
            lp_max_window_size=20,
            enable_streaming_loading=True,
            streaming_window_size=300.0,
            streaming_lookback=60.0,
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
            gpu_type=GPU_TYPE,
            enable_switching=enable_sw,
            switch_policy=sw_policy,
            alpha_v6_allow_decode_to_prefill=True,
            alpha_v6_allow_prefill_to_decode=True,
            enable_dynamic_lp=True,
            enable_decode_protection=True,
            tpot_sla=0.1,
            budget_scaling_factor=2,
            slo_target=1.0,
            lp_max_window_size=20,
            enable_streaming_loading=True,
            streaming_window_size=300.0,
            streaming_lookback=60.0,
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
    print("ALPHA V6 EXPERIMENT — A100 80GB (670,484 KV TOKENS/INSTANCE)")
    print("="*80)
    print(f"[INIT] Results will be saved to {RESULTS_DIR}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    test_mode = "--test" in sys.argv
    workers = 1 if test_mode else None
    if test_mode:
        print("\n" + "!"*80)
        print("!!! TEST MODE: Using 1 worker !!!")
        print("!"*80 + "\n")

    print(f"[INIT] Starting alpha_v6 80G experiment (1h trace)...")
    results = experiment_alpha_v6_80g(max_workers=workers)
    save_experiment("alpha_v6_80g", results)

    print("\n" + "="*80)
    print("✓ EXPERIMENT COMPLETE!")
    print(f"✓ {len(results)} configurations tested")
    print(f"✓ Results saved to {RESULTS_DIR}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
