"""Systematic experiments for role switching policies (Alpha and V1).

Runs 4 core experiment groups:
1. Baseline policy comparison (no-switch vs alpha vs v1) on Azure trace
2. Instance configuration scaling (different P:D ratios) on Azure trace
3. Load level sensitivity (varying arrival rates) on synthetic trace
4. Switching + offload interaction (test combinations)

Results saved as JSON to experiments/results/.
"""

from __future__ import annotations

import gc
import json
import multiprocessing
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add project root to path
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from config import SimConfig
from core.engine import SimulationEngine


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
AZURE_TRACE = Path(__file__).resolve().parent.parent / "azure_code_8000.csv"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Use first N requests from Azure trace to keep runs fast
AZURE_MAX_REQUESTS = 2000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_timestamp() -> str:
    """Get current timestamp string for filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def trim_azure_trace(max_requests: int = AZURE_MAX_REQUESTS) -> Path:
    """Create a trimmed version of the Azure trace."""
    trimmed = Path(tempfile.mkdtemp(prefix="azure_trim_")) / f"azure_{max_requests}.csv"
    with open(AZURE_TRACE) as fin, open(trimmed, "w") as fout:
        header = fin.readline()
        fout.write(header)
        for i, line in enumerate(fin):
            if i >= max_requests:
                break
            fout.write(line)
    return trimmed


def generate_csv_workload(
    path: Path,
    num_requests: int,
    context_dist: str,
    context_params: Tuple,
    output_len: int = 128,
    arrival_rate: float = 5.0,
    seed: int = 42,
) -> Path:
    """Generate a synthetic CSV workload file with Poisson arrivals."""
    rng = np.random.default_rng(seed)

    if context_dist == "uniform":
        low, high = context_params
        context_tokens = rng.integers(low, high + 1, size=num_requests)
    elif context_dist == "fixed":
        context_tokens = np.full(num_requests, context_params[0], dtype=int)
    else:
        raise ValueError(f"Unknown distribution: {context_dist}")

    inter_arrivals = rng.exponential(1.0 / arrival_rate, size=num_requests)
    inter_arrivals[0] = 0.0
    timestamps = np.cumsum(inter_arrivals)

    base_dt = datetime(2023, 1, 1, 0, 0, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("TIMESTAMP,ContextTokens,GeneratedTokens\n")
        for ts, ct in zip(timestamps, context_tokens):
            dt = base_dt + timedelta(seconds=float(ts))
            ts_str = dt.strftime("%Y-%m-%d %H:%M:%S.") + f"{dt.microsecond:06d}0"
            f.write(f"{ts_str},{int(ct)},{output_len}\n")

    return path


# ---------------------------------------------------------------------------
# Parallel experiment runner
# ---------------------------------------------------------------------------

def _run_one(args: Tuple) -> Dict[str, Any]:
    """Worker function for parallel execution (must be module-level for pickle)."""
    config, label, extras = args
    engine = SimulationEngine(config, enable_iteration_logging=False)
    results = engine.run()
    d = results.to_dict()
    d["label"] = label
    d["config"] = {
        "trace_path": os.path.basename(config.trace_path),
        "num_prefill_instances": config.num_prefill_instances,
        "num_decode_instances": config.num_decode_instances,
        "enable_switching": config.enable_switching,
        "switch_policy": config.switch_policy,
        "enable_dynamic_lp": config.enable_dynamic_lp,
        "slo_target": config.slo_target,
        "max_prefill_tokens": config.max_prefill_tokens,
    }
    d.update(extras)
    return d


def run_parallel(tasks: List[Tuple], max_workers: Optional[int] = None) -> List[Dict]:
    """Run multiple simulations in parallel using ProcessPoolExecutor."""
    if max_workers is None:
        max_workers = min(len(tasks), multiprocessing.cpu_count())
    print(f"  Running {len(tasks)} simulations with {max_workers} workers...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_run_one, tasks))
    return results


def save_experiment(name: str, results: List[Dict]) -> Path:
    """Save experiment results to JSON with datetime suffix."""
    out_dir = RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = get_timestamp()
    path = out_dir / f"{name}_{ts}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved {len(results)} results to {path}")
    return path


POLICIES = [
    ("none", False, "never"),
    ("alpha", True, "alpha"),
    ("v1", True, "v1"),
]


# ---------------------------------------------------------------------------
# Experiment 1: Baseline Policy Comparison
# ---------------------------------------------------------------------------

def experiment_baseline_comparison(trace: str) -> List[Dict]:
    """Compare no-switching vs alpha vs v1 policies (2P4D, no offload)."""
    print("\n=== Experiment 1: Baseline Policy Comparison ===")
    tasks = []

    for policy_name, enable_sw, sw_policy in POLICIES:
        config = SimConfig(
            trace_path=trace,
            num_prefill_instances=2,
            num_decode_instances=4,
            enable_switching=enable_sw,
            switch_policy=sw_policy,
            enable_dynamic_lp=False,
        )
        tasks.append((config, policy_name, {"policy": policy_name}))

    return run_parallel(tasks)


# ---------------------------------------------------------------------------
# Experiment 2: Instance Configuration Scaling
# ---------------------------------------------------------------------------

def experiment_instance_config_scaling(trace: str) -> List[Dict]:
    """Test policies across different P:D ratios (no offload)."""
    print("\n=== Experiment 2: Instance Configuration Scaling ===")
    tasks = []

    configs = [
        (1, 2, "1P2D"),
        (2, 4, "2P4D"),
        (2, 6, "2P6D"),
        (4, 4, "4P4D"),
    ]

    for num_p, num_d, config_name in configs:
        for policy_name, enable_sw, sw_policy in POLICIES:
            config = SimConfig(
                trace_path=trace,
                num_prefill_instances=num_p,
                num_decode_instances=num_d,
                enable_switching=enable_sw,
                switch_policy=sw_policy,
                enable_dynamic_lp=False,
            )
            tasks.append((config, f"{config_name}_{policy_name}", {"pd_config": config_name, "policy": policy_name}))

    return run_parallel(tasks)


# ---------------------------------------------------------------------------
# Experiment 3: Load Level Sensitivity
# ---------------------------------------------------------------------------

def experiment_load_level() -> List[Dict]:
    """Test policies at different arrival rates (synthetic workloads)."""
    print("\n=== Experiment 3: Load Level Sensitivity ===")
    tasks = []

    arrival_rates = [2, 5, 8, 12, 16]
    temp_dir = Path(tempfile.mkdtemp(prefix="switching_exp_"))

    # Generate all workloads first (sequential, fast)
    for rate in arrival_rates:
        trace_path = generate_csv_workload(
            path=temp_dir / f"workload_rate{rate}.csv",
            num_requests=1000,
            context_dist="uniform",
            context_params=(1024, 4096),
            output_len=128,
            arrival_rate=rate,
            seed=42 + rate,
        )

        for policy_name, enable_sw, sw_policy in POLICIES:
            config = SimConfig(
                trace_path=str(trace_path),
                num_prefill_instances=2,
                num_decode_instances=4,
                enable_switching=enable_sw,
                switch_policy=sw_policy,
                enable_dynamic_lp=False,
            )
            tasks.append((config, f"rate{rate}_{policy_name}", {"arrival_rate": rate, "policy": policy_name}))

    return run_parallel(tasks)


# ---------------------------------------------------------------------------
# Experiment 4: Switching + Offload Interaction
# ---------------------------------------------------------------------------

def experiment_switching_offload_interaction(trace: str) -> List[Dict]:
    """Test switching + offload combinations (lightweight, no offload is baseline)."""
    print("\n=== Experiment 4: Switching + Offload Interaction ===")
    tasks = []

    for policy_name, enable_sw, sw_policy in POLICIES:
        # Without offload
        config = SimConfig(
            trace_path=trace,
            num_prefill_instances=2,
            num_decode_instances=4,
            enable_switching=enable_sw,
            switch_policy=sw_policy,
            enable_dynamic_lp=False,
        )
        tasks.append((config, f"{policy_name}_no_offload", {"policy": policy_name, "offload_enabled": False}))

        # With offload
        config = SimConfig(
            trace_path=trace,
            num_prefill_instances=2,
            num_decode_instances=4,
            enable_switching=enable_sw,
            switch_policy=sw_policy,
            enable_dynamic_lp=True,
            slo_target=1.0,
            lp_max_window_size=5,
        )
        tasks.append((config, f"{policy_name}_with_offload", {"policy": policy_name, "offload_enabled": True}))

    return run_parallel(tasks)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not AZURE_TRACE.exists():
        print(f"Error: Azure trace not found at {AZURE_TRACE}")
        sys.exit(1)

    print(f"Results will be saved to {RESULTS_DIR}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Trim Azure trace for faster runs
    trimmed_trace = str(trim_azure_trace(AZURE_MAX_REQUESTS))
    print(f"Using trimmed Azure trace ({AZURE_MAX_REQUESTS} requests)")

    all_results = {}

    # Exp 1: Baseline (3 runs)
    print("\n" + "=" * 70)
    r1 = experiment_baseline_comparison(trimmed_trace)
    save_experiment("switching_exp1_baseline", r1)
    all_results["exp1_baseline"] = r1
    del r1
    gc.collect()

    # Exp 2: Instance config (12 runs)
    print("\n" + "=" * 70)
    r2 = experiment_instance_config_scaling(trimmed_trace)
    save_experiment("switching_exp2_instance_config", r2)
    all_results["exp2_instance_config"] = r2
    del r2
    gc.collect()

    # Exp 3: Load level (15 runs, synthetic 1000-req traces)
    print("\n" + "=" * 70)
    r3 = experiment_load_level()
    save_experiment("switching_exp3_load_level", r3)
    all_results["exp3_load_level"] = r3
    del r3
    gc.collect()

    # Exp 4: Switching + offload interaction (6 runs)
    print("\n" + "=" * 70)
    r4 = experiment_switching_offload_interaction(trimmed_trace)
    save_experiment("switching_exp4_interaction", r4)
    all_results["exp4_interaction"] = r4
    del r4
    gc.collect()

    # Combined
    combined_path = RESULTS_DIR / f"switching_all_{get_timestamp()}.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {combined_path}")
    print("Done!")


if __name__ == "__main__":
    main()
