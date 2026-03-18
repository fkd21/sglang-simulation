"""Systematic experiments comparing partial offload vs baseline.

Runs 6 experiment groups:
1. SLO target sensitivity
2. Instance configuration scaling
3. Context length distribution
4. LP window size
5. Load level (arrival rate)
6. 8-instance P:D ratio sweep (1P7D through 7P1D)
7. Switching + offload interaction at 1P7D (3 policies x 2 offload modes)

Results saved as JSON to simulation/experiments/results/.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add project root to path
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))

from simulation.config import SimConfig
from simulation.core.engine import SimulationEngine


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
AZURE_TRACE = Path(__file__).resolve().parent.parent / "azure_code_8000.csv"
AZURE_TRACE_500 = Path(__file__).resolve().parent.parent / "azure_code_500.csv"
AZURE_TRACE_32000 = Path(__file__).resolve().parent.parent / "azure_code_32000.csv"
AZURE_CODE_WEEK = Path(__file__).resolve().parent.parent / "AzureLLMInferenceTrace_code_1week.csv"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ---------------------------------------------------------------------------
# Synthetic workload generation
# ---------------------------------------------------------------------------

def generate_jsonl_workload(
    path: Path,
    num_requests: int,
    context_dist: str,
    context_params: Tuple,
    output_len: int = 128,
    seed: int = 42,
) -> Path:
    """Generate a synthetic JSONL workload file.

    Args:
        path: Output file path.
        context_dist: Distribution type - "uniform", "lognormal", or "fixed".
        context_params: Parameters for the distribution.
            uniform: (low, high)
            lognormal: (mean, sigma)
            fixed: (value,)
        output_len: Fixed output length for all requests.
        seed: Random seed.
    """
    rng = np.random.default_rng(seed)

    if context_dist == "uniform":
        low, high = context_params
        context_tokens = rng.integers(low, high + 1, size=num_requests)
    elif context_dist == "lognormal":
        mu, sigma = context_params
        raw = rng.lognormal(mu, sigma, size=num_requests)
        context_tokens = np.clip(raw, 64, 8192).astype(int)
    elif context_dist == "fixed":
        context_tokens = np.full(num_requests, context_params[0], dtype=int)
    else:
        raise ValueError(f"Unknown distribution: {context_dist}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ct in context_tokens:
            f.write(json.dumps({"input_len": int(ct), "output_len": output_len}) + "\n")

    return path


def generate_csv_workload(
    path: Path,
    num_requests: int,
    context_dist: str,
    context_params: Tuple,
    output_len: int = 128,
    arrival_rate: float = 5.0,
    seed: int = 42,
) -> Path:
    """Generate a synthetic CSV workload file with timestamps.

    Uses Poisson arrivals at the specified rate.
    """
    rng = np.random.default_rng(seed)

    if context_dist == "uniform":
        low, high = context_params
        context_tokens = rng.integers(low, high + 1, size=num_requests)
    elif context_dist == "lognormal":
        mu, sigma = context_params
        raw = rng.lognormal(mu, sigma, size=num_requests)
        context_tokens = np.clip(raw, 64, 8192).astype(int)
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
            dt = base_dt + __import__("datetime").timedelta(seconds=float(ts))
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
        "enable_dynamic_lp": config.enable_dynamic_lp,
        "slo_target": config.slo_target,
        "lp_max_window_size": config.lp_max_window_size,
        "max_prefill_tokens": config.max_prefill_tokens,
        "enable_switching": config.enable_switching,
        "switch_policy": config.switch_policy,
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
    """Save experiment results to JSON."""
    out_dir = RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved {len(results)} results to {path}")
    return path


# ---------------------------------------------------------------------------
# Experiment 1: SLO Target Sensitivity
# ---------------------------------------------------------------------------

def experiment_slo_sensitivity() -> List[Dict]:
    """Vary SLO target, compare baseline vs partial offload."""
    print("\n=== Experiment 1: SLO Target Sensitivity ===")
    trace = str(AZURE_TRACE)
    slo_targets = [0.1, 0.3, 0.5, 0.8, 1.0, 2.0, 5.0]
    tasks = []

    # Baseline (no offload)
    config = SimConfig(
        trace_path=trace,
        num_prefill_instances=2,
        num_decode_instances=4,
        enable_dynamic_lp=False,
    )
    tasks.append((config, "baseline", {"slo_target": None}))

    # Partial offload with different SLO targets
    for slo in slo_targets:
        config = SimConfig(
            trace_path=trace,
            num_prefill_instances=2,
            num_decode_instances=4,
            enable_dynamic_lp=True,
            slo_target=slo,
        )
        tasks.append((config, f"offload_slo={slo}", {"slo_target": slo}))

    return run_parallel(tasks)


# ---------------------------------------------------------------------------
# Experiment 2: Instance Configuration Scaling
# ---------------------------------------------------------------------------

def experiment_instance_config() -> List[Dict]:
    """Vary P:D ratio, compare baseline vs partial offload."""
    print("\n=== Experiment 2: Instance Configuration ===")
    trace = str(AZURE_TRACE)
    configs = [(1, 1), (1, 2), (1, 4), (2, 4), (2, 6), (4, 4)]
    tasks = []

    for p, d in configs:
        # Baseline
        config = SimConfig(
            trace_path=trace,
            num_prefill_instances=p,
            num_decode_instances=d,
            enable_dynamic_lp=False,
        )
        tasks.append((config, f"{p}P{d}D_baseline", {"pd_config": f"{p}P{d}D", "mode": "baseline"}))

        # Partial offload
        config = SimConfig(
            trace_path=trace,
            num_prefill_instances=p,
            num_decode_instances=d,
            enable_dynamic_lp=True,
            slo_target=1.0,
        )
        tasks.append((config, f"{p}P{d}D_offload", {"pd_config": f"{p}P{d}D", "mode": "offload"}))

    return run_parallel(tasks)


# ---------------------------------------------------------------------------
# Experiment 3: Context Length Distribution
# ---------------------------------------------------------------------------

def experiment_context_length() -> List[Dict]:
    """Vary context token distribution, compare baseline vs offload."""
    print("\n=== Experiment 3: Context Length Distribution ===")
    tmp_dir = Path(tempfile.mkdtemp(prefix="sim_exp_ctx_"))
    tasks = []

    workloads = {
        "short": ("uniform", (64, 512)),
        "medium": ("uniform", (1024, 4096)),
        "long": ("uniform", (4096, 8192)),
        "mixed": ("lognormal", (7.0, 1.0)),  # mean ~ exp(7) ≈ 1097, wide spread
    }

    # Generate all workloads first (sequential, fast)
    for name, (dist, params) in workloads.items():
        trace_path = generate_jsonl_workload(
            tmp_dir / f"ctx_{name}.jsonl",
            num_requests=500,
            context_dist=dist,
            context_params=params,
            output_len=128,
        )

        # Baseline
        config = SimConfig(
            trace_path=str(trace_path),
            num_prefill_instances=2,
            num_decode_instances=4,
            enable_dynamic_lp=False,
        )
        tasks.append((config, f"ctx_{name}_baseline", {"workload": name, "mode": "baseline"}))

        # Offload
        config = SimConfig(
            trace_path=str(trace_path),
            num_prefill_instances=2,
            num_decode_instances=4,
            enable_dynamic_lp=True,
            slo_target=1.0,
        )
        tasks.append((config, f"ctx_{name}_offload", {"workload": name, "mode": "offload"}))

    # Also test with Azure trace
    config_bl = SimConfig(
        trace_path=str(AZURE_TRACE),
        num_prefill_instances=2,
        num_decode_instances=4,
        enable_dynamic_lp=False,
    )
    tasks.append((config_bl, "ctx_azure_baseline", {"workload": "azure", "mode": "baseline"}))

    config_off = SimConfig(
        trace_path=str(AZURE_TRACE),
        num_prefill_instances=2,
        num_decode_instances=4,
        enable_dynamic_lp=True,
        slo_target=1.0,
    )
    tasks.append((config_off, "ctx_azure_offload", {"workload": "azure", "mode": "offload"}))

    return run_parallel(tasks)


# ---------------------------------------------------------------------------
# Experiment 4: LP Window Size
# ---------------------------------------------------------------------------

def experiment_window_size() -> List[Dict]:
    """Vary LP solver window size."""
    print("\n=== Experiment 4: LP Window Size ===")
    trace = str(AZURE_TRACE)
    window_sizes = [1, 3, 5, 10, 15]
    tasks = []

    # Baseline
    config = SimConfig(
        trace_path=trace,
        num_prefill_instances=2,
        num_decode_instances=4,
        enable_dynamic_lp=False,
    )
    tasks.append((config, "baseline", {"window_size": 0}))

    for ws in window_sizes:
        config = SimConfig(
            trace_path=trace,
            num_prefill_instances=2,
            num_decode_instances=4,
            enable_dynamic_lp=True,
            slo_target=1.0,
            lp_max_window_size=ws,
        )
        tasks.append((config, f"window={ws}", {"window_size": ws}))

    return run_parallel(tasks)


# ---------------------------------------------------------------------------
# Experiment 5: Load Level
# ---------------------------------------------------------------------------

def experiment_load_level() -> List[Dict]:
    """Vary arrival rate, compare baseline vs partial offload."""
    print("\n=== Experiment 5: Load Level ===")
    tmp_dir = Path(tempfile.mkdtemp(prefix="sim_exp_load_"))
    arrival_rates = [1, 2, 5, 10, 20]
    tasks = []

    # Generate all workloads first (sequential, fast)
    for rate in arrival_rates:
        trace_path = generate_csv_workload(
            tmp_dir / f"load_{rate}rps.csv",
            num_requests=1000,
            context_dist="uniform",
            context_params=(1024, 4096),
            output_len=128,
            arrival_rate=float(rate),
        )

        # Baseline
        config = SimConfig(
            trace_path=str(trace_path),
            num_prefill_instances=2,
            num_decode_instances=4,
            enable_dynamic_lp=False,
        )
        tasks.append((config, f"load_{rate}rps_baseline", {"arrival_rate": rate, "mode": "baseline"}))

        # Offload
        config = SimConfig(
            trace_path=str(trace_path),
            num_prefill_instances=2,
            num_decode_instances=4,
            enable_dynamic_lp=True,
            slo_target=1.0,
        )
        tasks.append((config, f"load_{rate}rps_offload", {"arrival_rate": rate, "mode": "offload"}))

    return run_parallel(tasks)


# ---------------------------------------------------------------------------
# Experiment 6: 8-Instance P:D Ratio Sweep
# ---------------------------------------------------------------------------

def experiment_pd_ratio_8instances() -> List[Dict]:
    """Sweep P:D ratio with total 8 instances (1P7D..7P1D), baseline vs offload."""
    print("\n=== Experiment 6: 8-Instance P:D Ratio Sweep ===")
    trace = str(AZURE_TRACE_500)
    configs = [(1, 7), (2, 6), (3, 5), (4, 4), (5, 3), (6, 2), (7, 1)]
    tasks = []

    for p, d in configs:
        # Baseline
        config = SimConfig(
            trace_path=trace,
            num_prefill_instances=p,
            num_decode_instances=d,
            enable_dynamic_lp=False,
        )
        tasks.append((config, f"{p}P{d}D_baseline", {"pd_config": f"{p}P{d}D", "mode": "baseline", "total_instances": 8}))

        # Partial offload
        config = SimConfig(
            trace_path=trace,
            num_prefill_instances=p,
            num_decode_instances=d,
            enable_dynamic_lp=True,
            slo_target=1.0,
        )
        tasks.append((config, f"{p}P{d}D_offload", {"pd_config": f"{p}P{d}D", "mode": "offload", "total_instances": 8}))

    return run_parallel(tasks)


# ---------------------------------------------------------------------------
# Experiment 7: Switching + Offload Interaction at 1P7D
# ---------------------------------------------------------------------------

def _ensure_azure_32000() -> Path:
    """Extract first 32000 rows from the week-long Azure code trace if needed."""
    if AZURE_TRACE_32000.exists():
        # Quick check row count
        with open(AZURE_TRACE_32000) as f:
            n = sum(1 for _ in f) - 1  # minus header
        if n >= 32000:
            return AZURE_TRACE_32000

    if not AZURE_CODE_WEEK.exists():
        raise FileNotFoundError(
            f"Week-long Azure trace not found at {AZURE_CODE_WEEK}. "
            "Please place the file there or create azure_code_32000.csv manually."
        )

    print(f"  Extracting first 32000 rows from {AZURE_CODE_WEEK}...")
    df = pd.read_csv(AZURE_CODE_WEEK, nrows=32000)
    df.to_csv(AZURE_TRACE_32000, index=False)
    print(f"  Saved {len(df)} rows to {AZURE_TRACE_32000}")
    return AZURE_TRACE_32000


def experiment_switching_offload_1p7d() -> List[Dict]:
    """Test 3 switching policies x 2 offload modes at 1P7D with 32K Azure code requests."""
    print("\n=== Experiment 7: Switching + Offload at 1P7D ===")
    trace = str(_ensure_azure_32000())

    policies = [
        ("none", False, "never"),
        ("alpha", True, "alpha"),
        ("v1", True, "v1"),
    ]
    tasks = []

    for policy_name, enable_sw, sw_policy in policies:
        # Without offload
        config = SimConfig(
            trace_path=trace,
            num_prefill_instances=1,
            num_decode_instances=7,
            enable_switching=enable_sw,
            switch_policy=sw_policy,
            enable_dynamic_lp=False,
        )
        tasks.append((
            config,
            f"{policy_name}_no_offload",
            {"policy": policy_name, "offload_enabled": False},
        ))

        # With offload
        config = SimConfig(
            trace_path=trace,
            num_prefill_instances=1,
            num_decode_instances=7,
            enable_switching=enable_sw,
            switch_policy=sw_policy,
            enable_dynamic_lp=True,
            slo_target=1.0,
            lp_max_window_size=5,
        )
        tasks.append((
            config,
            f"{policy_name}_offload",
            {"policy": policy_name, "offload_enabled": True},
        ))

    return run_parallel(tasks)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not AZURE_TRACE.exists():
        print(f"Error: Azure trace not found at {AZURE_TRACE}")
        sys.exit(1)
    if not AZURE_TRACE_500.exists():
        print(f"Error: Azure trace not found at {AZURE_TRACE_500}")
        sys.exit(1)

    print(f"Results will be saved to {RESULTS_DIR}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # Run all experiments
    r1 = experiment_slo_sensitivity()
    save_experiment("exp1_slo_sensitivity", r1)
    all_results["exp1_slo_sensitivity"] = r1

    r2 = experiment_instance_config()
    save_experiment("exp2_instance_config", r2)
    all_results["exp2_instance_config"] = r2

    r3 = experiment_context_length()
    save_experiment("exp3_context_length", r3)
    all_results["exp3_context_length"] = r3

    r4 = experiment_window_size()
    save_experiment("exp4_window_size", r4)
    all_results["exp4_window_size"] = r4

    r5 = experiment_load_level()
    save_experiment("exp5_load_level", r5)
    all_results["exp5_load_level"] = r5

    r6 = experiment_pd_ratio_8instances()
    save_experiment("exp6_pd_ratio_8instances", r6)
    all_results["exp6_pd_ratio_8instances"] = r6

    r7 = experiment_switching_offload_1p7d()
    save_experiment("exp7_switching_offload_1p7d", r7)
    all_results["exp7_switching_offload_1p7d"] = r7

    # Save combined results
    combined_path = RESULTS_DIR / "all_experiments.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {combined_path}")
    print("Done!")


if __name__ == "__main__":
    main()
