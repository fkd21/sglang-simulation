"""Decode-heavy experiments for role switching policies.

Tests how alpha and v1 policies behave under decode-heavy workloads
(short prefill, long generation) where the bottleneck is decode capacity.

Experiments:
1. Decode-heavy vs prefill-heavy baseline comparison
2. Varying output length (decode heaviness) sensitivity
3. P:D ratio sensitivity under decode-heavy load
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

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from config import SimConfig
from core.engine import SimulationEngine

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def generate_csv_workload(
    path: Path,
    num_requests: int,
    context_tokens: int,
    output_len: int,
    arrival_rate: float,
    seed: int = 42,
) -> Path:
    """Generate CSV workload with fixed context/output lengths."""
    rng = np.random.default_rng(seed)
    inter_arrivals = rng.exponential(1.0 / arrival_rate, size=num_requests)
    inter_arrivals[0] = 0.0
    timestamps = np.cumsum(inter_arrivals)

    base_dt = datetime(2023, 1, 1, 0, 0, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("TIMESTAMP,ContextTokens,GeneratedTokens\n")
        for ts in timestamps:
            dt = base_dt + timedelta(seconds=float(ts))
            ts_str = dt.strftime("%Y-%m-%d %H:%M:%S.") + f"{dt.microsecond:06d}0"
            f.write(f"{ts_str},{context_tokens},{output_len}\n")
    return path


# ---------------------------------------------------------------------------
# Parallel experiment runner
# ---------------------------------------------------------------------------

def _run_one(args: Tuple) -> Dict[str, Any]:
    """Worker function for parallel execution (must be module-level for pickle)."""
    config, label, extras = args
    engine = SimulationEngine(config)
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


def get_timestamp() -> str:
    """Get current timestamp string for filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_experiment(name: str, results: List[Dict]) -> Path:
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
# Experiment 1: Decode-heavy vs prefill-heavy baseline
# ---------------------------------------------------------------------------

def experiment_workload_profile() -> List[Dict]:
    """Compare decode-heavy vs prefill-heavy workloads.

    Decode-heavy: short context (256), long output (1024)
    Balanced:     medium context (1024), medium output (256)
    Prefill-heavy: long context (4096), short output (32)
    """
    print("\n=== Exp1: Workload Profile Comparison ===")
    temp_dir = Path(tempfile.mkdtemp(prefix="decode_heavy_"))
    tasks = []

    profiles = [
        ("decode_heavy", 256, 1024),
        ("balanced", 1024, 256),
        ("prefill_heavy", 4096, 32),
    ]

    rate = 8  # Fixed arrival rate

    # Generate all workloads first (sequential, fast)
    for profile_name, ctx, out in profiles:
        trace = generate_csv_workload(
            path=temp_dir / f"{profile_name}.csv",
            num_requests=1000,
            context_tokens=ctx,
            output_len=out,
            arrival_rate=rate,
            seed=42,
        )
        for policy_name, enable_sw, sw_policy in POLICIES:
            config = SimConfig(
                trace_path=str(trace),
                num_prefill_instances=2,
                num_decode_instances=4,
                enable_switching=enable_sw,
                switch_policy=sw_policy,
                slo_target=1.0,
            )
            tasks.append((config, f"{profile_name}_{policy_name}", {
                "profile": profile_name,
                "context_tokens": ctx,
                "output_len": out,
                "policy": policy_name,
                "arrival_rate": rate,
            }))

    return run_parallel(tasks)


# ---------------------------------------------------------------------------
# Experiment 2: Output length sensitivity (decode heaviness)
# ---------------------------------------------------------------------------

def experiment_output_length_sensitivity() -> List[Dict]:
    """Vary output length while keeping context short to change decode pressure.

    Fixed: context=256, rate=8 req/s, 2P4D
    Sweep: output_len in [64, 256, 512, 1024, 2048]
    """
    print("\n=== Exp2: Output Length Sensitivity ===")
    temp_dir = Path(tempfile.mkdtemp(prefix="outlen_sweep_"))
    tasks = []

    output_lens = [64, 256, 512, 1024, 2048]
    ctx = 256
    rate = 8

    # Generate all workloads first
    for out_len in output_lens:
        trace = generate_csv_workload(
            path=temp_dir / f"out{out_len}.csv",
            num_requests=1000,
            context_tokens=ctx,
            output_len=out_len,
            arrival_rate=rate,
            seed=42,
        )
        for policy_name, enable_sw, sw_policy in POLICIES:
            config = SimConfig(
                trace_path=str(trace),
                num_prefill_instances=2,
                num_decode_instances=4,
                enable_switching=enable_sw,
                switch_policy=sw_policy,
                slo_target=1.0,
            )
            tasks.append((config, f"out{out_len}_{policy_name}", {
                "output_len": out_len,
                "context_tokens": ctx,
                "policy": policy_name,
                "arrival_rate": rate,
            }))

    return run_parallel(tasks)


# ---------------------------------------------------------------------------
# Experiment 3: P:D ratio under decode-heavy load
# ---------------------------------------------------------------------------

def experiment_pd_ratio_decode_heavy() -> List[Dict]:
    """Test different P:D ratios under decode-heavy workload.

    Fixed: context=256, output=1024, rate=8
    Sweep: P:D ratios from prefill-heavy to decode-heavy allocation
    """
    print("\n=== Exp3: P:D Ratio under Decode-Heavy Load ===")
    temp_dir = Path(tempfile.mkdtemp(prefix="pd_decode_"))
    tasks = []

    trace = generate_csv_workload(
        path=temp_dir / "decode_heavy.csv",
        num_requests=1000,
        context_tokens=256,
        output_len=1024,
        arrival_rate=8,
        seed=42,
    )

    pd_configs = [
        (1, 2, "1P2D"),
        (1, 5, "1P5D"),
        (2, 4, "2P4D"),
        (3, 3, "3P3D"),
        (4, 2, "4P2D"),
    ]

    for num_p, num_d, config_name in pd_configs:
        for policy_name, enable_sw, sw_policy in POLICIES:
            config = SimConfig(
                trace_path=str(trace),
                num_prefill_instances=num_p,
                num_decode_instances=num_d,
                enable_switching=enable_sw,
                switch_policy=sw_policy,
                slo_target=1.0,
            )
            tasks.append((config, f"{config_name}_{policy_name}", {
                "pd_config": config_name,
                "policy": policy_name,
            }))

    return run_parallel(tasks)


# ---------------------------------------------------------------------------
# Experiment 4: High decode pressure (large batch, long output, light prefill)
# ---------------------------------------------------------------------------

def experiment_high_decode_pressure() -> List[Dict]:
    """Stress-test decode with high arrival rate + long output + short context.

    This creates scenarios where decode batch size grows very large,
    making decode the clear bottleneck while prefill is idle.

    Sweep: arrival_rate x output_len, fixed ctx=128, 2P4D
    """
    print("\n=== Exp4: High Decode Pressure ===")
    temp_dir = Path(tempfile.mkdtemp(prefix="high_decode_"))
    tasks = []

    ctx = 128
    configs = [
        # (rate, output_len) - increasing decode pressure
        (10, 2048),   # moderate pressure
        (15, 2048),   # high pressure
        (20, 2048),   # very high pressure
        (10, 4096),   # extreme output length
        (15, 4096),   # extreme everything
    ]

    # Generate all workloads first
    for rate, out_len in configs:
        trace = generate_csv_workload(
            path=temp_dir / f"r{rate}_o{out_len}.csv",
            num_requests=500,
            context_tokens=ctx,
            output_len=out_len,
            arrival_rate=rate,
            seed=42,
        )
        for policy_name, enable_sw, sw_policy in POLICIES:
            config = SimConfig(
                trace_path=str(trace),
                num_prefill_instances=2,
                num_decode_instances=4,
                enable_switching=enable_sw,
                switch_policy=sw_policy,
                slo_target=2.0,
            )
            tasks.append((config, f"r{rate}_o{out_len}_{policy_name}", {
                "output_len": out_len,
                "context_tokens": ctx,
                "policy": policy_name,
                "arrival_rate": rate,
            }))

    return run_parallel(tasks)


# ---------------------------------------------------------------------------
# Experiment 5: P:D ratio under extreme decode pressure
# ---------------------------------------------------------------------------

def experiment_pd_ratio_extreme_decode() -> List[Dict]:
    """Test P:D ratios under extreme decode pressure.

    Fixed: ctx=128, output=4096, rate=12
    Sweep: P:D ratios with total=6 instances
    """
    print("\n=== Exp5: P:D Ratio under Extreme Decode Pressure ===")
    temp_dir = Path(tempfile.mkdtemp(prefix="pd_extreme_"))
    tasks = []

    trace = generate_csv_workload(
        path=temp_dir / "extreme_decode.csv",
        num_requests=500,
        context_tokens=128,
        output_len=4096,
        arrival_rate=12,
        seed=42,
    )

    pd_configs = [
        (1, 5, "1P5D"),
        (2, 4, "2P4D"),
        (3, 3, "3P3D"),
        (4, 2, "4P2D"),
        (5, 1, "5P1D"),
    ]

    for num_p, num_d, config_name in pd_configs:
        for policy_name, enable_sw, sw_policy in POLICIES:
            config = SimConfig(
                trace_path=str(trace),
                num_prefill_instances=num_p,
                num_decode_instances=num_d,
                enable_switching=enable_sw,
                switch_policy=sw_policy,
                slo_target=2.0,
            )
            tasks.append((config, f"{config_name}_{policy_name}", {
                "pd_config": config_name,
                "policy": policy_name,
                "output_len": 4096,
                "context_tokens": 128,
                "arrival_rate": 12,
            }))

    return run_parallel(tasks)


def main():
    print(f"Results will be saved to {RESULTS_DIR}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}

    print("\n" + "=" * 70)
    r1 = experiment_workload_profile()
    save_experiment("decode_heavy_exp1_profile", r1)
    all_results["exp1_profile"] = r1
    del r1
    gc.collect()

    print("\n" + "=" * 70)
    r2 = experiment_output_length_sensitivity()
    save_experiment("decode_heavy_exp2_outlen", r2)
    all_results["exp2_outlen"] = r2
    del r2
    gc.collect()

    print("\n" + "=" * 70)
    r3 = experiment_pd_ratio_decode_heavy()
    save_experiment("decode_heavy_exp3_pd_ratio", r3)
    all_results["exp3_pd_ratio"] = r3
    del r3
    gc.collect()

    print("\n" + "=" * 70)
    r4 = experiment_high_decode_pressure()
    save_experiment("decode_heavy_exp4_high_pressure", r4)
    all_results["exp4_high_pressure"] = r4
    del r4
    gc.collect()

    print("\n" + "=" * 70)
    r5 = experiment_pd_ratio_extreme_decode()
    save_experiment("decode_heavy_exp5_pd_extreme", r5)
    all_results["exp5_pd_extreme"] = r5
    del r5
    gc.collect()

    combined = RESULTS_DIR / f"decode_heavy_all_{get_timestamp()}.json"
    with open(combined, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {combined}")
    print("Done!")


if __name__ == "__main__":
    main()
