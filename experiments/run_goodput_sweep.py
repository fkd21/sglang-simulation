"""Goodput sweep: FluidPD (alpha_v6) vs static baselines on synthetic PDP workload.

Generates synthetic P(600s)-D(600s)-P(600s) traces at multiple arrival rates,
runs each through static baselines (6p2d, 5p3d, 4p4d) and alpha_v6 variants
(switching-only and switching+offload), then finds the maximum RPS where each
achieves SLA attainment > 90% (goodput threshold).

Reports:
  - SLA attainment table (config × RPS)
  - Goodput for best baseline and each alpha_v6 variant
  - Goodput ratio (FluidPD / baseline)
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

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from config import SimConfig
from core.engine import SimulationEngine
from workload.gen_synthetic_pdp import generate as gen_pdp

RESULTS_DIR = Path(__file__).resolve().parent / "results"
TRACES_DIR = _project_root  # save traces alongside existing *.jsonl files

# RPS values to sweep
RPS_LIST = [8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
SLA_THRESHOLD = 90.0  # percent


# ---------------------------------------------------------------------------
# Helpers (mirrors run_alpha_v6_only.py)
# ---------------------------------------------------------------------------

def _run_one(args: Tuple) -> Dict[str, Any]:
    config, label, extras = args
    print(f"[START] {label}", flush=True)
    engine = SimulationEngine(config)
    results = engine.run()
    print(f"[DONE]  {label}", flush=True)

    d = results.to_dict()
    d["label"] = label
    d["config"] = {
        "trace_path": os.path.basename(config.trace_path),
        "num_prefill_instances": config.num_prefill_instances,
        "num_decode_instances": config.num_decode_instances,
        "enable_dynamic_lp": config.enable_dynamic_lp,
        "enable_decode_protection": config.enable_decode_protection,
        "enable_switching": config.enable_switching,
        "switch_policy": config.switch_policy,
        "budget_scaling_factor": getattr(config, "budget_scaling_factor", None),
    }
    d.update(extras)
    return d


def run_parallel(tasks: List[Tuple], max_workers: Optional[int] = None) -> List[Dict]:
    if max_workers is None:
        max_workers = min(len(tasks), multiprocessing.cpu_count(), 12)
    print(f"\n{'='*60}")
    print(f"Running {len(tasks)} simulations with {max_workers} workers...")
    print(f"{'='*60}\n")

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_run_one, task) for task in tasks]
        for i, future in enumerate(futures):
            try:
                result = future.result()
                results.append(result)
                print(f"[{len(results)}/{len(tasks)}] done: {tasks[i][1]}")
            except Exception as e:
                import traceback
                print(f"[ERROR] {tasks[i][1]}: {e}")
                traceback.print_exc()

    print(f"\nCompleted {len(results)}/{len(tasks)} simulations\n")
    return results


def save_experiment(name: str, results: List[Dict]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"{ts}_{name}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[SAVE] {path}")
    return path


# ---------------------------------------------------------------------------
# Trace generation
# ---------------------------------------------------------------------------

def ensure_traces(rps_list: List[int], test_mode: bool = False) -> Dict[int, str]:
    """Generate PDP traces for each RPS value if not already on disk."""
    traces = {}
    for rps in rps_list:
        path = TRACES_DIR / f"synthetic_pdp_rps{rps:02d}.jsonl"
        if not path.exists():
            print(f"[TRACE] Generating {path.name} at {rps} req/s ...")
            gen_pdp(
                pattern="pdp",
                t1=600.0,
                t2=600.0,
                t3=600.0,
                prefill_rate=float(rps),
                decode_rate=float(rps),
                output_stem=path.with_suffix(""),
                seed=42,
            )
        else:
            n = sum(1 for _ in open(path))
            print(f"[TRACE] Reusing {path.name} ({n:,} requests)")
        traces[rps] = str(path)
    return traces


# ---------------------------------------------------------------------------
# Task builder
# ---------------------------------------------------------------------------

def build_tasks(traces: Dict[int, str], test_mode: bool = False) -> List[Tuple]:
    """Build (config, label, extras) for each RPS × config combination."""
    rps_list = [min(traces.keys())] if test_mode else sorted(traces.keys())
    tasks = []

    for rps in rps_list:
        trace = traces[rps]

        # ── Baseline: 6p2d ──────────────────────────────────────────────
        tasks.append((
            SimConfig(
                trace_path=trace,
                num_prefill_instances=6,
                num_decode_instances=2,
                enable_switching=False,
                switch_policy="none",
                enable_dynamic_lp=False,
                enable_monitoring=False,
            ),
            f"baseline_6p2d_rps{rps:02d}",
            {"rps": rps, "policy": "none", "variant": "baseline_6p2d"},
        ))

        # ── Baseline: 5p3d ──────────────────────────────────────────────
        tasks.append((
            SimConfig(
                trace_path=trace,
                num_prefill_instances=5,
                num_decode_instances=3,
                enable_switching=False,
                switch_policy="none",
                enable_dynamic_lp=False,
                enable_monitoring=False,
            ),
            f"baseline_5p3d_rps{rps:02d}",
            {"rps": rps, "policy": "none", "variant": "baseline_5p3d"},
        ))

        # ── Baseline: 4p4d ──────────────────────────────────────────────
        tasks.append((
            SimConfig(
                trace_path=trace,
                num_prefill_instances=4,
                num_decode_instances=4,
                enable_switching=False,
                switch_policy="none",
                enable_dynamic_lp=False,
                enable_monitoring=False,
            ),
            f"baseline_4p4d_rps{rps:02d}",
            {"rps": rps, "policy": "none", "variant": "baseline_4p4d"},
        ))

        # ── Alpha V6: switching only, no offload ─────────────────────────
        tasks.append((
            SimConfig(
                trace_path=trace,
                num_prefill_instances=4,
                num_decode_instances=4,
                enable_switching=True,
                switch_policy="alpha_v6",
                alpha_v6_allow_decode_to_prefill=True,
                alpha_v6_allow_prefill_to_decode=True,
                enable_dynamic_lp=False,
                enable_monitoring=False,
            ),
            f"alpha_v6_no_offload_rps{rps:02d}",
            {"rps": rps, "policy": "alpha_v6", "variant": "alpha_v6_no_offload"},
        ))

        # ── Alpha V6: switching + offload with decode protection ─────────
        tasks.append((
            SimConfig(
                trace_path=trace,
                num_prefill_instances=4,
                num_decode_instances=4,
                enable_switching=True,
                switch_policy="alpha_v6",
                alpha_v6_allow_decode_to_prefill=True,
                alpha_v6_allow_prefill_to_decode=True,
                enable_dynamic_lp=True,
                enable_decode_protection=True,
                tpot_sla=0.1,
                budget_scaling_factor=1,
                slo_target=1.0,
                lp_max_window_size=20,
                enable_monitoring=False,
            ),
            f"alpha_v6_offload_rps{rps:02d}",
            {"rps": rps, "policy": "alpha_v6", "variant": "alpha_v6_offload"},
        ))

    return tasks


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze(results: List[Dict]) -> None:
    from collections import defaultdict

    # sla[variant][rps] = attainment %
    sla: Dict[str, Dict[int, float]] = defaultdict(dict)
    for d in results:
        sla[d["variant"]][d["rps"]] = d.get("sla_attainment_rate", 0.0)

    all_rps = sorted({d["rps"] for d in results})
    variants = ["baseline_6p2d", "baseline_5p3d", "baseline_4p4d",
                "alpha_v6_no_offload", "alpha_v6_offload"]

    # Print table
    print(f"\n{'='*80}")
    print("SLA Attainment (%) by RPS and Config")
    print(f"{'='*80}")
    header = f"{'RPS':>5}  " + "  ".join(f"{v:>22}" for v in variants)
    print(header)
    print("-" * len(header))
    for rps in all_rps:
        row = f"{rps:>5}  " + "  ".join(
            f"{sla[v].get(rps, float('nan')):>22.1f}" for v in variants
        )
        print(row)

    # Goodput = max RPS with SLA > threshold
    def goodput(variant: str) -> Optional[int]:
        candidates = [rps for rps in all_rps if sla[variant].get(rps, 0) > SLA_THRESHOLD]
        return max(candidates) if candidates else None

    best_baseline_goodput = None
    best_baseline_variant = None
    for v in ["baseline_6p2d", "baseline_5p3d", "baseline_4p4d"]:
        g = goodput(v)
        if g is not None and (best_baseline_goodput is None or g > best_baseline_goodput):
            best_baseline_goodput = g
            best_baseline_variant = v

    gp_no_offload = goodput("alpha_v6_no_offload")
    gp_offload = goodput("alpha_v6_offload")

    print(f"\n{'='*80}")
    print("Goodput Summary (max RPS with SLA > 90%)")
    print(f"{'='*80}")
    print(f"  Best baseline ({best_baseline_variant}): {best_baseline_goodput} req/s")
    print(f"  alpha_v6 (no offload):                 {gp_no_offload} req/s")
    print(f"  alpha_v6 (offload, budget_sf=1):        {gp_offload} req/s")

    if best_baseline_goodput:
        if gp_no_offload:
            print(f"\n  Goodput ratio (no offload):  {gp_no_offload / best_baseline_goodput:.3f}×")
        if gp_offload:
            print(f"  Goodput ratio (offload):     {gp_offload / best_baseline_goodput:.3f}×")
    print(f"{'='*80}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    test_mode = "--test" in sys.argv

    print("\n" + "=" * 80)
    print("GOODPUT SWEEP: FluidPD (alpha_v6) vs Static Baselines — Synthetic PDP")
    if test_mode:
        print("  *** TEST MODE: 1 RPS level, 1 worker ***")
    print("=" * 80 + "\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    traces = ensure_traces(RPS_LIST, test_mode=test_mode)
    tasks = build_tasks(traces, test_mode=test_mode)
    print(f"\n[SETUP] {len(tasks)} total simulations queued\n")

    workers = 1 if test_mode else None
    results = run_parallel(tasks, max_workers=workers)

    path = save_experiment("goodput_sweep_pdp", results)
    analyze(results)

    print(f"Results saved to: {path}")


if __name__ == "__main__":
    main()
