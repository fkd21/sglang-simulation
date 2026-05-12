"""
Generate a 30-minute P-D-P synthetic workload for policy evaluation.

Three equal phases (10 min each), pure Poisson arrivals, no bursts:
  Phase 1 (0–600s):    Prefill-heavy  rate=14 req/s  ctx≈3646  out≈50    input≈51K tok/s
  Phase 2 (600–1200s): Decode-heavy   rate=14 req/s  ctx≈684   out≈1138  output≈15.9K tok/s  (85% decode + 15% prefill mix)
  Phase 3 (1200–1800s): Prefill-heavy  rate=14 req/s  ctx≈3646  out≈50    input≈51K tok/s

The P→D→P pattern forces two switching decisions:
  D→P switch around t=0–600s (or upon entry to Phase 1)
  P→D switch around t=600s (Phase 2 onset, decode becomes bottleneck)
  D→P switch around t=1200s (Phase 3 onset, prefill becomes bottleneck again)

Capacity constraints (from 4P4D Azure mixed 24h experiment):
  - Input tput safe zone:  20K–40K tok/s
  - Output tput safe limit: ~15K tok/s (Phase2-level confirmed feasible)

Output format (Mooncake JSONL):
  {"timestamp": <ms>, "input_length": <int>, "output_length": <int>}

Usage:
  python workload/gen_synthetic_30min.py [--output PATH] [--seed INT]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------

def _lognormal_sampler(mu: float, sigma: float) -> Callable[[np.random.Generator], int]:
    def sample(rng: np.random.Generator) -> int:
        return int(rng.lognormal(mu, sigma))
    return sample


def _uniform_sampler(low: int, high: int) -> Callable[[np.random.Generator], int]:
    def sample(rng: np.random.Generator) -> int:
        return int(rng.integers(low, high + 1))
    return sample


def _mixed_sampler(
    p_heavy: float,
    heavy_sampler: Callable[[np.random.Generator], int],
    light_sampler: Callable[[np.random.Generator], int],
) -> Callable[[np.random.Generator], int]:
    """Return heavy_sampler with probability p_heavy, else light_sampler."""
    def sample(rng: np.random.Generator) -> int:
        return heavy_sampler(rng) if rng.random() < p_heavy else light_sampler(rng)
    return sample


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------

def _make_phase_requests(
    start_s: float,
    end_s: float,
    rate: float,
    ctx_sampler: Callable[[np.random.Generator], int],
    out_sampler: Callable[[np.random.Generator], int],
    rng: np.random.Generator,
    ctx_clip: tuple[int, int] = (32, 32768),
    out_clip: tuple[int, int] = (1, 8192),
) -> list[dict]:
    requests = []
    t = start_s
    while True:
        t += rng.exponential(1.0 / rate)
        if t >= end_s:
            break
        ctx = int(np.clip(ctx_sampler(rng), *ctx_clip))
        out = int(np.clip(out_sampler(rng), *out_clip))
        requests.append({"timestamp": int(t * 1000), "input_length": ctx, "output_length": out})
    return requests


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _phase_stats(requests: list[dict], start_s: float, end_s: float, name: str) -> None:
    if not requests:
        print(f"  {name}: 0 requests")
        return
    ctxs = np.array([r["input_length"] for r in requests])
    outs = np.array([r["output_length"] for r in requests])
    duration = end_s - start_s
    n = len(requests)
    print(
        f"  {name}: n={n:5d}, rate={n/duration:5.1f} req/s, "
        f"avg_ctx={ctxs.mean():6.0f}, avg_out={outs.mean():6.0f}, "
        f"input_tput={ctxs.mean()*n/duration:7.0f} tok/s, "
        f"output_tput={outs.mean()*n/duration:6.0f} tok/s, "
        f"optimal_P:D={ctxs.mean()/max(outs.mean(),1):.2f}:1"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate(output_path: Path, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    all_requests: list[dict] = []
    phase_records = []

    # Phase 1: Prefill-heavy (0–600s)
    # input_tput ≈ 14 × 3646 = 51,044 tok/s  — exceeds 4P4D capacity (~40K ceiling);
    # 6P2D handles 51K/6 ≈ 8.5K tok/s per instance → comfortable.
    # optimal P:D ≈ 73:1  →  ~7P1D
    p1 = _make_phase_requests(
        0.0, 600.0, rate=14.0,
        ctx_sampler=_lognormal_sampler(8.2, 0.4),  # mean ≈ 3646
        out_sampler=_uniform_sampler(20, 80),       # mean ≈ 50
        rng=rng,
    )
    phase_records.append(("Phase1 prefill-heavy", p1, 0.0, 600.0))
    all_requests.extend(p1)

    # Phase 2: Decode-heavy with 15% prefill-heavy mixture (600–1200s)
    # 85% decode-heavy: ctx≈275, out≈1330
    # 15% prefill-heavy: ctx≈3004, out≈50  (mirrors Phase 1/3 distribution)
    # Replicates real Azure patterns where request types don't drop to zero.
    # Effective avg_ctx ≈ 0.85×275 + 0.15×3004 = 684; avg_out ≈ 0.85×1330 + 0.15×50 = 1138
    # input_tput  ≈ 14 × 684  =  9,576 tok/s   (prefill tail keeps alpha signal non-zero)
    # output_tput ≈ 14 × 1138 = 15,932 tok/s   (decode still dominant → 2P6D optimal)
    # optimal P:D ≈ 684:1138 = 0.60:1  →  ~3P5D / 2P6D
    p2 = _make_phase_requests(
        600.0, 1200.0, rate=14.0,
        ctx_sampler=_mixed_sampler(
            p_heavy=0.15,
            heavy_sampler=_lognormal_sampler(8.0, 0.4),  # prefill-heavy: mean ≈ 3004
            light_sampler=_uniform_sampler(150, 400),     # decode-heavy:  mean ≈ 275
        ),
        out_sampler=_mixed_sampler(
            p_heavy=0.15,
            heavy_sampler=_uniform_sampler(20, 80),       # prefill-heavy: mean ≈ 50
            light_sampler=_lognormal_sampler(7.2, 0.4),  # decode-heavy:  mean ≈ 1330
        ),
        rng=rng,
    )
    phase_records.append(("Phase2 decode-heavy", p2, 600.0, 1200.0))
    all_requests.extend(p2)

    # Phase 3: Prefill-heavy (1200–1800s)  — same as Phase 1
    p3 = _make_phase_requests(
        1200.0, 1800.0, rate=14.0,
        ctx_sampler=_lognormal_sampler(8.2, 0.4),
        out_sampler=_uniform_sampler(20, 80),
        rng=rng,
    )
    phase_records.append(("Phase3 prefill-heavy", p3, 1200.0, 1800.0))
    all_requests.extend(p3)

    # Sort and write
    all_requests.sort(key=lambda r: r["timestamp"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for req in all_requests:
            f.write(json.dumps(req) + "\n")

    # Print statistics
    print(f"\n{'='*70}")
    print(f"Synthetic 30-min workload: {output_path}")
    print(f"{'='*70}")
    print(f"Total requests : {len(all_requests):,}")
    print(f"Total duration : 1800s (30.0 min)")
    print(f"Overall rate   : {len(all_requests)/1800:.1f} req/s")
    print()
    print("Per-phase breakdown:")
    for name, reqs, s, e in phase_records:
        _phase_stats(reqs, s, e, name)
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 30-min P-D-P synthetic workload")
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "synthetic_30min.jsonl",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate(args.output, seed=args.seed)
