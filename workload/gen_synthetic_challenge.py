"""
Generate a synthetic challenge workload for alpha_v4 + offload with protection evaluation.

The trace is designed for a 5P3D (5 prefill + 3 decode) configuration and contains
6 phases over 90 minutes (5400 seconds), each with a different optimal P/D ratio to
stress the policy's switching decisions.

Phase overview:
  Phase 1 (0–600s):    Warm-up, balanced load.          Optimal P:D ≈ 6.7:1 → ~6P2D
  Phase 2 (600–1800s): Decode-heavy ramp.               Optimal P:D ≈ 1:4  → ~2P6D
  Phase 3 (1800–2700s): Prefill-heavy + periodic bursts. Optimal P:D ≈ 64:1 → ~7P1D
  Phase 4 (2700–3600s): Bimodal burst stress.            Optimal P:D ≈ 2:1  (volatile)
  Phase 5 (3600–4800s): Decode memory pressure.         Optimal P:D ≈ 0.6:1 → ~3P5D
  Phase 6 (4800–5400s): Cool-down + final large burst.  Optimal P:D ≈ 6.7:1

Output format (Mooncake JSONL, one record per line):
  {"timestamp": <ms>, "input_length": <int>, "output_length": <int>}

Usage:
  python workload/gen_synthetic_challenge.py [--output PATH] [--seed INT]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, List, Optional

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


def _bimodal_ctx_sampler(
    p_decode_heavy: float = 0.5,
    low1: int = 150, high1: int = 500,        # decode-heavy arm: short context
    mu2: float = 8.2, sigma2: float = 0.4,    # prefill-heavy arm: long context
) -> Callable[[np.random.Generator], int]:
    def sample(rng: np.random.Generator) -> int:
        if rng.random() < p_decode_heavy:
            return int(rng.integers(low1, high1 + 1))
        else:
            return int(rng.lognormal(mu2, sigma2))
    return sample


def _bimodal_out_sampler(
    p_decode_heavy: float = 0.5,
    low1: int = 800, high1: int = 2500,   # decode-heavy arm: long output
    low2: int = 30,  high2: int = 100,    # prefill-heavy arm: short output
) -> Callable[[np.random.Generator], int]:
    def sample(rng: np.random.Generator) -> int:
        if rng.random() < p_decode_heavy:
            return int(rng.integers(low1, high1 + 1))
        else:
            return int(rng.integers(low2, high2 + 1))
    return sample


# ---------------------------------------------------------------------------
# Core generation helpers
# ---------------------------------------------------------------------------

def _make_phase_requests(
    start_s: float,
    end_s: float,
    rate_fn: Callable[[float], float],
    ctx_sampler: Callable[[np.random.Generator], int],
    out_sampler: Callable[[np.random.Generator], int],
    rng: np.random.Generator,
    ctx_clip: tuple[int, int] = (32, 32768),
    out_clip: tuple[int, int] = (1, 8192),
) -> list[dict]:
    """Generate base Poisson arrival stream for one phase."""
    requests = []
    t = start_s
    while True:
        rate = rate_fn(t)
        dt = rng.exponential(1.0 / rate)
        t += dt
        if t >= end_s:
            break
        ctx = int(np.clip(ctx_sampler(rng), *ctx_clip))
        out = int(np.clip(out_sampler(rng), *out_clip))
        requests.append({"timestamp": int(t * 1000), "input_length": ctx, "output_length": out})
    return requests


def _inject_bursts(
    burst_times_s: list[float],
    n_per_burst: int,
    window_s: float,
    ctx_sampler: Callable[[np.random.Generator], int],
    out_sampler: Callable[[np.random.Generator], int],
    rng: np.random.Generator,
    ctx_clip: tuple[int, int] = (32, 32768),
    out_clip: tuple[int, int] = (1, 8192),
) -> list[dict]:
    """Generate burst cluster events (separate from the base Poisson stream)."""
    extra = []
    for bt in burst_times_s:
        # Exponential inter-arrivals within the window, re-scaled to fit exactly
        inter = rng.exponential(window_s / n_per_burst, size=n_per_burst)
        offsets = np.cumsum(inter)
        offsets = offsets / offsets[-1] * window_s  # normalize into [0, window_s]
        for off in offsets:
            t = bt + float(off)
            ctx = int(np.clip(ctx_sampler(rng), *ctx_clip))
            out = int(np.clip(out_sampler(rng), *out_clip))
            extra.append({"timestamp": int(t * 1000), "input_length": ctx, "output_length": out})
    return extra


# ---------------------------------------------------------------------------
# Statistics helper
# ---------------------------------------------------------------------------

def _phase_stats(requests: list[dict], start_s: float, end_s: float, name: str) -> None:
    if not requests:
        print(f"  {name}: 0 requests")
        return
    ctxs = [r["input_length"] for r in requests]
    outs = [r["output_length"] for r in requests]
    duration = end_s - start_s
    n = len(requests)
    avg_ctx = np.mean(ctxs)
    avg_out = np.mean(outs)
    input_tput = avg_ctx * (n / duration)
    output_tput = avg_out * (n / duration)
    print(
        f"  {name}: n={n:5d}, rate={n/duration:5.1f} req/s, "
        f"avg_ctx={avg_ctx:6.0f}, avg_out={avg_out:6.0f}, "
        f"input_tput={input_tput:7.0f} tok/s, output_tput={output_tput:6.0f} tok/s, "
        f"optimal_P:D≈{avg_ctx/max(avg_out,1):.2f}:1"
    )


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate(output_path: Path, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    all_requests: list[dict] = []
    phase_records: list[tuple[str, list[dict], float, float]] = []

    # ---- Phase 1: Warm-up, balanced (0–600s) --------------------------------
    # Optimal P:D ≈ 992:148 = 6.7:1  →  system ~6P2D ideal; 5P3D slightly under-prefill
    p1_start, p1_end = 0.0, 600.0
    p1_base = _make_phase_requests(
        start_s=p1_start, end_s=p1_end,
        rate_fn=lambda t: 10.0,
        ctx_sampler=_lognormal_sampler(6.9, 0.5),   # mean ≈ 992
        out_sampler=_lognormal_sampler(5.0, 0.5),   # mean ≈ 148
        rng=rng,
    )
    phase_records.append(("Phase1 warmup", p1_base, p1_start, p1_end))
    all_requests.extend(p1_base)

    # ---- Phase 2: Decode-heavy ramp (600–1800s) ------------------------------
    # Optimal P:D ≈ 300:1213 = 1:4  →  ~2P6D ideal; 5P3D decode bottleneck
    # Arrival rate ramps linearly 8→15 req/s
    p2_start, p2_end = 600.0, 1800.0
    p2_duration = p2_end - p2_start
    def _p2_rate(t: float) -> float:
        frac = (t - p2_start) / p2_duration
        return 8.0 + 7.0 * frac   # 8 → 15 req/s

    p2_base = _make_phase_requests(
        start_s=p2_start, end_s=p2_end,
        rate_fn=_p2_rate,
        ctx_sampler=_uniform_sampler(150, 450),       # mean ≈ 300
        out_sampler=_lognormal_sampler(7.1, 0.4),    # mean ≈ 1213
        rng=rng,
    )
    # Small bursts every 150s (all decode-heavy)
    p2_burst_times = list(np.arange(p2_start + 150, p2_end, 150))
    p2_bursts = _inject_bursts(
        burst_times_s=p2_burst_times,
        n_per_burst=20, window_s=5.0,
        ctx_sampler=_uniform_sampler(150, 400),
        out_sampler=_lognormal_sampler(7.2, 0.3),
        rng=rng,
    )
    p2_all = p2_base + p2_bursts
    phase_records.append(("Phase2 decode-heavy", p2_all, p2_start, p2_end))
    all_requests.extend(p2_all)

    # ---- Phase 3: Prefill-heavy + periodic bursts (1800–2700s) ---------------
    # Optimal P:D ≈ 4002:62 = 64:1  →  ~7P1D ideal; 5P3D heavy prefill bottleneck
    # Input tput ≈ 10 × 4002 = 40,020 tok/s  (near SLA boundary)
    p3_start, p3_end = 1800.0, 2700.0
    p3_base = _make_phase_requests(
        start_s=p3_start, end_s=p3_end,
        rate_fn=lambda t: 10.0,
        ctx_sampler=_lognormal_sampler(8.3, 0.5),   # mean ≈ 4002
        out_sampler=_uniform_sampler(25, 100),       # mean ≈ 62
        rng=rng,
    )
    # Medium bursts every 200s (prefill-heavy type, stress alpha spike)
    p3_burst_times = list(np.arange(p3_start + 200, p3_end, 200))
    p3_bursts = _inject_bursts(
        burst_times_s=p3_burst_times,
        n_per_burst=40, window_s=6.0,
        ctx_sampler=_lognormal_sampler(8.4, 0.4),   # even heavier prefill
        out_sampler=_uniform_sampler(20, 80),
        rng=rng,
    )
    p3_all = p3_base + p3_bursts
    phase_records.append(("Phase3 prefill-heavy", p3_all, p3_start, p3_end))
    all_requests.extend(p3_all)

    # ---- Phase 4: Bimodal burst stress (2700–3600s) --------------------------
    # 50% decode-heavy + 50% prefill-heavy → avg P:D ≈ 1986:858 = 2.3:1
    # Burst requests are ALL prefill-heavy to spike alpha sharply
    p4_start, p4_end = 2700.0, 3600.0
    p4_base = _make_phase_requests(
        start_s=p4_start, end_s=p4_end,
        rate_fn=lambda t: 14.0,
        ctx_sampler=_bimodal_ctx_sampler(p_decode_heavy=0.5),
        out_sampler=_bimodal_out_sampler(p_decode_heavy=0.5),
        rng=rng,
    )
    # High-freq bursts every 120s, all prefill-heavy (alpha spike trigger)
    p4_burst_times = list(np.arange(p4_start + 120, p4_end, 120))
    p4_bursts = _inject_bursts(
        burst_times_s=p4_burst_times,
        n_per_burst=50, window_s=4.0,
        ctx_sampler=_lognormal_sampler(8.2, 0.4),   # prefill-heavy
        out_sampler=_uniform_sampler(30, 100),
        rng=rng,
    )
    p4_all = p4_base + p4_bursts
    phase_records.append(("Phase4 bimodal+burst", p4_all, p4_start, p4_end))
    all_requests.extend(p4_all)

    # ---- Phase 5: Decode memory pressure (3600–4800s) ------------------------
    # Optimal P:D ≈ 665:1096 = 0.61:1  →  ~3P5D ideal; decode KV cache pressure.
    # avg_output kept at ~1096 (output_tput ≈ 8.8K tok/s) to stay well within
    # feasible range (Phase2 showed ~14K is near the edge for decode-heavy).
    # Shorter ctx (avg ~665) lets many requests accumulate in decode KV cache
    # simultaneously, pushing 3-instance utilization toward the 80% threshold
    # that triggers alpha_v4's P→D switching + offload protection.
    p5_start, p5_end = 3600.0, 4800.0
    p5_base = _make_phase_requests(
        start_s=p5_start, end_s=p5_end,
        rate_fn=lambda t: 8.0,
        ctx_sampler=_lognormal_sampler(6.5, 0.4),   # mean ≈ 665
        out_sampler=_lognormal_sampler(7.0, 0.4),   # mean ≈ 1096
        rng=rng,
    )
    # Light bursts every 300s (same distribution)
    p5_burst_times = list(np.arange(p5_start + 300, p5_end, 300))
    p5_bursts = _inject_bursts(
        burst_times_s=p5_burst_times,
        n_per_burst=15, window_s=6.0,
        ctx_sampler=_lognormal_sampler(6.5, 0.4),
        out_sampler=_lognormal_sampler(7.0, 0.4),
        rng=rng,
    )
    p5_all = p5_base + p5_bursts
    phase_records.append(("Phase5 decode-mem", p5_all, p5_start, p5_end))
    all_requests.extend(p5_all)

    # ---- Phase 6: Cool-down + final large burst (4800–5400s) -----------------
    # Optimal P:D ≈ 898:134 = 6.7:1  →  ~6P2D ideal; light load
    # Final large burst at t=5100s: 200 requests in 15s
    p6_start, p6_end = 4800.0, 5400.0
    p6_duration = p6_end - p6_start
    def _p6_rate(t: float) -> float:
        frac = (t - p6_start) / p6_duration
        return 12.0 - 7.0 * frac   # 12 → 5 req/s

    p6_base = _make_phase_requests(
        start_s=p6_start, end_s=p6_end,
        rate_fn=_p6_rate,
        ctx_sampler=_lognormal_sampler(6.8, 0.6),   # mean ≈ 898
        out_sampler=_lognormal_sampler(4.9, 0.5),   # mean ≈ 134
        rng=rng,
    )
    # One large burst at t=5100s (mixed type, to test recovery)
    p6_bursts = _inject_bursts(
        burst_times_s=[5100.0],
        n_per_burst=200, window_s=15.0,
        ctx_sampler=_lognormal_sampler(7.5, 0.6),   # moderate context
        out_sampler=_lognormal_sampler(5.5, 0.5),   # moderate output
        rng=rng,
    )
    p6_all = p6_base + p6_bursts
    phase_records.append(("Phase6 cooldown+burst", p6_all, p6_start, p6_end))
    all_requests.extend(p6_all)

    # ---- Sort and write ------------------------------------------------------
    all_requests.sort(key=lambda r: r["timestamp"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for req in all_requests:
            f.write(json.dumps(req) + "\n")

    # ---- Print statistics ----------------------------------------------------
    print(f"\n{'='*70}")
    print(f"Synthetic challenge workload: {output_path}")
    print(f"{'='*70}")
    print(f"Total requests : {len(all_requests):,}")
    total_s = 5400.0
    print(f"Total duration : {total_s:.0f}s ({total_s/60:.1f} min)")
    print(f"Overall rate   : {len(all_requests)/total_s:.1f} req/s")
    print()
    print("Per-phase breakdown:")
    for name, reqs, start, end in phase_records:
        _phase_stats(reqs, start, end, name)

    # Global token throughput
    all_ctx = [r["input_length"] for r in all_requests]
    all_out = [r["output_length"] for r in all_requests]
    print()
    print(f"Global avg input throughput : {np.mean(all_ctx) * len(all_requests) / total_s:,.0f} tok/s")
    print(f"Global avg output throughput: {np.mean(all_out) * len(all_requests) / total_s:,.0f} tok/s")
    print(f"Global avg context length   : {np.mean(all_ctx):.0f} tokens")
    print(f"Global avg output length    : {np.mean(all_out):.0f} tokens")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic challenge workload")
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "synthetic_challenge_90min.jsonl",
        help="Output JSONL file path",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    generate(args.output, seed=args.seed)
