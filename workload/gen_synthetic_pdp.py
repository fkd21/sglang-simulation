"""
Generate a configurable P-D-P or D-P-D synthetic workload.

Each phase duration is specified on the command line. Token distributions and
arrival rates match the validated synthetic_30min_v2 parameters.

Prefill-heavy phase:  rate=14 req/s, ctx~Lognormal(8.2,0.4)≈3646, out~Uniform(20,80)
Decode-heavy phase:   rate=14 req/s, 85% decode (ctx~Uniform(150,400), out~Lognormal(7.2,0.4)≈1330)
                                   + 15% prefill (ctx~Lognormal(8.0,0.4)≈3004, out~Uniform(20,80))

Output: <stem>.jsonl  and  <stem>.csv  (kartik_syn format)

Usage examples:
  # P-D-P: 10min prefill, 10min decode, 10min prefill  (default)
  python workload/gen_synthetic_pdp.py

  # D-P-D: 5min decode, 8min prefill, 5min decode
  python workload/gen_synthetic_pdp.py --pattern dpd --t1 300 --t2 480 --t3 300

  # P-D-P with custom durations (seconds)
  python workload/gen_synthetic_pdp.py --pattern pdp --t1 180 --t2 240 --t3 180 --output synthetic_10min

  # Override arrival rate or token distributions
  python workload/gen_synthetic_pdp.py --prefill-rate 12 --decode-rate 16
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------

def _lognormal(mu: float, sigma: float) -> Callable[[np.random.Generator], int]:
    def sample(rng): return int(rng.lognormal(mu, sigma))
    return sample


def _uniform(low: int, high: int) -> Callable[[np.random.Generator], int]:
    def sample(rng): return int(rng.integers(low, high + 1))
    return sample


def _mixed(p_heavy: float,
           heavy: Callable[[np.random.Generator], int],
           light: Callable[[np.random.Generator], int]) -> Callable[[np.random.Generator], int]:
    def sample(rng): return heavy(rng) if rng.random() < p_heavy else light(rng)
    return sample


# ---------------------------------------------------------------------------
# Phase generators
# ---------------------------------------------------------------------------

def _prefill_phase(start_s: float, end_s: float, rate: float,
                   rng: np.random.Generator) -> list[dict]:
    """Prefill-heavy: long context, short output. input_tput ≈ rate × 3646 tok/s."""
    return _make_requests(start_s, end_s, rate,
                          ctx=_lognormal(8.2, 0.4),    # mean ≈ 3646
                          out=_uniform(20, 80),         # mean ≈ 50
                          rng=rng)


def _decode_phase(start_s: float, end_s: float, rate: float,
                  rng: np.random.Generator) -> list[dict]:
    """Decode-heavy: 85% short ctx/long out + 15% prefill-heavy mix.
    output_tput ≈ rate × 1138 tok/s; prefill tail keeps alpha signal non-zero."""
    return _make_requests(start_s, end_s, rate,
                          ctx=_mixed(0.15, _lognormal(8.0, 0.4), _uniform(150, 400)),
                          out=_mixed(0.15, _uniform(20, 80),      _lognormal(7.2, 0.4)),
                          rng=rng)


def _make_requests(start_s: float, end_s: float, rate: float,
                   ctx: Callable, out: Callable,
                   rng: np.random.Generator) -> list[dict]:
    reqs, t = [], start_s
    while True:
        t += rng.exponential(1.0 / rate)
        if t >= end_s:
            break
        reqs.append({
            "timestamp": int(t * 1000),
            "input_length":  int(np.clip(ctx(rng), 32, 32768)),
            "output_length": int(np.clip(out(rng),  1,  8192)),
        })
    return reqs


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def _stats(name: str, reqs: list[dict], start_s: float, end_s: float) -> None:
    if not reqs:
        print(f"  {name}: 0 requests")
        return
    ctxs = np.array([r["input_length"]  for r in reqs])
    outs = np.array([r["output_length"] for r in reqs])
    n, d = len(reqs), end_s - start_s
    print(f"  {name}: n={n:5d}  rate={n/d:5.1f} req/s  "
          f"avg_ctx={ctxs.mean():5.0f}  avg_out={outs.mean():5.0f}  "
          f"input_tput={ctxs.mean()*n/d:6.0f} tok/s  "
          f"output_tput={outs.mean()*n/d:6.0f} tok/s  "
          f"opt_P:D={ctxs.mean()/max(outs.mean(),1):.2f}:1")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate(pattern: str, t1: float, t2: float, t3: float,
             prefill_rate: float, decode_rate: float,
             output_stem: Path, seed: int) -> None:

    assert pattern in ("pdp", "dpd"), "--pattern must be 'pdp' or 'dpd'"
    rng = np.random.default_rng(seed)

    boundaries = [0.0, t1, t1 + t2, t1 + t2 + t3]
    total_s = boundaries[-1]

    # Build phases according to pattern
    phases: list[tuple[str, list[dict]]] = []
    for i, (label_char, s, e) in enumerate(zip(pattern + pattern[-1],   # pdp or dpd
                                               boundaries[:-1], boundaries[1:])):
        if label_char == "p":
            reqs = _prefill_phase(s, e, prefill_rate, rng)
            phases.append((f"Phase{i+1} prefill-heavy ({int(e-s)}s)", reqs, s, e))
        else:
            reqs = _decode_phase(s, e, decode_rate, rng)
            phases.append((f"Phase{i+1} decode-heavy  ({int(e-s)}s)", reqs, s, e))

    all_requests = sorted(
        [r for _, reqs, *_ in phases for r in reqs],
        key=lambda r: r["timestamp"]
    )

    # Write JSONL
    jsonl_path = output_stem.with_suffix(".jsonl")
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "w") as f:
        for req in all_requests:
            f.write(json.dumps(req) + "\n")

    # Write CSV (kartik_syn format)
    base_dt = datetime(2024, 5, 14, 13, 0, 0, tzinfo=timezone.utc)
    csv_path = output_stem.with_suffix(".csv")
    with open(csv_path, "w") as f:
        f.write("TIMESTAMP,ContextTokens,GeneratedTokens\n")
        for req in all_requests:
            ts = base_dt + timedelta(milliseconds=req["timestamp"])
            f.write(f"{ts.strftime('%Y-%m-%dT%H:%M:%S.%f')}Z,"
                    f"{req['input_length']},{req['output_length']}\n")

    # Print stats
    pattern_str = "-".join(
        f"{'P' if c == 'p' else 'D'}({int(d)}s)"
        for c, d in zip(pattern, [t1, t2, t3])
    )
    print(f"\n{'='*72}")
    print(f"Pattern  : {pattern_str}  (total {total_s:.0f}s = {total_s/60:.1f} min)")
    print(f"JSONL    : {jsonl_path}  ({len(all_requests):,} requests)")
    print(f"CSV      : {csv_path}")
    print(f"{'='*72}")
    for name, reqs, s, e in phases:
        _stats(name, reqs, s, e)
    print(f"{'='*72}\n")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent

    p = argparse.ArgumentParser(description="Generate P-D-P or D-P-D synthetic workload")
    p.add_argument("--pattern",      default="pdp",  choices=["pdp", "dpd"],
                   help="Phase order: pdp (prefill-decode-prefill) or dpd (default: pdp)")
    p.add_argument("--t1",           type=float, default=600.0,
                   help="Duration of phase 1 in seconds (default: 600)")
    p.add_argument("--t2",           type=float, default=600.0,
                   help="Duration of phase 2 in seconds (default: 600)")
    p.add_argument("--t3",           type=float, default=600.0,
                   help="Duration of phase 3 in seconds (default: 600)")
    p.add_argument("--prefill-rate", type=float, default=14.0,
                   help="Arrival rate for prefill phases in req/s (default: 14)")
    p.add_argument("--decode-rate",  type=float, default=14.0,
                   help="Arrival rate for decode phases in req/s (default: 14)")
    p.add_argument("--output", "-o", type=Path,
                   default=None,
                   help="Output path stem (no extension); defaults to synthetic_<pattern>_<t1>+<t2>+<t3>")
    p.add_argument("--seed",         type=int, default=42)
    args = p.parse_args()

    if args.output is None:
        name = f"synthetic_{args.pattern}_{int(args.t1)}+{int(args.t2)}+{int(args.t3)}"
        args.output = root / name

    generate(
        pattern=args.pattern,
        t1=args.t1, t2=args.t2, t3=args.t3,
        prefill_rate=args.prefill_rate,
        decode_rate=args.decode_rate,
        output_stem=args.output,
        seed=args.seed,
    )
