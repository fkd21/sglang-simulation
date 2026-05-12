# SGLang Simulation

A discrete-event simulator for LLM inference clusters running SGLang-style prefill/decode disaggregation. It models GPU instances, KV-cache memory, KV-transfer latency, and role-switching policies, enabling offline evaluation of scheduling strategies before deployment.

---

## Quick Start

# Change the dataset path in simulation scripts under "experiments/" to change the dataset.
```bash
cd /path/to/sglang-simulation

# Fast sanity check (single worker, small subset)
python experiments/run_alpha_v6_only.py --test

# Full parallel run
python experiments/run_alpha_v6_only.py

# Goodput sweep across RPS levels
python experiments/run_goodput_sweep.py [--test]
```

Each script spawns a `ProcessPoolExecutor` over multiple `SimConfig` variants and writes a JSON summary to `experiments/results/`.

---

## Dataset / Workload Formats

### Supported input files

| Format | Extension | Required columns |
|--------|-----------|-----------------|
| Azure inference trace | `.csv` | `TIMESTAMP, ContextTokens, GeneratedTokens` |
| BurstGPT trace | `.csv` | `Timestamp, Request tokens, Response tokens` |
| JSONL (Mooncake style) | `.jsonl` | `{"timestamp": <ms>, "input_length": …, "output_length": …}` |
| JSONL (legacy) | `.jsonl` | `{"input_len": …, "output_len": …}` |

- `TIMESTAMP` / `Timestamp` columns are converted to seconds relative to the first request.
- JSONL `timestamp` is in **milliseconds** and is converted to relative seconds at load time.
- JSONL without a timestamp field: all requests arrive at `t = 0`.
- Format is auto-detected from CSV column headers.

### Generating synthetic workloads

```bash
python workload/gen_synthetic_pdp.py \
  --pattern pdp \        # "pdp" = prefill-heavy → decode-heavy → prefill-heavy; "dpd" = inverse
  --t1 600 --t2 600 --t3 600 \   # phase durations in seconds
  --prefill-rate 14.0 \  # arrival rate during prefill phases (req/s)
  --decode-rate 14.0 \   # arrival rate during decode phase (req/s)
  --output my_workload   # output stem → my_workload.jsonl + my_workload.csv
```

### Large traces (streaming load)

For traces with >100 K requests, enable streaming to avoid OOM:

```python
SimConfig(
    trace_path="big_trace.csv",
    enable_streaming_loading=True,
    streaming_window_size=300.0,   # load 5-minute windows
    streaming_lookback=60.0,       # 1-minute safety margin
)
```

---

## Key SimConfig Parameters

All parameters live in `config.py` as a single `SimConfig` dataclass.

### Instance layout

| Parameter | Default | Description |
|-----------|---------|-------------|
| `trace_path` | — | Path to workload file (CSV or JSONL) |
| `num_prefill_instances` | `1` | Number of prefill GPU instances |
| `num_decode_instances` | `1` | Number of decode GPU instances |
| `gpu_type` | `"A100_40GB"` | GPU profile: `"A100_40GB"` or `"A100_80GB"` |

### Role-switching policy (Alpha V6)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_switching` | `False` | Enable instance role switching |
| `switch_policy` | `"never"` | `"never"`, `"alpha_v6"`, or other variants |
| `switch_min_blocking_time` | `5.0` | Drain period before a switch completes (s) |
| `alpha_v6_threshold_low` | `0.6` | Trigger P→D when alpha < this |
| `alpha_v6_threshold_high` | `1.0` | Trigger D→P when alpha > this |
| `decode_allocatable_low` | `0.2` | P→D guard: avg allocatable ratio below this = decode memory tight |
| `decode_allocatable_high` | `0.6` | D→P guard: avg allocatable ratio above this = decode memory roomy |
| `monitor_interval_s` | `5.0` | Policy evaluation interval (s) |
| `global_cooldown_s` | `30.0` | Minimum time between any switches |

### Partial offloading (LP-based)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_dynamic_lp` | `False` | Enable LP solver for partial KV offloading |
| `slo_target` | `1.0` | TTFT SLO in seconds (LP constraint) |
| `tpot_sla` | `0.1` | TPOT SLA threshold (s); 0.1 = 10 tok/s |
| `itl_sla` | `0.1` | ITL SLA threshold (s) |
| `enable_decode_protection` | `True` | Respect TPOT budget when offloading |
| `budget_scaling_factor` | `1.0` | Multiplier on offload budget (1.0 = baseline) |

### Scheduling

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_prefill_tokens` | `8192` | Max tokens per prefill batch |
| `chunked_prefill_size` | `-1` | `-1` = disabled; `>0` = chunk size |
| `schedule_policy` | `"fcfs"` | `"fcfs"` or `"lpm"` |
| `max_running_requests` | `1000` | Per-instance concurrency cap |
| `bootstrap_timeout_seconds` | `60.0` | Drop requests stuck in bootstrap queue after this |

### Metrics & monitoring

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_streaming_metrics` | `True` | O(1) memory statistics (always recommended) |
| `enable_monitoring` | `True` | Time-series sampling |
| `monitoring_sample_interval` | `10.0` | Sample every N simulation seconds |
| `enable_periodic_plots` | `True` | Generate PNG plots at wall-clock intervals |
| `monitoring_plot_interval_minutes` | `60.0` | Wall-clock minutes between plot renders |
| `enable_iteration_logging` | `False` | Per-instance iteration logs (storage-heavy) |
| `enable_request_trace_logging` | `False` | Per-request trace log (adds ~2.5 GB per run) |

---

## Common Configuration Patterns

```python
from config import SimConfig

# 1. Baseline — fixed roles, no offload
baseline = SimConfig(
    trace_path="my_workload.csv",
    num_prefill_instances=4,
    num_decode_instances=4,
    enable_switching=False,
    enable_dynamic_lp=False,
)

# 2. Switching only (Alpha V6)
switching = SimConfig(
    trace_path="my_workload.csv",
    num_prefill_instances=4,
    num_decode_instances=4,
    enable_switching=True,
    switch_policy="alpha_v6",
    enable_dynamic_lp=False,
)

# 3. Switching + offload (no decode protection)
offload_no_protect = SimConfig(
    trace_path="my_workload.csv",
    num_prefill_instances=4,
    num_decode_instances=4,
    enable_switching=True,
    switch_policy="alpha_v6",
    enable_dynamic_lp=True,
    enable_decode_protection=False,
)

# 4. Switching + offload (with decode protection)
offload_protect = SimConfig(
    trace_path="my_workload.csv",
    num_prefill_instances=4,
    num_decode_instances=4,
    enable_switching=True,
    switch_policy="alpha_v6",
    enable_dynamic_lp=True,
    enable_decode_protection=True,
    tpot_sla=0.1,
    budget_scaling_factor=1.0,
)
```

---

## Results Layout

### Experiment summary (written by every experiment script)

```
experiments/results/YYYYMMDD_HHMMSS_{experiment_name}.json
```

Each file is a JSON array — one object per `SimConfig` variant. Key fields:

| Field | Description |
|-------|-------------|
| `avg_e2e_latency` | Mean end-to-end latency (s) |
| `p50/p95/p99_e2e_latency` | Percentile E2E latency (s) |
| `avg_ttft` / `p99_ttft` | Time to first token (s) |
| `avg_itl` / `p99_itl` | Inter-token latency (s) |
| `throughput` | Completed requests per second |
| `token_throughput` | Output tokens per second |
| `prefill_utilization` / `decode_utilization` | GPU utilization [0, 1] |
| `sla_attainment_rate` | % of requests meeting both TTFT and ITL SLA |
| `ttft_sla_attainment` | % meeting TTFT SLA |
| `itl_sla_attainment` | % meeting ITL SLA |
| `num_switches` | Total role-switch events |
| `num_dropped` | Requests dropped (e.g., bootstrap timeout) |
| `label` | Human-readable config label |
| `config` | Full `SimConfig` snapshot |

### Per-run detailed outputs (when `enable_monitoring=True`)

```
result/YYYYMMDD_HHMMSS_{N}P{M}D_{trace_stem}/
├── config.json                    # Full SimConfig snapshot
├── prefill_00.jsonl               # Per-instance iteration logs (if enabled)
├── decode_00.jsonl
├── request_traces.jsonl           # Per-request trace (if enabled)
└── plots/
    ├── throughput_over_time.png
    ├── queue_metrics_prefill_0.png
    ├── queue_metrics_decode_0.png
    ├── sla_attainment_over_time.png
    ├── memory_usage_over_time.png
    ├── role_switches_timeline.png  # Only when switching is enabled
    └── time_series_samples.jsonl  # Raw sampled data
```

---

## Workflow Tips

- **Quick sanity check**: pass `--test` to any experiment script — runs a single worker on a subset.
- **Memory safety**: `enable_streaming_metrics=True` (default) keeps statistics at O(1) memory regardless of trace length.
- **Disk usage**: keep `enable_request_trace_logging=False` (default) for long runs; enabling it adds ~2.5 GB per simulation.
- **LP performance**: set `enable_parallel_lp_solver=True` to parallelize the LP solver across prefill instances.
- **Debugging latency**: set `enable_iteration_logging=True` to get per-instance step-by-step logs in `result/`.
