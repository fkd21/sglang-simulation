# Multi-threading and Monitoring Optimization Features

This document describes the multi-threading and monitoring optimization features added to the SGLang simulation framework.

## Overview

Two main categories of enhancements:
1. **Periodic Plot Generation**: Automatic visualization updates during long simulations
2. **Multi-threading Optimizations**: Parallel execution to reduce wall-clock simulation time

## 1. Periodic Plot Generation

### Feature Description
Automatically generates monitoring plots at regular wall-clock intervals during simulation execution, allowing you to monitor progress without waiting for completion.

### Configuration

```python
# In config.py or SimConfig
enable_periodic_plots: bool = True  # Enable feature (default: True)
monitoring_plot_interval_minutes: float = 15.0  # Update interval in minutes
```

### How It Works
- Background timer thread runs independently of simulation
- Every N minutes (wall-clock time), regenerates all plots
- Plot files are **overwritten** (same filenames, minimal disk usage)
- Thread-safe implementation with proper lifecycle management

### Plot Files Generated
All plots are saved to `result/{run_name}/plots/`:
- `throughput_over_time.png` - Input/output throughput
- `queue_metrics_*.png` - Per-instance queue lengths
- `sla_attainment_over_time.png` - TTFT/ITL SLA compliance
- `memory_usage_over_time.png` - KV cache utilization
- `dropped_requests_over_time.png` - Drop statistics

### Usage Example

```python
config = SimConfig(
    trace_path="azure_code_24h.csv",
    num_prefill_instances=4,
    num_decode_instances=4,
    enable_periodic_plots=True,
    monitoring_plot_interval_minutes=10.0,  # Update every 10 minutes
)

engine = SimulationEngine(config)
results = engine.run()  # Plots auto-update during execution
```

### Benefits
- **Progress monitoring**: See simulation progress in real-time
- **Early debugging**: Identify issues before simulation completes
- **Minimal overhead**: Background thread has negligible performance impact
- **Low disk usage**: Files overwritten, not accumulated

### Testing
```bash
python test_periodic_plots.py
```

## 2. Multi-threading Optimizations

### 2.1 Parallel LP Solver (Priority 1)

#### Feature Description
Parallelizes LP solver execution across multiple prefill instances when partial offloading is enabled.

#### Configuration (Opt-in)

```python
# In config.py or SimConfig
enable_parallel_lp_solver: bool = False  # Disabled by default (opt-in)
lp_solver_max_workers: int = -1  # -1 = auto (num_prefill_instances)
```

#### How It Works
- Uses `ThreadPoolExecutor` to solve LP problems in parallel
- Applies when: `enable_dynamic_lp=True` AND `>=2 prefill instances`
- Each prefill instance's LP problem is independent
- Results collected and applied before batch formation
- Falls back to sequential if thread pool not available

#### Performance Characteristics

**When beneficial:**
- ≥4 prefill instances
- Dynamic LP enabled with high queue depths
- Long-running simulations

**Expected speedup:**
- 1.1-1.5x for small traces (overhead dominates)
- 2-4x for medium traces
- Up to 8x for large traces with many prefill instances

**When NOT beneficial:**
- ≤2 prefill instances (thread overhead > savings)
- Low queue depths (LP solve time < thread overhead)
- Traces with few requests

#### Usage Example

```python
config = SimConfig(
    trace_path="azure_code_1week.csv",
    num_prefill_instances=8,
    num_decode_instances=8,
    enable_dynamic_lp=True,
    enable_parallel_lp_solver=True,  # Enable parallel LP
    lp_solver_max_workers=8,  # Or -1 for auto
)

engine = SimulationEngine(config)
results = engine.run()
```

#### Testing
```bash
python test_parallel_lp.py
```

### 2.2 Async Logging (Not Yet Implemented)

**Status**: Configuration parameters added, implementation pending

```python
enable_async_logging: bool = False  # Future feature
```

### 2.3 Parallel Policy Evaluation (Not Yet Implemented)

**Status**: Configuration parameters added, implementation pending

```python
enable_parallel_policy_eval: bool = False  # Future feature
```

## Configuration Summary

All new parameters in `config.py`:

```python
# Periodic plot generation (wall-clock time)
enable_periodic_plots: bool = True
monitoring_plot_interval_minutes: float = 15.0

# Multi-threading optimizations (opt-in)
enable_parallel_lp_solver: bool = False
lp_solver_max_workers: int = -1  # -1 = auto
enable_async_logging: bool = False  # Not yet implemented
enable_parallel_policy_eval: bool = False  # Not yet implemented
```

## Design Principles

### Thread Safety
- **LP Solver**: Each instance has independent waiting queue, no shared writes
- **Plot Generation**: Read-only access to samples deque (thread-safe)
- **Lifecycle Management**: Proper thread startup/shutdown

### Backward Compatibility
- **Default behavior unchanged**: Multi-threading features disabled by default
- **Opt-in design**: Users must explicitly enable parallel features
- **Periodic plots enabled by default**: Low risk, high value

### Performance Considerations
- **LP solver overhead**: ThreadPoolExecutor overhead ~1-5ms
- **Profiling recommended**: Measure actual speedup for your workload
- **GIL impact**: Python GIL may limit benefits, but LP solver is computational

### Correctness Guarantees
- **Discrete event semantics preserved**: Parallelism within handlers only
- **Results reproducibility**: Threading affects wall-clock time, not simulation outcomes
- **Minimal code changes**: Leverages existing infrastructure

## Verification

### Test Scripts
1. `test_periodic_plots.py` - Verify periodic plotting works correctly
2. `test_parallel_lp.py` - Compare sequential vs parallel LP solver

### Running Tests

```bash
# Test periodic plots (should see plots generated every 15 seconds)
python test_periodic_plots.py

# Test parallel LP solver (compares performance and correctness)
python test_parallel_lp.py
```

### Integration Testing

```bash
# Run existing experiment with new features enabled
python experiments/run_partial_offload_experiments.py --enable-parallel-lp
```

## Future Work

### Remaining Implementation Tasks
1. **Async Logging**: Queue-based async writer for iteration logging
2. **Parallel Policy Evaluation**: Parallelize state collection in policy monitor
3. **Event Loop Partitioning**: Per-instance worker threads (complex, high risk)

### Potential Enhancements
1. **Adaptive threading**: Automatically enable/disable based on workload
2. **Performance profiling**: Built-in metrics for thread pool utilization
3. **GPU acceleration**: Offload LP solver to GPU for very large windows

## Troubleshooting

### Periodic Plots Not Generating
- Check `enable_periodic_plots=True` in config
- Verify `enable_monitoring=True` (required for time-series data)
- Check console output for "[TimeSeriesMonitor] Started periodic plot generation"

### Parallel LP Slower Than Sequential
- Expected for small traces (overhead > benefit)
- Try with larger traces (≥8K requests) and more prefill instances (≥4)
- Monitor CPU usage - if cores not utilized, GIL may be limiting

### Thread Pool Warnings
- "LP solver thread pool shut down" should appear at end of simulation
- If missing, thread pool may not have cleaned up properly
- Check for exceptions in console output

## References

### Commits
1. `8066597` - Add time-series monitoring, streaming metrics
2. `42905d0` - Add periodic plot generation with wall-clock timer
3. `d817f13` - Add parallel LP solver execution

### Related Files
- `config.py` - Configuration parameters
- `core/engine.py` - Main simulation loop and parallel LP integration
- `metrics/time_series_monitor.py` - Periodic plotting implementation
- `mechanisms/partial_offload.py` - LP solver implementation

## Contact

For questions or issues, please file an issue at the project repository.
