"""Configuration for simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class SimConfig:
    """Simulation configuration.

    Attributes:
        trace_path: Path to workload trace CSV
        num_prefill_instances: Number of prefill instances
        num_decode_instances: Number of decode instances

        # New mechanism parameters
        M: Decode continuation length (tokens)
        enable_dynamic_lp: Enable LP-based dynamic beta solver for partial offloading
        slo_target: SLO target in seconds (for LP constraints)
        lp_max_window_size: Max requests in LP sliding window
        enable_continuation: Enable decode continuation
        enable_switching: Enable role switching
        switch_policy: Role switching policy ("never", "load_based", "adaptive")

        # Scheduling parameters
        max_prefill_tokens: Maximum prefill tokens per batch
        schedule_policy: Scheduling policy ("fcfs", "lpm")
    """

    trace_path: str
    num_prefill_instances: int = 1
    num_decode_instances: int = 1

    # New mechanisms
    M: int = 0
    enable_dynamic_lp: bool = False  # Enable LP-based dynamic beta solver for partial offloading
    slo_target: float = 1.0  # SLO target in seconds (for LP constraints)
    lp_max_window_size: int = 5  # Max requests in LP sliding window

    # Decode protection for partial offloading
    tpot_sla: float = 0.1  # 100ms TPOT threshold (10 tokens/sec)
    itl_sla: float = 0.1  # 100ms ITL SLA threshold
    enable_decode_protection: bool = True  # Enable decode TPOT constraint
    budget_scaling_factor: float = 1.0  # Scaling factor for decode offload budget (1.0 = baseline, 2.0 = double budget)

    enable_continuation: bool = False
    enable_switching: bool = False
    switch_policy: str = "never"  # "never", "alpha", "v1", "throughput"
    switch_schedule: List[Dict] = field(default_factory=list)  # Manual switch schedule
    switch_min_blocking_time: float = 5.0  # Minimum blocking period after drain

    # Policy monitor
    monitor_interval_s: float = 5.0  # How often to evaluate policy

    # Alpha policy parameters
    alpha_threshold: float = 1.0
    alpha_threshold_down: float = 0.5
    alpha_allow_decode_to_prefill: bool = True  # Allow decode→prefill switching in alpha policy
    alpha_allow_prefill_to_decode: bool = True  # Allow prefill→decode switching in alpha policy

    # Alpha V2 policy parameters (with decode pressure guard)
    alpha_v2_threshold: float = 1.0
    alpha_v2_allow_decode_to_prefill: bool = True
    alpha_v2_allow_prefill_to_decode: bool = True

    # Alpha V3 policy parameters (with decode memory guard)
    alpha_v3_threshold_low: float = 0.6   # Below 0.6 → alpha low (decode bottleneck)
    alpha_v3_threshold_high: float = 1.0  # Above 1.0 → alpha high (prefill bottleneck)
    alpha_v3_allow_decode_to_prefill: bool = True
    alpha_v3_allow_prefill_to_decode: bool = True
    decode_memory_low: float = 0.4   # Below 40% usage → decode underutilized
    decode_memory_high: float = 0.8  # Above 80% usage → decode overloaded

    # Alpha V4 policy parameters (with average decode memory guard and p95 alpha)
    alpha_v4_threshold_low: float = 0.6   # Below 0.6 → alpha low (decode bottleneck)
    alpha_v4_threshold_high: float = 1.0  # Above 1.0 → alpha high (prefill bottleneck)
    alpha_v4_allow_decode_to_prefill: bool = True
    alpha_v4_allow_prefill_to_decode: bool = True

    # Alpha V5 policy parameters (with Kalman Filter prediction)
    alpha_v5_threshold_low: float = 0.6   # Below 0.6 → alpha low (decode bottleneck)
    alpha_v5_threshold_high: float = 1.0  # Above 1.0 → alpha high (prefill bottleneck)
    alpha_v5_allow_decode_to_prefill: bool = True
    alpha_v5_allow_prefill_to_decode: bool = True

    # Kalman Filter parameters (for Alpha V5)
    kf_process_noise: float = 0.01         # Process noise covariance (lower = smoother)
    kf_measurement_noise: float = 0.1      # Measurement noise variance (higher = less trust)
    kf_prediction_steps: int = 3           # Prediction horizon (3 steps = 15s with 5s interval)
    kf_velocity_threshold: float = 0.05    # Velocity threshold for adaptive stable_evals
    kf_min_stable_evals: int = 2           # Minimum stable_evals when velocity is high

    # V1 policy parameters
    prefill_pressure_high: float = 10.0
    prefill_pressure_low: float = 2.0
    decode_pressure_high: float = 12.0  # Prefill→Decode: need very high decode pressure
    decode_pressure_low: float = 6.0    # Decode→Prefill: need relatively low decode pressure
    prefill_wait_weight: float = 1.0
    prefill_active_weight: float = 0.5
    prefill_inflight_weight: float = 0.3
    decode_prealloc_weight: float = 1.0
    decode_transfer_weight: float = 0.8
    decode_active_weight: float = 0.5
    decode_prefill_prealloc_weight: float = 0.3

    # Common policy parameters
    stable_evals: int = 5
    global_cooldown_s: float = 30.0
    per_worker_cooldown_s: float = 60.0
    min_prefill_instances: int = 1
    min_decode_instances: int = 1
    idle_scrapes: int = 2

    # Scheduling
    max_prefill_tokens: int = 8192
    chunked_prefill_size: int = -1  # -1 = disabled, >0 = max tokens per chunk
    enable_mixed_chunk: bool = False
    schedule_conservativeness: float = 0  # Scales new_token_ratio
    schedule_policy: str = "fcfs"
    max_running_requests: int = 1000 # Max concurrent running requests per instance
    num_reserved_decode_tokens: int = 512  # Per-request decode token reservation for bootstrap gating
    bootstrap_timeout_seconds: float = 60.0  # Timeout for requests stuck in bootstrap queue (reduced from 120s for faster cleanup)
    bootstrap_timeout_check_interval: float = 10.0  # How often to check for bootstrap timeouts (seconds)

    # Metrics and monitoring
    enable_streaming_metrics: bool = True  # Use O(1) memory streaming statistics
    metrics_reservoir_size: int = 10000  # Reservoir size for percentile sampling
    enable_monitoring: bool = True  # Enable time-series monitoring
    monitoring_sample_interval: float = 10.0  # Sample interval in simulation seconds (10s for better performance)
    monitoring_max_samples: int = 10000  # Max samples to keep in memory
    monitoring_sla_window: int = 1000  # Rolling window size for SLA calculation
    enable_iteration_logging: bool = False  # Enable per-instance iteration logging (can be memory/storage intensive)
    enable_request_trace_logging: bool = False  # Enable per-request trace logging (default OFF for large traces to save 2.5GB+ disk + I/O)

    # Periodic plot generation (wall-clock time)
    enable_periodic_plots: bool = True  # Enable periodic plot generation during simulation
    monitoring_plot_interval_minutes: float = 60.0  # Wall-clock interval for plot generation (60min for less overhead)

    # Multi-threading optimizations (opt-in for performance)
    enable_parallel_lp_solver: bool = False  # Parallelize LP solver across prefill instances
    lp_solver_max_workers: int = -1  # Thread pool size (-1 = num_prefill_instances, >0 = explicit)
    enable_async_logging: bool = False  # Use async I/O for iteration logging
    enable_parallel_policy_eval: bool = False  # Parallelize policy state collection

    # Streaming workload loading (for large traces >100K requests to avoid OOM)
    enable_streaming_loading: bool = False  # Enable streaming CSV loading
    streaming_window_size: float = 300.0  # Time window size in seconds (5 minutes)
    streaming_lookback: float = 60.0  # Lookback buffer in seconds (1 minute safety margin)

    # Memory profiling (to debug OOM issues)
    enable_memory_profiling: bool = False  # Enable detailed memory profiling
    memory_profiling_interval: float = 60.0  # Snapshot interval in simulation seconds
