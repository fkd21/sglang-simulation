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
    enable_continuation: bool = False
    enable_switching: bool = False
    switch_policy: str = "never"  # "never", "alpha", "v1"
    switch_schedule: List[Dict] = field(default_factory=list)  # Manual switch schedule
    switch_min_blocking_time: float = 5.0  # Minimum blocking period after drain

    # Policy monitor
    monitor_interval_s: float = 5.0  # How often to evaluate policy

    # Alpha policy parameters
    alpha_threshold: float = 1.0
    alpha_threshold_down: float = 0.5

    # V1 policy parameters
    prefill_pressure_high: float = 10.0
    prefill_pressure_low: float = 2.0
    decode_pressure_high: float = 10.0
    decode_pressure_low: float = 2.0
    prefill_wait_weight: float = 1.0
    prefill_active_weight: float = 0.5
    prefill_inflight_weight: float = 0.3
    decode_prealloc_weight: float = 1.0
    decode_transfer_weight: float = 0.8
    decode_active_weight: float = 0.5
    decode_prefill_prealloc_weight: float = 0.3

    # Common policy parameters
    stable_evals: int = 3
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
