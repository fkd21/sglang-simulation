"""Metrics collection for simulation."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from request.request import SimReq


@dataclass
class SimulationResults:
    """Simulation results and metrics."""

    # Basic metrics
    total_requests: int
    total_simulation_time: float

    # Latency metrics (seconds)
    avg_e2e_latency: float
    p50_e2e_latency: float
    p95_e2e_latency: float
    p99_e2e_latency: float

    avg_ttft: float  # Time to first token
    p50_ttft: float
    p99_ttft: float

    avg_itl: float  # Inter-token latency
    p50_itl: float
    p99_itl: float

    # Throughput
    throughput: float  # requests per second
    token_throughput: float  # tokens per second

    # Instance utilization
    prefill_utilization: float
    decode_utilization: float

    # SLA attainment
    sla_attainment_rate: float  # Percentage meeting both TTFT and ITL SLAs
    ttft_sla_attainment: float  # Percentage meeting TTFT SLA
    itl_sla_attainment: float  # Percentage meeting ITL SLA

    # New mechanism metrics
    num_offloaded: int = 0
    num_continued: int = 0
    num_switches: int = 0
    role_switches: List[Dict] = field(default_factory=list)

    # Dropped requests metrics (SGLang-style)
    num_dropped: int = 0
    dropped_breakdown: Dict[str, int] = field(default_factory=dict)  # reason -> count

    def to_dict(self) -> dict:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            'total_requests': self.total_requests,
            'total_simulation_time': self.total_simulation_time,
            'avg_e2e_latency': self.avg_e2e_latency,
            'p50_e2e_latency': self.p50_e2e_latency,
            'p95_e2e_latency': self.p95_e2e_latency,
            'p99_e2e_latency': self.p99_e2e_latency,
            'avg_ttft': self.avg_ttft,
            'p50_ttft': self.p50_ttft,
            'p99_ttft': self.p99_ttft,
            'avg_itl': self.avg_itl,
            'p50_itl': self.p50_itl,
            'p99_itl': self.p99_itl,
            'throughput': self.throughput,
            'token_throughput': self.token_throughput,
            'prefill_utilization': self.prefill_utilization,
            'decode_utilization': self.decode_utilization,
            'sla_attainment_rate': self.sla_attainment_rate,
            'ttft_sla_attainment': self.ttft_sla_attainment,
            'itl_sla_attainment': self.itl_sla_attainment,
            'num_offloaded': self.num_offloaded,
            'num_continued': self.num_continued,
            'num_switches': self.num_switches,
            'role_switches': self.role_switches,
            'num_dropped': self.num_dropped,
            'dropped_breakdown': self.dropped_breakdown,
        }


class MetricsCollector:
    """Collects metrics during simulation with memory-efficient streaming."""

    def __init__(self, ttft_sla: float = 1.0, itl_sla: float = 0.1,
                 enable_streaming: bool = True, reservoir_size: int = 10000):
        """Initialize metrics collector.

        Args:
            ttft_sla: TTFT SLA threshold in seconds (default: 1.0s)
            itl_sla: ITL SLA threshold in seconds (default: 0.1s = 100ms)
            enable_streaming: Use streaming statistics to save memory (default: True)
            reservoir_size: Size of reservoir for percentile sampling (default: 10000)
        """
        self.enable_streaming = enable_streaming
        self.reservoir_size = reservoir_size

        # Counters (O(1) memory)
        self.num_arrivals = 0
        self.num_completions = 0
        self.first_arrival_time = None
        self.last_completion_time = None

        # SLA thresholds
        self.ttft_sla = ttft_sla
        self.itl_sla = itl_sla

        if enable_streaming:
            # Streaming approach: O(1) memory
            # TTFT statistics
            self.ttft_count = 0
            self.ttft_mean = 0.0
            self.ttft_m2 = 0.0  # For variance calculation (Welford's algorithm)
            self.ttft_reservoir = []  # Reservoir sampling for percentiles

            # ITL statistics
            self.itl_count = 0
            self.itl_mean = 0.0
            self.itl_m2 = 0.0
            self.itl_reservoir = []

            # E2E latency statistics
            self.e2e_count = 0
            self.e2e_mean = 0.0
            self.e2e_m2 = 0.0
            self.e2e_reservoir = []

            # SLA tracking (streaming)
            self.ttft_sla_met_count = 0
            self.itl_sla_met_count = 0
            self.both_sla_met_count = 0
        else:
            # Legacy approach: O(N) memory (for small simulations)
            self.arrivals: List[Tuple[float, str]] = []
            self.completions: List[Tuple[float, str, float]] = []
            self.ttft_values: List[float] = []
            self.itl_values: List[float] = []
            self.ttft_sla_met: List[bool] = []
            self.itl_sla_met: List[bool] = []

        # System metrics
        self.prefill_busy_time = 0.0
        self.decode_busy_time = 0.0

        # New mechanism counters
        self.num_offloaded = 0
        self.num_continued = 0
        self.num_switches = 0

        # Role switching details (keep as-is, usually small)
        self.role_switches: List[Dict] = []

        # Dropped requests tracking (SGLang-style)
        self.dropped_requests: List[Dict] = []
        self.dropped_breakdown_cache: Dict[str, int] = {}  # Incremental cache for O(1) breakdown

    def record_arrival(self, req: SimReq, timestamp: float):
        """Record request arrival.

        Args:
            req: Request that arrived
            timestamp: Arrival timestamp
        """
        self.num_arrivals += 1
        if self.first_arrival_time is None:
            self.first_arrival_time = timestamp

        if not self.enable_streaming:
            self.arrivals.append((timestamp, req.rid))

    def record_completion(self, req: SimReq, timestamp: float):
        """Record request completion.

        Args:
            req: Completed request
            timestamp: Completion timestamp
        """
        e2e_latency = timestamp - req.arrival_time
        self.num_completions += 1
        self.last_completion_time = timestamp

        # TTFT = prefill_end_time - arrival_time
        ttft = 0.0
        ttft_sla_met = False
        if req.prefill_end_time > 0:
            ttft = req.prefill_end_time - req.arrival_time
            ttft_sla_met = ttft <= self.ttft_sla

        # ITL = (decode_end_time - decode_start_time) / num_tokens
        itl = 0.0
        itl_sla_met = False
        if req.decode_tokens_generated > 0 and req.decode_end_time > req.decode_start_time:
            itl = (req.decode_end_time - req.decode_start_time) / req.decode_tokens_generated
            itl_sla_met = itl <= self.itl_sla

        if self.enable_streaming:
            # Update streaming statistics
            self._update_streaming_stats(e2e_latency, ttft, itl, ttft_sla_met, itl_sla_met)
        else:
            # Legacy: store all values
            self.completions.append((timestamp, req.rid, e2e_latency))
            if req.prefill_end_time > 0:
                self.ttft_values.append(ttft)
                self.ttft_sla_met.append(ttft_sla_met)
            if req.decode_tokens_generated > 0 and req.decode_end_time > req.decode_start_time:
                self.itl_values.append(itl)
                self.itl_sla_met.append(itl_sla_met)

        # Record mechanism usage
        if req.partial_offload_amount > 0:
            self.num_offloaded += 1
        if req.continued_on_prefill > 0:
            self.num_continued += 1

    def record_dropped_request(self, req: SimReq, reason: str):
        """Record a request that was dropped/aborted.

        SGLang-style drop tracking. Reasons include:
        - "oversized": Request context_tokens exceeds max capacity
        - "bootstrap_timeout": Request stuck in bootstrap queue too long

        Args:
            req: Request that was dropped
            reason: Drop reason string
        """
        self.dropped_requests.append({
            'request_id': req.rid,  # Use 'rid' field
            'reason': reason,
            'arrival_time': req.arrival_time,
            'context_tokens': req.context_tokens,
            'generated_tokens': req.generated_tokens,
        })
        # Update incremental cache for O(1) breakdown retrieval
        self.dropped_breakdown_cache[reason] = self.dropped_breakdown_cache.get(reason, 0) + 1

    def _update_streaming_stats(self, e2e_latency: float, ttft: float, itl: float,
                                ttft_sla_met: bool, itl_sla_met: bool):
        """Update streaming statistics using Welford's algorithm and reservoir sampling.

        Args:
            e2e_latency: End-to-end latency
            ttft: Time to first token
            itl: Inter-token latency
            ttft_sla_met: Whether TTFT met SLA
            itl_sla_met: Whether ITL met SLA
        """
        # Update E2E latency statistics
        self.e2e_count += 1
        delta = e2e_latency - self.e2e_mean
        self.e2e_mean += delta / self.e2e_count
        delta2 = e2e_latency - self.e2e_mean
        self.e2e_m2 += delta * delta2
        self._add_to_reservoir(self.e2e_reservoir, e2e_latency, self.e2e_count)

        # Update TTFT statistics
        if ttft > 0:
            self.ttft_count += 1
            delta = ttft - self.ttft_mean
            self.ttft_mean += delta / self.ttft_count
            delta2 = ttft - self.ttft_mean
            self.ttft_m2 += delta * delta2
            self._add_to_reservoir(self.ttft_reservoir, ttft, self.ttft_count)
            if ttft_sla_met:
                self.ttft_sla_met_count += 1

        # Update ITL statistics
        if itl > 0:
            self.itl_count += 1
            delta = itl - self.itl_mean
            self.itl_mean += delta / self.itl_count
            delta2 = itl - self.itl_mean
            self.itl_m2 += delta * delta2
            self._add_to_reservoir(self.itl_reservoir, itl, self.itl_count)
            if itl_sla_met:
                self.itl_sla_met_count += 1

        # Both SLA met
        if ttft_sla_met and itl_sla_met:
            self.both_sla_met_count += 1

    def _add_to_reservoir(self, reservoir: List[float], value: float, count: int):
        """Add value to reservoir using Algorithm R (reservoir sampling).

        Args:
            reservoir: Reservoir list to update
            value: New value to add
            count: Total count of values seen so far
        """
        if len(reservoir) < self.reservoir_size:
            reservoir.append(value)
        else:
            # Randomly replace with decreasing probability
            j = random.randint(0, count - 1)
            if j < self.reservoir_size:
                reservoir[j] = value

    def record_prefill_busy(self, duration: float):
        """Record prefill instance busy time.

        Args:
            duration: Time spent busy
        """
        self.prefill_busy_time += duration

    def record_decode_busy(self, duration: float):
        """Record decode instance busy time.

        Args:
            duration: Time spent busy
        """
        self.decode_busy_time += duration

    def record_switch(self, time: float, instance_id: str,
                     from_role: str, to_role: str, drain_time: float):
        """Record role switch event.

        Args:
            time: When the switch completed
            instance_id: Instance that switched
            from_role: Original role
            to_role: New role
            drain_time: How long the drain took
        """
        self.num_switches += 1
        self.role_switches.append({
            "time": time,
            "instance_id": instance_id,
            "from_role": from_role,
            "to_role": to_role,
            "drain_time": drain_time,
        })

    def finalize(self, total_time: float, num_prefill: int, num_decode: int) -> SimulationResults:
        """Finalize metrics and compute statistics.

        Args:
            total_time: Total simulation time
            num_prefill: Number of prefill instances
            num_decode: Number of decode instances

        Returns:
            Simulation results
        """
        # Compute dropped request statistics (even if no completions)
        num_dropped = len(self.dropped_requests)
        dropped_breakdown = {}
        for drop_record in self.dropped_requests:
            reason = drop_record['reason']
            dropped_breakdown[reason] = dropped_breakdown.get(reason, 0) + 1

        if self.num_completions == 0:
            # No completions, return empty results (but include dropped stats)
            return SimulationResults(
                total_requests=0,
                total_simulation_time=total_time,
                avg_e2e_latency=0.0,
                p50_e2e_latency=0.0,
                p95_e2e_latency=0.0,
                p99_e2e_latency=0.0,
                avg_ttft=0.0,
                p50_ttft=0.0,
                p99_ttft=0.0,
                avg_itl=0.0,
                p50_itl=0.0,
                p99_itl=0.0,
                throughput=0.0,
                token_throughput=0.0,
                prefill_utilization=0.0,
                decode_utilization=0.0,
                sla_attainment_rate=0.0,
                ttft_sla_attainment=0.0,
                itl_sla_attainment=0.0,
                role_switches=[],
                num_dropped=num_dropped,
                dropped_breakdown=dropped_breakdown,
            )

        # Compute percentiles
        def safe_percentile(values, p):
            return float(np.percentile(values, p)) if values else 0.0

        if self.enable_streaming:
            # Use streaming statistics
            # E2E latency
            avg_e2e = self.e2e_mean
            p50_e2e = safe_percentile(self.e2e_reservoir, 50)
            p95_e2e = safe_percentile(self.e2e_reservoir, 95)
            p99_e2e = safe_percentile(self.e2e_reservoir, 99)

            # TTFT
            avg_ttft = self.ttft_mean if self.ttft_count > 0 else 0.0
            p50_ttft = safe_percentile(self.ttft_reservoir, 50)
            p99_ttft = safe_percentile(self.ttft_reservoir, 99)

            # ITL
            avg_itl = self.itl_mean if self.itl_count > 0 else 0.0
            p50_itl = safe_percentile(self.itl_reservoir, 50)
            p99_itl = safe_percentile(self.itl_reservoir, 99)

            # SLA attainment rates
            # IMPORTANT: Dropped requests count as SLA violations (they are included in denominator)
            total_requests_served = self.ttft_count + num_dropped
            ttft_sla_attainment = (self.ttft_sla_met_count / total_requests_served * 100) if total_requests_served > 0 else 0.0

            total_requests_with_decode = self.itl_count + num_dropped
            itl_sla_attainment = (self.itl_sla_met_count / total_requests_with_decode * 100) if total_requests_with_decode > 0 else 0.0

            # Overall SLA (both must meet) - dropped requests count as violations
            total_for_overall_sla = min(self.ttft_count, self.itl_count) + num_dropped
            sla_attainment_rate = (self.both_sla_met_count / total_for_overall_sla * 100) if total_for_overall_sla > 0 else 0.0
        else:
            # Legacy: use stored lists
            e2e_latencies = [lat for _, _, lat in self.completions]
            avg_e2e = float(np.mean(e2e_latencies))
            p50_e2e = safe_percentile(e2e_latencies, 50)
            p95_e2e = safe_percentile(e2e_latencies, 95)
            p99_e2e = safe_percentile(e2e_latencies, 99)

            avg_ttft = float(np.mean(self.ttft_values)) if self.ttft_values else 0.0
            p50_ttft = safe_percentile(self.ttft_values, 50)
            p99_ttft = safe_percentile(self.ttft_values, 99)

            avg_itl = float(np.mean(self.itl_values)) if self.itl_values else 0.0
            p50_itl = safe_percentile(self.itl_values, 50)
            p99_itl = safe_percentile(self.itl_values, 99)

            # IMPORTANT: Dropped requests count as SLA violations (they are included in denominator)
            total_requests_served = len(self.ttft_sla_met) + num_dropped
            ttft_sla_attainment = (sum(self.ttft_sla_met) / total_requests_served * 100) if total_requests_served > 0 else 0.0

            total_requests_with_decode = len(self.itl_sla_met) + num_dropped
            itl_sla_attainment = (sum(self.itl_sla_met) / total_requests_with_decode * 100) if total_requests_with_decode > 0 else 0.0

            if self.ttft_sla_met and self.itl_sla_met:
                both_met = [ttft and itl for ttft, itl in zip(self.ttft_sla_met, self.itl_sla_met)]
                total_for_overall_sla = len(both_met) + num_dropped
                sla_attainment_rate = (sum(both_met) / total_for_overall_sla * 100) if total_for_overall_sla > 0 else 0.0
            else:
                sla_attainment_rate = 0.0

        # Throughput
        throughput = self.num_completions / total_time if total_time > 0 else 0.0

        # Token throughput (estimated as throughput * avg tokens per request)
        token_throughput = throughput * 100  # Rough estimate for now

        # Utilization
        max_prefill_time = total_time * num_prefill
        max_decode_time = total_time * num_decode
        prefill_util = self.prefill_busy_time / max_prefill_time if max_prefill_time > 0 else 0.0
        decode_util = self.decode_busy_time / max_decode_time if max_decode_time > 0 else 0.0

        # Dropped stats already computed at top of function

        return SimulationResults(
            total_requests=self.num_completions,
            total_simulation_time=total_time,
            avg_e2e_latency=avg_e2e,
            p50_e2e_latency=p50_e2e,
            p95_e2e_latency=p95_e2e,
            p99_e2e_latency=p99_e2e,
            avg_ttft=avg_ttft,
            p50_ttft=p50_ttft,
            p99_ttft=p99_ttft,
            avg_itl=avg_itl,
            p50_itl=p50_itl,
            p99_itl=p99_itl,
            throughput=throughput,
            token_throughput=token_throughput,
            prefill_utilization=prefill_util,
            decode_utilization=decode_util,
            sla_attainment_rate=sla_attainment_rate,
            ttft_sla_attainment=ttft_sla_attainment,
            itl_sla_attainment=itl_sla_attainment,
            num_offloaded=self.num_offloaded,
            num_continued=self.num_continued,
            num_switches=self.num_switches,
            role_switches=self.role_switches,
            num_dropped=num_dropped,
            dropped_breakdown=dropped_breakdown,
        )
