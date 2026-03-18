"""Metrics collection for simulation."""

from __future__ import annotations

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
        }


class MetricsCollector:
    """Collects metrics during simulation."""

    def __init__(self, ttft_sla: float = 1.0, itl_sla: float = 0.05):
        """Initialize metrics collector.

        Args:
            ttft_sla: TTFT SLA threshold in seconds (default: 1.0s)
            itl_sla: ITL SLA threshold in seconds (default: 0.05s = 50ms)
        """
        self.arrivals: List[Tuple[float, str]] = []
        self.completions: List[Tuple[float, str, float]] = []
        self.ttft_values: List[float] = []
        self.itl_values: List[float] = []

        # SLA thresholds
        self.ttft_sla = ttft_sla
        self.itl_sla = itl_sla

        # SLA tracking
        self.ttft_sla_met: List[bool] = []
        self.itl_sla_met: List[bool] = []

        # System metrics
        self.prefill_busy_time = 0.0
        self.decode_busy_time = 0.0

        # New mechanism counters
        self.num_offloaded = 0
        self.num_continued = 0
        self.num_switches = 0

        # Role switching details
        self.role_switches: List[Dict] = []

    def record_arrival(self, req: SimReq, timestamp: float):
        """Record request arrival.

        Args:
            req: Request that arrived
            timestamp: Arrival timestamp
        """
        self.arrivals.append((timestamp, req.rid))

    def record_completion(self, req: SimReq, timestamp: float):
        """Record request completion.

        Args:
            req: Completed request
            timestamp: Completion timestamp
        """
        e2e_latency = timestamp - req.arrival_time
        self.completions.append((timestamp, req.rid, e2e_latency))

        # TTFT = prefill_end_time - arrival_time
        ttft = 0.0
        if req.prefill_end_time > 0:
            ttft = req.prefill_end_time - req.arrival_time
            self.ttft_values.append(ttft)
            # Check TTFT SLA
            self.ttft_sla_met.append(ttft <= self.ttft_sla)

        # ITL = (decode_end_time - decode_start_time) / num_tokens
        itl = 0.0
        if req.decode_tokens_generated > 0 and req.decode_end_time > req.decode_start_time:
            itl = (req.decode_end_time - req.decode_start_time) / req.decode_tokens_generated
            self.itl_values.append(itl)
            # Check ITL SLA
            self.itl_sla_met.append(itl <= self.itl_sla)

        # Record mechanism usage
        if req.partial_offload_amount > 0:
            self.num_offloaded += 1
        if req.continued_on_prefill > 0:
            self.num_continued += 1

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
        if not self.completions:
            # No completions, return empty results
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
            )

        # Extract latencies
        e2e_latencies = [lat for _, _, lat in self.completions]

        # Compute percentiles
        def safe_percentile(values, p):
            return float(np.percentile(values, p)) if values else 0.0

        # Throughput
        throughput = len(self.completions) / total_time if total_time > 0 else 0.0

        # Token throughput (estimated as throughput * avg tokens per request)
        token_throughput = throughput * 100  # Rough estimate for now

        # Utilization
        max_prefill_time = total_time * num_prefill
        max_decode_time = total_time * num_decode
        prefill_util = self.prefill_busy_time / max_prefill_time if max_prefill_time > 0 else 0.0
        decode_util = self.decode_busy_time / max_decode_time if max_decode_time > 0 else 0.0

        # SLA attainment rates
        ttft_sla_attainment = (sum(self.ttft_sla_met) / len(self.ttft_sla_met) * 100) if self.ttft_sla_met else 0.0
        itl_sla_attainment = (sum(self.itl_sla_met) / len(self.itl_sla_met) * 100) if self.itl_sla_met else 0.0

        # Overall SLA: both TTFT and ITL must meet SLA
        if self.ttft_sla_met and self.itl_sla_met:
            both_met = [ttft and itl for ttft, itl in zip(self.ttft_sla_met, self.itl_sla_met)]
            sla_attainment_rate = (sum(both_met) / len(both_met) * 100) if both_met else 0.0
        else:
            sla_attainment_rate = 0.0

        return SimulationResults(
            total_requests=len(self.completions),
            total_simulation_time=total_time,
            avg_e2e_latency=float(np.mean(e2e_latencies)),
            p50_e2e_latency=safe_percentile(e2e_latencies, 50),
            p95_e2e_latency=safe_percentile(e2e_latencies, 95),
            p99_e2e_latency=safe_percentile(e2e_latencies, 99),
            avg_ttft=float(np.mean(self.ttft_values)) if self.ttft_values else 0.0,
            p50_ttft=safe_percentile(self.ttft_values, 50),
            p99_ttft=safe_percentile(self.ttft_values, 99),
            avg_itl=float(np.mean(self.itl_values)) if self.itl_values else 0.0,
            p50_itl=safe_percentile(self.itl_values, 50),
            p99_itl=safe_percentile(self.itl_values, 99),
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
        )
