"""Worker state data structure for policy evaluation.

This module provides a simplified WorkerState class that mimics the production
monitoring interface, allowing policy_alpha and policy_v1 to work in simulation.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class WorkerState:
    """Simulated worker state.

    Attributes:
        port: Worker port number (mapped from instance_id hash)
        role: Worker role ("prefill" or "decode")
        last_parsed: Dict of metrics from the worker
    """
    port: int
    role: str  # "prefill" or "decode"
    last_parsed: Dict[str, Any]


def _get_metric(parsed_dict, metric_name):
    """Extract metric from parsed dict.

    Mimics monitor.metrics._get_metric for simulation.

    Args:
        parsed_dict: Dictionary of parsed metrics
        metric_name: Name of metric to extract

    Returns:
        Metric value or None if not found
    """
    if not isinstance(parsed_dict, dict):
        return None
    return parsed_dict.get(metric_name)


def _idle_for_k_scrapes(worker, k):
    """Check if worker has been idle for k scrapes.

    Mimics monitor.metrics._idle_for_k_scrapes for simulation.
    In simulation, we simplify by checking if currently not busy.

    Args:
        worker: WorkerState instance
        k: Number of scrapes to check

    Returns:
        True if worker is idle (num_running_reqs == 0)
    """
    num_running = _get_metric(worker.last_parsed, "num_running_reqs")
    return num_running == 0 if num_running is not None else False
