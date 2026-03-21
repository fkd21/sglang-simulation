"""Event system for discrete-event simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict

# Global counter for FIFO ordering of equal-priority events
_event_seq = 0


def _next_seq() -> int:
    global _event_seq
    _event_seq += 1
    return _event_seq


class EventType(Enum):
    """Types of events in the simulation."""

    REQUEST_ARRIVAL = auto()
    PREFILL_BATCH_START = auto()
    PREFILL_BATCH_COMPLETE = auto()
    KV_TRANSFER_START = auto()
    KV_TRANSFER_COMPLETE = auto()
    DECODE_BATCH_START = auto()
    DECODE_BATCH_COMPLETE = auto()
    REQUEST_COMPLETE = auto()
    # Decode continuation on prefill instance
    PREFILL_DECODE_START = auto()
    PREFILL_DECODE_COMPLETE = auto()
    # Partial offload: prefill portion on decode instance
    OFFLOAD_PREFILL_START = auto()
    OFFLOAD_PREFILL_COMPLETE = auto()
    # Role switching
    ROLE_SWITCH = auto()
    SWITCH_UNBLOCK = auto()
    MONITOR_EVAL = auto()  # Periodic policy monitor evaluation
    # Streaming workload loading
    LOAD_WINDOW_CHECK = auto()  # Check if next time window should be loaded


@dataclass(order=True)
class Event:
    """Simulation event with timestamp and data.

    Events are ordered by timestamp, then priority, then insertion order (seq)
    to guarantee deterministic FIFO processing of simultaneous events.
    """

    timestamp: float
    event_type: EventType = field(compare=False)
    priority: int = field(default=0, compare=True)
    seq: int = field(default_factory=_next_seq, compare=True)
    data: Dict[str, Any] = field(default_factory=dict, compare=False)

    def __repr__(self) -> str:
        return (
            f"Event(t={self.timestamp:.6f}, "
            f"type={self.event_type.name}, "
            f"data_keys={list(self.data.keys())})"
        )
