"""Trace loader for Azure LLM inference traces and JSONL specs."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import List

from request.request import SimReq


class JsonlTraceLoader:
    """Loads requests from JSONL spec files.

    Expected JSONL format (one JSON object per line):
        {"input_len": 1, "output_len": 2312}

    All requests arrive at time 0.
    """

    def __init__(self, trace_path: str):
        self.trace_path = Path(trace_path)
        if not self.trace_path.exists():
            raise FileNotFoundError(f"Trace file not found: {trace_path}")

    def load(self) -> List[SimReq]:
        requests = []
        with open(self.trace_path, 'r') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                req = SimReq(
                    rid=f"req_{idx}",
                    arrival_time=0.0,
                    context_tokens=entry["input_len"],
                    generated_tokens=entry["output_len"]
                )
                requests.append(req)

        print(f"Loaded {len(requests)} requests from {self.trace_path.name}")
        print(f"All requests arrive at t=0.0s")
        print(f"Avg context tokens: {sum(r.context_tokens for r in requests) / len(requests):.1f}")
        print(f"Avg generated tokens: {sum(r.generated_tokens for r in requests) / len(requests):.1f}")
        return requests


class TraceLoader:
    """Loads Azure LLM inference traces from CSV.

    Expected CSV format:
    TIMESTAMP,ContextTokens,GeneratedTokens

    Example:
    2023-11-16 18:17:03,4808,10
    2023-11-16 18:17:04,110,27
    """

    def __init__(self, trace_path: str):
        """Initialize trace loader.

        Args:
            trace_path: Path to the CSV trace file
        """
        self.trace_path = Path(trace_path)
        if not self.trace_path.exists():
            raise FileNotFoundError(f"Trace file not found: {trace_path}")

    def load(self) -> List[SimReq]:
        """Load trace and create SimReq objects.

        Returns:
            List of SimReq objects with arrival times in seconds from start
        """
        requests = []
        start_time = None

        with open(self.trace_path, 'r') as f:
            reader = csv.DictReader(f)

            for idx, row in enumerate(reader):
                # Parse timestamp (handles microseconds with 7 digits)
                timestamp_str = row['TIMESTAMP'].strip()

                # Azure timestamps have 7-digit microseconds, Python supports 6
                if '.' in timestamp_str:
                    parts = timestamp_str.split('.')
                    # Truncate microseconds to 6 digits
                    microseconds = parts[1][:6]
                    timestamp_str = f"{parts[0]}.{microseconds}"
                    timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f')
                else:
                    timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')

                # Set start time from first request
                if start_time is None:
                    start_time = timestamp

                # Calculate arrival time in seconds from start
                arrival_time = (timestamp - start_time).total_seconds()

                # Parse token counts
                context_tokens = int(row['ContextTokens'])
                generated_tokens = int(row['GeneratedTokens'])

                # Create request
                req = SimReq(
                    rid=f"req_{idx}",
                    arrival_time=arrival_time,
                    context_tokens=context_tokens,
                    generated_tokens=generated_tokens
                )
                requests.append(req)

        print(f"Loaded {len(requests)} requests from {self.trace_path.name}")
        print(f"Time range: 0.0s to {requests[-1].arrival_time:.2f}s")
        print(f"Avg context tokens: {sum(r.context_tokens for r in requests) / len(requests):.1f}")
        print(f"Avg generated tokens: {sum(r.generated_tokens for r in requests) / len(requests):.1f}")

        return requests


class WorkloadDriver:
    """Drives workload by converting trace to REQUEST_ARRIVAL events."""

    def __init__(self, trace_path: str):
        """Initialize workload driver.

        Args:
            trace_path: Path to the trace file (CSV or JSONL)
        """
        path = Path(trace_path)
        if path.suffix == '.jsonl':
            self.loader = JsonlTraceLoader(trace_path)
        else:
            self.loader = TraceLoader(trace_path)
        self.requests = []

    def load_trace(self) -> List[SimReq]:
        """Load trace requests.

        Returns:
            List of SimReq objects
        """
        self.requests = self.loader.load()
        return self.requests

    def get_requests(self) -> List[SimReq]:
        """Get loaded requests.

        Returns:
            List of SimReq objects
        """
        return self.requests
