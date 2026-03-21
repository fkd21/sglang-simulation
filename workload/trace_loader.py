"""Trace loader for Azure LLM inference traces and JSONL specs."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

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


class StreamingTraceLoader:
    """Streams Azure LLM inference traces from CSV without loading entire file.

    Yields requests in time-ordered chunks for windowed loading.
    Designed to handle large traces (millions of requests) without OOM.
    """

    def __init__(self, trace_path: str):
        """Initialize streaming trace loader.

        Args:
            trace_path: Path to the CSV trace file
        """
        self.trace_path = Path(trace_path)
        if not self.trace_path.exists():
            raise FileNotFoundError(f"Trace file not found: {trace_path}")

        self._file_handle = None
        self._csv_reader = None
        self._start_time = None
        self._last_row_index = -1
        self._peek_buffer: Optional[Tuple[SimReq, float]] = None  # (request, arrival_time)
        self._trace_exhausted = False

    def open(self):
        """Open CSV file and initialize reader."""
        if self._file_handle is not None:
            return  # Already open

        self._file_handle = open(self.trace_path, 'r')
        self._csv_reader = csv.DictReader(self._file_handle)

    def close(self):
        """Close CSV file handle."""
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None
            self._csv_reader = None

    def _parse_row(self, row: dict) -> Tuple[SimReq, float]:
        """Parse CSV row to (SimReq, arrival_time) tuple.

        Args:
            row: Dict from CSV DictReader

        Returns:
            Tuple of (SimReq object, arrival_time in seconds)
        """
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
        if self._start_time is None:
            self._start_time = timestamp

        # Calculate arrival time in seconds from start
        arrival_time = (timestamp - self._start_time).total_seconds()

        # Parse token counts
        context_tokens = int(row['ContextTokens'])
        generated_tokens = int(row['GeneratedTokens'])

        # Create request
        self._last_row_index += 1
        req = SimReq(
            rid=f"req_{self._last_row_index}",
            arrival_time=arrival_time,
            context_tokens=context_tokens,
            generated_tokens=generated_tokens
        )

        return req, arrival_time

    def stream_window(self, start_time: float, end_time: float) -> Iterator[SimReq]:
        """Stream requests with arrival times in [start_time, end_time).

        Args:
            start_time: Window start (seconds from trace start)
            end_time: Window end (seconds from trace start)

        Yields:
            SimReq objects with arrival_time in window
        """
        if self._file_handle is None:
            raise RuntimeError("Must call open() before stream_window()")

        # First, check peek buffer (request read beyond previous window)
        if self._peek_buffer is not None:
            req, arrival_time = self._peek_buffer
            self._peek_buffer = None

            if start_time <= arrival_time < end_time:
                yield req
            elif arrival_time >= end_time:
                # Put it back in peek buffer
                self._peek_buffer = (req, arrival_time)
                return
            # else: arrival_time < start_time, skip it (shouldn't happen in normal usage)

        # Stream from CSV
        if self._trace_exhausted:
            return

        try:
            for row in self._csv_reader:
                req, arrival_time = self._parse_row(row)

                # Stop if beyond window
                if arrival_time >= end_time:
                    self._peek_buffer = (req, arrival_time)
                    return

                # Skip if before window
                if arrival_time < start_time:
                    continue

                yield req
        except StopIteration:
            pass

        # Mark trace as exhausted
        self._trace_exhausted = True

    def get_statistics(self) -> dict:
        """Get loading statistics.

        Returns:
            dict with metadata about loaded trace
        """
        return {
            "start_time": self._start_time,
            "last_row_index": self._last_row_index,
            "trace_exhausted": self._trace_exhausted
        }


class StreamingWorkloadDriver:
    """Drives workload using streaming lazy loading with time windows.

    Loads requests in configurable time chunks (default 5 minutes) with lookback buffer.
    Dynamically injects REQUEST_ARRIVAL events as simulation progresses.
    Designed to handle millions of requests without OOM by keeping only current window in memory.
    """

    def __init__(self, trace_path: str, window_size: float = 300.0, lookback: float = 60.0):
        """Initialize streaming workload driver.

        Args:
            trace_path: Path to trace CSV
            window_size: Time window size in seconds (default: 5 minutes = 300s)
            lookback: Lookback buffer in seconds (default: 1 minute = 60s)
        """
        self.trace_path = trace_path
        self.window_size = window_size
        self.lookback = lookback

        # Streaming loader
        path = Path(trace_path)
        if path.suffix == '.jsonl':
            raise NotImplementedError("Streaming JSONL not yet implemented. Use CSV traces.")
        else:
            self.loader = StreamingTraceLoader(trace_path)

        # Window tracking
        self.current_window_start = 0.0
        self.current_window_end = window_size
        self.loaded_requests: List[SimReq] = []  # Requests in current window
        self.loaded_until = 0.0  # Time up to which we've loaded
        self.trace_exhausted = False

        # Statistics
        self.total_requests_loaded = 0

    def open(self):
        """Open trace file."""
        self.loader.open()

    def close(self):
        """Close trace file."""
        self.loader.close()

    def load_initial_window(self) -> List[SimReq]:
        """Load first window of requests.

        Returns:
            List of SimReq objects in [0, window_size)
        """
        requests = list(self.loader.stream_window(0.0, self.window_size))
        self.loaded_requests = requests
        self.loaded_until = self.window_size
        self.total_requests_loaded += len(requests)

        print(f"[StreamingLoader] Loaded initial window: {len(requests)} requests [0.0s, {self.window_size}s)")
        return requests

    def should_load_next_window(self, current_time: float) -> bool:
        """Check if we should load the next window.

        Load next window when simulation time reaches (loaded_until - lookback).
        This ensures we never run out of requests to schedule.

        Args:
            current_time: Current simulation time

        Returns:
            True if next window should be loaded
        """
        if self.trace_exhausted:
            return False

        # Prefetch threshold: load next window when we're within lookback of boundary
        prefetch_threshold = self.loaded_until - self.lookback
        return current_time >= prefetch_threshold

    def load_next_window(self) -> List[SimReq]:
        """Load next time window of requests.

        Returns:
            List of SimReq objects in next window
        """
        if self.trace_exhausted:
            return []

        # Update window boundaries
        self.current_window_start = self.loaded_until
        self.current_window_end = self.current_window_start + self.window_size

        # Stream next window
        requests = list(self.loader.stream_window(
            self.current_window_start,
            self.current_window_end
        ))

        self.loaded_requests = requests
        self.loaded_until = self.current_window_end
        self.total_requests_loaded += len(requests)

        if self.loader._trace_exhausted:
            self.trace_exhausted = True
            print(f"[StreamingLoader] Loaded final window: {len(requests)} requests [{self.current_window_start:.1f}s, {self.current_window_end:.1f}s)")
            print(f"[StreamingLoader] Trace exhausted. Total requests loaded: {self.total_requests_loaded}")
        else:
            print(f"[StreamingLoader] Loaded next window: {len(requests)} requests [{self.current_window_start:.1f}s, {self.current_window_end:.1f}s)")

        return requests

    def cleanup_old_window(self):
        """Clear old window's request list to free memory.

        Note: Individual requests are freed via cleanup_after_completion() when they finish.
        This just clears our reference list to allow GC.
        """
        self.loaded_requests.clear()

    def get_statistics(self) -> dict:
        """Get loading statistics.

        Returns:
            dict with loading progress info
        """
        return {
            "total_loaded": self.total_requests_loaded,
            "loaded_until": self.loaded_until,
            "trace_exhausted": self.trace_exhausted,
            "window_size": self.window_size,
            "lookback": self.lookback
        }
