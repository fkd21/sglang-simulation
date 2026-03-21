"""Memory profiling utilities for simulation debugging."""

import gc
import os
import sys
import tracemalloc
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class MemorySnapshot:
    """Memory snapshot at a point in time."""

    timestamp: float  # Simulation time
    rss_mb: float  # Resident set size in MB
    vms_mb: float  # Virtual memory size in MB
    python_mb: float  # Python heap size in MB
    num_objects: int  # Total Python objects

    # Breakdown by type
    top_types: Dict[str, int]  # type_name -> count

    # Custom tracking
    event_queue_size: int = 0
    num_active_requests: int = 0
    num_completed_requests: int = 0


class MemoryProfiler:
    """Tracks memory usage during simulation to identify leaks and hotspots.

    Usage:
        profiler = MemoryProfiler(enabled=True, interval=60.0)
        profiler.start()

        # During simulation
        profiler.snapshot(current_time, event_queue, metrics)

        # At end
        profiler.report()
    """

    def __init__(self, enabled: bool = True, interval: float = 60.0, max_snapshots: int = 500):
        """Initialize memory profiler.

        Args:
            enabled: Whether to enable profiling (default: True)
            interval: Snapshot interval in simulation seconds (default: 60s)
            max_snapshots: Maximum snapshots to keep (default: 500)
        """
        self.enabled = enabled
        self.interval = interval
        self.max_snapshots = max_snapshots

        self.snapshots: List[MemorySnapshot] = []
        self.last_snapshot_time = -interval  # Force first snapshot at t=0

        self._tracemalloc_started = False

    def start(self):
        """Start memory tracking."""
        if not self.enabled:
            return

        # Start tracemalloc to track Python memory allocations
        if not tracemalloc.is_tracing():
            tracemalloc.start()
            self._tracemalloc_started = True
            print("[MemoryProfiler] Started tracemalloc")

        print(f"[MemoryProfiler] Enabled (interval={self.interval}s, max_snapshots={self.max_snapshots})")

    def should_snapshot(self, current_time: float) -> bool:
        """Check if we should take a snapshot now.

        Args:
            current_time: Current simulation time

        Returns:
            True if snapshot should be taken
        """
        if not self.enabled:
            return False

        return current_time - self.last_snapshot_time >= self.interval

    def snapshot(self, current_time: float, event_queue=None, metrics=None):
        """Take a memory snapshot.

        Args:
            current_time: Current simulation time
            event_queue: Event queue (to track size)
            metrics: MetricsCollector (to track request counts)
        """
        if not self.enabled:
            return

        # Check if we should snapshot
        if not self.should_snapshot(current_time):
            return

        self.last_snapshot_time = current_time

        # Get process memory (RSS and VMS)
        try:
            import psutil
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            rss_mb = mem_info.rss / 1024 / 1024
            vms_mb = mem_info.vms / 1024 / 1024
        except ImportError:
            rss_mb = 0.0
            vms_mb = 0.0

        # Get Python heap memory
        python_mb = 0.0
        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            python_mb = current / 1024 / 1024

        # Count objects by type
        gc.collect()  # Force GC to get accurate counts
        type_counts = defaultdict(int)
        for obj in gc.get_objects():
            type_counts[type(obj).__name__] += 1

        # Get top 10 types
        top_types = dict(sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:10])

        # Create snapshot
        snapshot = MemorySnapshot(
            timestamp=current_time,
            rss_mb=rss_mb,
            vms_mb=vms_mb,
            python_mb=python_mb,
            num_objects=sum(type_counts.values()),
            top_types=top_types,
            event_queue_size=len(event_queue) if event_queue else 0,
            num_active_requests=0,  # Filled by caller if needed
            num_completed_requests=metrics.num_completions if metrics and hasattr(metrics, 'num_completions') else 0,
        )

        # Add to history
        self.snapshots.append(snapshot)

        # Limit history size
        if len(self.snapshots) > self.max_snapshots:
            self.snapshots.pop(0)

        # Print summary
        if len(self.snapshots) % 10 == 0:  # Every 10 snapshots
            self._print_summary(snapshot)

    def _print_summary(self, snapshot: MemorySnapshot):
        """Print memory summary."""
        print(f"\n[MemoryProfiler @ t={snapshot.timestamp:.0f}s]")
        print(f"  RSS: {snapshot.rss_mb:.1f} MB | Python heap: {snapshot.python_mb:.1f} MB")
        print(f"  Objects: {snapshot.num_objects:,} | Events: {snapshot.event_queue_size:,}")
        print(f"  Completed: {snapshot.num_completed_requests:,}")

        # Show top object types
        print("  Top types:")
        for i, (type_name, count) in enumerate(list(snapshot.top_types.items())[:5], 1):
            print(f"    {i}. {type_name}: {count:,}")

    def report(self, output_path: Optional[str] = None):
        """Generate final memory report.

        Args:
            output_path: Optional path to save CSV report
        """
        if not self.enabled or not self.snapshots:
            print("[MemoryProfiler] No snapshots to report")
            return

        print("\n" + "="*80)
        print("MEMORY PROFILING REPORT")
        print("="*80)

        # Summary statistics
        start = self.snapshots[0]
        end = self.snapshots[-1]
        peak_rss = max(s.rss_mb for s in self.snapshots)
        peak_python = max(s.python_mb for s in self.snapshots)

        print(f"\nSimulation time: {start.timestamp:.0f}s -> {end.timestamp:.0f}s")
        print(f"Snapshots taken: {len(self.snapshots)}")
        print(f"\nMemory (RSS):")
        print(f"  Start: {start.rss_mb:.1f} MB")
        print(f"  End: {end.rss_mb:.1f} MB")
        print(f"  Peak: {peak_rss:.1f} MB")
        print(f"  Growth: {end.rss_mb - start.rss_mb:+.1f} MB ({(end.rss_mb/start.rss_mb - 1)*100:+.1f}%)")

        print(f"\nPython heap:")
        print(f"  Start: {start.python_mb:.1f} MB")
        print(f"  End: {end.python_mb:.1f} MB")
        print(f"  Peak: {peak_python:.1f} MB")
        print(f"  Growth: {end.python_mb - start.python_mb:+.1f} MB")

        print(f"\nObject count:")
        print(f"  Start: {start.num_objects:,}")
        print(f"  End: {end.num_objects:,}")
        print(f"  Growth: {end.num_objects - start.num_objects:+,}")

        # Detect memory leaks (continuous growth)
        if len(self.snapshots) >= 10:
            recent = self.snapshots[-10:]
            growth_rate = (recent[-1].rss_mb - recent[0].rss_mb) / len(recent)
            if growth_rate > 1.0:  # >1 MB/snapshot
                print(f"\n⚠️  WARNING: Possible memory leak detected!")
                print(f"   Recent growth rate: {growth_rate:.2f} MB/snapshot")

        # Top memory-consuming types at end
        print(f"\nTop object types at end:")
        for i, (type_name, count) in enumerate(list(end.top_types.items())[:10], 1):
            print(f"  {i:2d}. {type_name:30s} {count:>10,}")

        # Save to CSV if requested
        if output_path:
            self._save_csv(output_path)
            print(f"\n✓ Saved detailed report to {output_path}")

        print("="*80 + "\n")

    def _save_csv(self, output_path: str):
        """Save snapshots to CSV."""
        import csv

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'rss_mb', 'vms_mb', 'python_mb', 'num_objects',
                'event_queue_size', 'num_completed', 'top_type', 'top_type_count'
            ])

            for s in self.snapshots:
                top_type = list(s.top_types.keys())[0] if s.top_types else ''
                top_count = list(s.top_types.values())[0] if s.top_types else 0

                writer.writerow([
                    s.timestamp,
                    s.rss_mb,
                    s.vms_mb,
                    s.python_mb,
                    s.num_objects,
                    s.event_queue_size,
                    s.num_completed_requests,
                    top_type,
                    top_count,
                ])

    def stop(self):
        """Stop memory tracking."""
        if self._tracemalloc_started and tracemalloc.is_tracing():
            tracemalloc.stop()
            print("[MemoryProfiler] Stopped tracemalloc")


def get_current_memory_mb() -> float:
    """Get current process memory usage in MB.

    Returns:
        RSS memory in MB, or 0 if psutil not available
    """
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0


def print_memory_summary(label: str = ""):
    """Print quick memory summary.

    Args:
        label: Optional label for this measurement
    """
    mem_mb = get_current_memory_mb()
    if mem_mb > 0:
        print(f"[Memory{': ' + label if label else ''}] {mem_mb:.1f} MB RSS")
