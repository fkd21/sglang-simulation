"""Time-series monitoring for simulation with memory-efficient sampling."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


class TimeSeriesMonitor:
    """Memory-efficient time-series monitoring with visualization."""

    def __init__(
        self,
        sample_interval: float = 1.0,
        max_samples: int = 10000,
        sla_window_size: int = 1000,
        ttft_sla: float = 1.0,
        itl_sla: float = 0.1,
        output_dir: str = "result",
        run_name: str = None,
    ):
        """Initialize time-series monitor.

        Args:
            sample_interval: Time between samples in simulation seconds (default: 1.0)
            max_samples: Maximum samples to keep in memory (default: 10000)
            sla_window_size: Rolling window size for SLA calculation (default: 1000 requests)
            ttft_sla: TTFT SLA threshold in seconds
            itl_sla: ITL SLA threshold in seconds
            output_dir: Base directory for output files
            run_name: Run name for the output directory (e.g., "20260319_123456_2P2D_azure_code_500")
        """
        self.sample_interval = sample_interval
        self.max_samples = max_samples
        self.sla_window_size = sla_window_size
        self.ttft_sla = ttft_sla
        self.itl_sla = itl_sla

        # Create output directory (can be updated later by engine)
        if run_name:
            self.output_dir = Path(output_dir) / run_name / "plots"
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Ring buffer for samples (O(1) memory)
        self.samples: deque = deque(maxlen=max_samples)
        self.next_sample_time = 0.0

        # Track role switches for plot markers
        self.role_switches: List[Dict] = []

        # Track recent completions for SLA rolling window
        self.recent_completions: deque = deque(maxlen=sla_window_size)

        # Throughput tracking (for interval calculation)
        self.interval_prefill_tokens = 0
        self.interval_decode_tokens = 0
        self.last_sample_time = 0.0

        # Output file for samples (periodic writes)
        self.samples_file = self.output_dir / "time_series_samples.jsonl"
        self.samples_file_handle = None

    def maybe_sample(
        self, current_time: float, engine_state: Dict[str, Any]
    ) -> None:
        """Sample system state if interval has elapsed.

        Args:
            current_time: Current simulation time
            engine_state: Dictionary containing engine state to sample
        """
        if current_time >= self.next_sample_time:
            sample = self._collect_sample(current_time, engine_state)
            self.samples.append(sample)
            self._write_sample_to_disk(sample)
            self.next_sample_time = current_time + self.sample_interval

            # Reset interval counters
            self.interval_prefill_tokens = 0
            self.interval_decode_tokens = 0
            self.last_sample_time = current_time

    def record_throughput(self, prefill_tokens: int, decode_tokens: int):
        """Record throughput for current interval.

        Args:
            prefill_tokens: Prefill tokens processed
            decode_tokens: Decode tokens generated
        """
        self.interval_prefill_tokens += prefill_tokens
        self.interval_decode_tokens += decode_tokens

    def record_role_switch(
        self, time: float, instance_id: str, from_role: str, to_role: str
    ):
        """Record role switch for plot markers.

        Args:
            time: When the switch completed
            instance_id: Instance that switched
            from_role: Original role
            to_role: New role
        """
        self.role_switches.append({
            'time': time,
            'instance_id': instance_id,
            'from_role': from_role,
            'to_role': to_role,
        })

    def record_completion(self, req, timestamp: float):
        """Track completion for SLA window calculation.

        Args:
            req: Completed request
            timestamp: Completion timestamp
        """
        # Calculate TTFT and ITL
        ttft = 0.0
        itl = 0.0

        if req.prefill_end_time > 0:
            ttft = req.prefill_end_time - req.arrival_time

        if req.decode_tokens_generated > 0 and req.decode_end_time > req.decode_start_time:
            itl = (req.decode_end_time - req.decode_start_time) / req.decode_tokens_generated

        self.recent_completions.append({
            'timestamp': timestamp,
            'ttft': ttft,
            'itl': itl,
            'ttft_sla_met': ttft <= self.ttft_sla if ttft > 0 else False,
            'itl_sla_met': itl <= self.itl_sla if itl > 0 else False,
        })

    def _collect_sample(
        self, current_time: float, engine_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Collect a single sample of system state.

        Args:
            current_time: Current simulation time
            engine_state: Engine state dictionary

        Returns:
            Sample dictionary
        """
        # Calculate interval throughput
        interval_duration = current_time - self.last_sample_time if self.last_sample_time > 0 else self.sample_interval
        input_throughput = self.interval_prefill_tokens / interval_duration if interval_duration > 0 else 0
        output_throughput = self.interval_decode_tokens / interval_duration if interval_duration > 0 else 0

        # Calculate SLA attainment from rolling window
        sla_metrics = self._calculate_sla_metrics()

        # Get dropped request count from metrics collector
        metrics_collector = engine_state.get('metrics_collector')
        num_dropped = len(metrics_collector.dropped_requests) if metrics_collector else 0
        dropped_breakdown = {}
        if metrics_collector:
            for drop_record in metrics_collector.dropped_requests:
                reason = drop_record.get('reason', 'unknown')
                dropped_breakdown[reason] = dropped_breakdown.get(reason, 0) + 1

        # Collect queue metrics for each instance
        instance_metrics = {}
        for instance in engine_state.get('all_instances', []):
            instance_metrics[instance.instance_id] = {
                'instance_type': instance.instance_type.name,
                'waiting_queue': len(instance.waiting_queue) if hasattr(instance, 'waiting_queue') else 0,
                'running_batch': len(instance.running_batch.reqs) if hasattr(instance, 'running_batch') and instance.running_batch else 0,
                'bootstrap_queue': len(instance.bootstrap_queue) if hasattr(instance, 'bootstrap_queue') else 0,
                'transfer_queue': len(instance.transfer_queue) if hasattr(instance, 'transfer_queue') else 0,
                'prealloc_queue': len(instance.prealloc_queue) if hasattr(instance, 'prealloc_queue') else 0,
                'prealloc_reserved': len(instance.prealloc_reserved) if hasattr(instance, 'prealloc_reserved') else 0,
                # Memory utilization
                'kv_cache_used': (instance.token_to_kv_pool.total_kv_tokens - instance.token_to_kv_pool.available_size()) if hasattr(instance, 'token_to_kv_pool') else 0,
                'kv_cache_total': instance.token_to_kv_pool.total_kv_tokens if hasattr(instance, 'token_to_kv_pool') else 1,
            }

        return {
            'time': current_time,
            'input_throughput': input_throughput,
            'output_throughput': output_throughput,
            'sla_attainment': sla_metrics,
            'num_dropped': num_dropped,
            'dropped_breakdown': dropped_breakdown,
            'instances': instance_metrics,
        }

    def _calculate_sla_metrics(self) -> Dict[str, float]:
        """Calculate SLA metrics from rolling window.

        Returns:
            Dictionary with SLA attainment rates
        """
        if not self.recent_completions:
            return {'ttft': 0.0, 'itl': 0.0, 'overall': 0.0}

        ttft_met = sum(1 for c in self.recent_completions if c['ttft_sla_met'])
        itl_met = sum(1 for c in self.recent_completions if c['itl_sla_met'])
        both_met = sum(1 for c in self.recent_completions if c['ttft_sla_met'] and c['itl_sla_met'])

        total = len(self.recent_completions)
        return {
            'ttft': (ttft_met / total * 100) if total > 0 else 0.0,
            'itl': (itl_met / total * 100) if total > 0 else 0.0,
            'overall': (both_met / total * 100) if total > 0 else 0.0,
        }

    def _write_sample_to_disk(self, sample: Dict[str, Any]):
        """Write sample to JSONL file.

        Args:
            sample: Sample dictionary
        """
        if self.samples_file_handle is None:
            self.samples_file_handle = open(self.samples_file, 'w')

        self.samples_file_handle.write(json.dumps(sample) + '\n')
        self.samples_file_handle.flush()

    def finalize(self):
        """Close file handles."""
        if self.samples_file_handle:
            self.samples_file_handle.close()
            self.samples_file_handle = None

    def generate_plots(self):
        """Generate all visualization plots."""
        if not self.samples:
            print("No samples collected, skipping plot generation")
            return

        print(f"Generating time-series plots from {len(self.samples)} samples...")

        try:
            self._plot_throughput()
            self._plot_queue_metrics_with_switches()
            self._plot_sla_attainment()
            self._plot_memory_usage()
            self._plot_dropped_requests()
            print(f"Plots saved to {self.output_dir}/")
        except Exception as e:
            print(f"Error generating plots: {e}")
            import traceback
            traceback.print_exc()

    def _plot_throughput(self):
        """Plot input and output throughput over time."""
        times = [s['time'] for s in self.samples]
        input_tp = [s['input_throughput'] for s in self.samples]
        output_tp = [s['output_throughput'] for s in self.samples]

        plt.figure(figsize=(12, 6))
        plt.plot(times, input_tp, label='Input Throughput (prefill tokens/s)', linewidth=2)
        plt.plot(times, output_tp, label='Output Throughput (decode tokens/s)', linewidth=2)
        plt.xlabel('Simulation Time (s)', fontsize=12)
        plt.ylabel('Throughput (tokens/s)', fontsize=12)
        plt.title('System Throughput Over Time', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'throughput_over_time.png', dpi=150)
        plt.close()

    def _plot_queue_metrics_with_switches(self):
        """Plot queue metrics for each instance with role switch markers."""
        # Group samples by instance
        instance_ids = set()
        for sample in self.samples:
            instance_ids.update(sample['instances'].keys())

        for instance_id in sorted(instance_ids):
            times = []
            waiting_queue = []
            running = []
            bootstrap_queue = []
            transfer_queue = []
            prealloc_queue = []
            prealloc_reserved = []

            for sample in self.samples:
                if instance_id in sample['instances']:
                    inst = sample['instances'][instance_id]
                    times.append(sample['time'])
                    waiting_queue.append(inst['waiting_queue'])
                    running.append(inst['running_batch'])
                    bootstrap_queue.append(inst['bootstrap_queue'])
                    transfer_queue.append(inst['transfer_queue'])
                    prealloc_queue.append(inst['prealloc_queue'])
                    prealloc_reserved.append(inst['prealloc_reserved'])

            if not times:
                continue

            fig, ax = plt.subplots(figsize=(14, 7))

            # Plot queue metrics
            ax.plot(times, waiting_queue, label='waiting queue', linewidth=1.5)
            ax.plot(times, running, label='running', linewidth=1.5)

            if max(bootstrap_queue) > 0:
                ax.plot(times, bootstrap_queue, label='bootstrap queue', linewidth=1.5)
            if max(transfer_queue) > 0:
                ax.plot(times, transfer_queue, label='transfer queue', linewidth=1.5)
            if max(prealloc_queue) > 0:
                ax.plot(times, prealloc_queue, label='prealloc queue', linewidth=1.5)
            if max(prealloc_reserved) > 0:
                ax.plot(times, prealloc_reserved, label='prealloc reserved', linewidth=1.5)

            # Add role switch markers
            for switch in self.role_switches:
                if switch['instance_id'] == instance_id:
                    ax.axvline(x=switch['time'], color='red',
                              linestyle='--', alpha=0.7, linewidth=2)
                    # Add annotation
                    label = f"{switch['from_role'][0]}→{switch['to_role'][0]}"
                    ax.annotate(label, xy=(switch['time'], ax.get_ylim()[1] * 0.95),
                               xytext=(0, -10), textcoords='offset points',
                               ha='center', fontsize=10, color='red', weight='bold',
                               bbox=dict(boxstyle='round,pad=0.4',
                                       facecolor='yellow', alpha=0.7, edgecolor='red'))

            ax.set_xlabel('Simulation Time (s)', fontsize=12)
            ax.set_ylabel('Queue Length / Batch Size', fontsize=12)
            ax.set_title(f'Queue Metrics - {instance_id}', fontsize=14)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.output_dir / f'queue_metrics_{instance_id}.png', dpi=150)
            plt.close()

    def _plot_sla_attainment(self):
        """Plot SLA attainment over time (rolling window)."""
        times = [s['time'] for s in self.samples]
        ttft_sla = [s['sla_attainment']['ttft'] for s in self.samples]
        itl_sla = [s['sla_attainment']['itl'] for s in self.samples]
        overall_sla = [s['sla_attainment']['overall'] for s in self.samples]

        plt.figure(figsize=(12, 6))
        plt.plot(times, ttft_sla, label='TTFT SLA Attainment', linewidth=2)
        plt.plot(times, itl_sla, label='ITL SLA Attainment', linewidth=2)
        plt.plot(times, overall_sla, label='Overall SLA Attainment (both)', linewidth=2, linestyle='--')
        plt.xlabel('Simulation Time (s)', fontsize=12)
        plt.ylabel('SLA Attainment (%)', fontsize=12)
        plt.title(f'SLA Attainment Over Time ({self.sla_window_size}-request rolling window)', fontsize=14)
        plt.ylim(0, 105)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'sla_attainment_over_time.png', dpi=150)
        plt.close()

    def _plot_memory_usage(self):
        """Plot memory (KV cache) utilization over time."""
        # Group by instance
        instance_ids = set()
        for sample in self.samples:
            instance_ids.update(sample['instances'].keys())

        plt.figure(figsize=(12, 6))

        for instance_id in sorted(instance_ids):
            times = []
            utilizations = []

            for sample in self.samples:
                if instance_id in sample['instances']:
                    inst = sample['instances'][instance_id]
                    times.append(sample['time'])
                    used = inst['kv_cache_used']
                    total = inst['kv_cache_total']
                    util = (used / total * 100) if total > 0 else 0
                    utilizations.append(util)

            if times:
                # Get instance type from last sample
                inst_type = sample['instances'][instance_id]['instance_type']
                linestyle = '-' if inst_type == 'PREFILL' else '--'
                plt.plot(times, utilizations, label=f'{instance_id} ({inst_type})',
                        linewidth=1.5, linestyle=linestyle)

        plt.xlabel('Simulation Time (s)', fontsize=12)
        plt.ylabel('KV Cache Utilization (%)', fontsize=12)
        plt.title('Memory Usage Over Time', fontsize=14)
        plt.ylim(0, 105)
        plt.legend(fontsize=9, ncol=2)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'memory_usage_over_time.png', dpi=150)
        plt.close()

    def _plot_dropped_requests(self):
        """Plot cumulative dropped request count over time."""
        times = [s['time'] for s in self.samples]
        num_dropped = [s.get('num_dropped', 0) for s in self.samples]

        # Get breakdown by reason from last sample
        last_sample = self.samples[-1] if self.samples else {}
        breakdown = last_sample.get('dropped_breakdown', {})

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Left plot: cumulative dropped count over time
        ax1.plot(times, num_dropped, label='Total Dropped Requests',
                linewidth=2, color='red')
        ax1.set_xlabel('Simulation Time (s)', fontsize=12)
        ax1.set_ylabel('Cumulative Dropped Requests', fontsize=12)
        ax1.set_title('Dropped Requests Over Time', fontsize=14)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Right plot: pie chart of drop reasons
        if breakdown and sum(breakdown.values()) > 0:
            reasons = list(breakdown.keys())
            counts = list(breakdown.values())
            colors = plt.cm.Set3(range(len(reasons)))

            ax2.pie(counts, labels=reasons, autopct='%1.1f%%',
                   colors=colors, startangle=90)
            ax2.set_title(f'Drop Reasons (Total: {sum(counts)})', fontsize=14)
        else:
            ax2.text(0.5, 0.5, 'No dropped requests',
                    ha='center', va='center', fontsize=14)
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
            ax2.axis('off')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'dropped_requests_over_time.png', dpi=150)
        plt.close()
