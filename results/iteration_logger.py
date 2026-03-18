"""Per-iteration inference statistics logger.

Logs detailed statistics for every batch execution to JSONL files,
split by instance type and device ID.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from request.batch import SimBatch, ForwardMode


class IterationLogger:
    """Logger for per-iteration inference statistics.

    Records detailed batch execution data including per-request metrics.
    Output files are organized in timestamped run directories with config info, e.g.:
        result/20260311_143052_1P1D_b0.3_M5_chunk256/
            prefill_00.jsonl
            prefill_01.jsonl
            decode_00.jsonl
            config.json
    """

    def __init__(
        self,
        output_dir: str = "result",
        config: Any = None,
        num_prefill: int = 1,
        num_decode: int = 1,
    ):
        """Initialize iteration logger.

        Args:
            output_dir: Base directory for result files
            config: SimConfig object (used for run naming and metadata)
            num_prefill: Number of prefill instances
            num_decode: Number of decode instances
        """
        base_dir = Path(output_dir)

        # Build descriptive run directory name
        run_dir_name = self._build_run_name(config, num_prefill, num_decode)
        self.output_dir = base_dir / run_dir_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.run_name = run_dir_name

        # Create per-device file handles: instance_id -> Path
        self.file_map: Dict[str, Path] = {}
        self.count_map: Dict[str, int] = {}

        for i in range(num_prefill):
            instance_id = f"prefill_{i}"
            path = self.output_dir / f"prefill_{i:02d}.jsonl"
            self.file_map[instance_id] = path
            self.count_map[instance_id] = 0
            path.write_text("")

        for i in range(num_decode):
            instance_id = f"decode_{i}"
            path = self.output_dir / f"decode_{i:02d}.jsonl"
            self.file_map[instance_id] = path
            self.count_map[instance_id] = 0
            path.write_text("")

        # Save config to run directory
        if config is not None:
            config_path = self.output_dir / "config.json"
            from dataclasses import asdict
            with open(config_path, 'w') as f:
                json.dump(asdict(config), f, indent=2)

        self.iteration_count = 0

        print(f"IterationLogger initialized:")
        print(f"  Output dir: {self.output_dir}")
        print(f"  Files: {len(self.file_map)} per-device logs")

    @staticmethod
    def _build_run_name(config: Any, num_prefill: int, num_decode: int) -> str:
        """Build a descriptive run directory name from config.

        Format: YYYYMMDD_HHMMSS_{P}P{D}D[_b{beta}][_M{M}][_chunk{size}]
        Examples:
            20260311_143052_1P1D
            20260311_143052_2P2D_b0.3_M5
            20260311_143052_1P1D_chunk256
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        parts = [timestamp, f"{num_prefill}P{num_decode}D"]

        if config is not None:
            # Trace name (without path and extension)
            trace_name = Path(config.trace_path).stem
            parts.append(trace_name)

            if getattr(config, 'enable_dynamic_lp', False):
                parts.append(f"lp_slo{config.slo_target}")
            if getattr(config, 'enable_continuation', False) and getattr(config, 'M', 0) > 0:
                parts.append(f"M{config.M}")
            if getattr(config, 'chunked_prefill_size', -1) > 0:
                parts.append(f"chunk{config.chunked_prefill_size}")

        return "_".join(parts)

    def log_batch_execution(
        self,
        batch: SimBatch,
        instance_id: str,
        instance_type: str,
        start_time: float,
        end_time: float,
        iteration: int
    ):
        """Log a batch execution.

        Args:
            batch: The batch that was executed
            instance_id: ID of the instance (e.g. "prefill_0", "decode_1")
            instance_type: "prefill" or "decode"
            start_time: Batch start timestamp
            end_time: Batch end timestamp
            iteration: Global iteration number
        """
        if batch.is_empty():
            return

        inference_time = end_time - start_time

        # Build per-request data
        request_data = []
        for req in batch.reqs:
            req_info = {
                "rid": req.rid,
                "input_length": req.context_tokens,
                "output_length": len(req.output_ids),
                "extend_input_len": req.extend_input_len,
                "is_prefill": req.decode_tokens_generated == 0,
                "prefix_matched": len(req.prefix_indices),
            }
            request_data.append(req_info)

        record = {
            "iteration": iteration,
            "timestamp": end_time,
            "instance_id": instance_id,
            "instance_type": instance_type,
            "forward_mode": batch.forward_mode.name,
            "inference_time": inference_time,
            "batch_size": len(batch.reqs),
            "total_prefill_tokens": batch.total_prefill_tokens,
            "decode_batch_size": batch.decode_batch_size,
            "decode_computed_token_sum": batch.decode_computed_token_sum,
            "requests": request_data,
        }

        # Write to the per-device file
        output_file = self.file_map.get(instance_id)
        if output_file is None:
            # Fallback: create file on the fly for unknown instance
            idx = instance_id.split("_")[-1]
            output_file = self.output_dir / f"{instance_type}_{int(idx):02d}.jsonl"
            self.file_map[instance_id] = output_file
            self.count_map[instance_id] = 0

        with open(output_file, 'a') as f:
            f.write(json.dumps(record) + '\n')

        self.count_map[instance_id] = self.count_map.get(instance_id, 0) + 1
        self.iteration_count += 1

    def finalize(self):
        """Print summary statistics."""
        print(f"\nIterationLogger Summary:")
        print(f"  Output dir: {self.output_dir}")
        print(f"  Total iterations: {self.iteration_count}")
        for inst_id in sorted(self.count_map.keys()):
            count = self.count_map[inst_id]
            path = self.file_map[inst_id]
            if count > 0:
                size_kb = path.stat().st_size / 1024
                print(f"  {path.name}: {count} iterations ({size_kb:.1f} KB)")
