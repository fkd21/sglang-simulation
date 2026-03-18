"""Policy search over mechanism parameters.

Runs grid search over SLO targets, M, and switching policies to find optimal
configurations for a given workload trace.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import List, Optional

from config import SimConfig
from core.engine import SimulationEngine
from metrics.metrics_collector import SimulationResults


@dataclass
class SearchConfig:
    """Configuration for a single search point."""

    enable_dynamic_lp: bool = False
    slo_target: float = 1.0
    M: int = 0
    switch_policy: str = "never"
    num_prefill: int = 1
    num_decode: int = 1
    chunked_prefill_size: int = -1

    def label(self) -> str:
        parts = [f"{self.num_prefill}P{self.num_decode}D"]
        if self.enable_dynamic_lp:
            parts.append(f"lp_slo{self.slo_target}")
        if self.M > 0:
            parts.append(f"M{self.M}")
        if self.switch_policy != "never":
            parts.append(f"sw_{self.switch_policy}")
        if self.chunked_prefill_size > 0:
            parts.append(f"ch{self.chunked_prefill_size}")
        return "_".join(parts)


@dataclass
class SearchResult:
    """Result from a single search point."""

    config: SearchConfig
    results: SimulationResults
    wall_time: float  # Wall-clock time to run this simulation

    def to_dict(self) -> dict:
        return {
            "config": {
                "enable_dynamic_lp": self.config.enable_dynamic_lp,
                "slo_target": self.config.slo_target,
                "M": self.config.M,
                "switch_policy": self.config.switch_policy,
                "num_prefill": self.config.num_prefill,
                "num_decode": self.config.num_decode,
                "chunked_prefill_size": self.config.chunked_prefill_size,
            },
            "results": self.results.to_dict(),
            "wall_time_seconds": self.wall_time,
        }


class PolicySearcher:
    """Grid search over policy parameters."""

    def __init__(
        self,
        trace_path: str,
        output_dir: str = "result",
        slo_targets: Optional[List[float]] = None,
        M_values: Optional[List[int]] = None,
        switch_policies: Optional[List[str]] = None,
        num_prefill_values: Optional[List[int]] = None,
        num_decode_values: Optional[List[int]] = None,
        chunked_prefill_sizes: Optional[List[int]] = None,
        max_prefill_tokens: int = 16384,
        schedule_policy: str = "fcfs",
    ):
        self.trace_path = trace_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.slo_targets = slo_targets or [1.0]
        self.M_values = M_values or [0]
        self.switch_policies = switch_policies or ["never"]
        self.num_prefill_values = num_prefill_values or [1]
        self.num_decode_values = num_decode_values or [1]
        self.chunked_prefill_sizes = chunked_prefill_sizes or [-1]
        self.max_prefill_tokens = max_prefill_tokens
        self.schedule_policy = schedule_policy

    def generate_configs(self) -> List[SearchConfig]:
        """Generate all search configurations from parameter grid."""
        configs = []
        for np_, nd, slo, M, sw, ch in product(
            self.num_prefill_values,
            self.num_decode_values,
            self.slo_targets,
            self.M_values,
            self.switch_policies,
            self.chunked_prefill_sizes,
        ):
            configs.append(
                SearchConfig(
                    enable_dynamic_lp=True,
                    slo_target=slo,
                    M=M,
                    switch_policy=sw,
                    num_prefill=np_,
                    num_decode=nd,
                    chunked_prefill_size=ch,
                )
            )
        return configs

    def run_single(self, search_config: SearchConfig) -> SearchResult:
        """Run a single simulation configuration."""
        config = SimConfig(
            trace_path=self.trace_path,
            num_prefill_instances=search_config.num_prefill,
            num_decode_instances=search_config.num_decode,
            M=search_config.M,
            enable_dynamic_lp=search_config.enable_dynamic_lp,
            slo_target=search_config.slo_target,
            enable_continuation=search_config.M > 0,
            enable_switching=search_config.switch_policy != "never",
            switch_policy=search_config.switch_policy,
            max_prefill_tokens=self.max_prefill_tokens,
            chunked_prefill_size=search_config.chunked_prefill_size,
            schedule_policy=self.schedule_policy,
        )

        start = time.time()
        engine = SimulationEngine(config)
        results = engine.run()
        wall_time = time.time() - start

        return SearchResult(
            config=search_config,
            results=results,
            wall_time=wall_time,
        )

    def run_search(self) -> List[SearchResult]:
        """Run full grid search."""
        configs = self.generate_configs()
        total = len(configs)
        print(f"\n{'='*60}")
        print(f"Policy Search: {total} configurations")
        print(f"Trace: {self.trace_path}")
        print(f"{'='*60}\n")

        all_results = []
        for i, sc in enumerate(configs):
            print(f"\n--- [{i+1}/{total}] {sc.label()} ---")
            result = self.run_single(sc)
            all_results.append(result)

            r = result.results
            print(
                f"  Throughput={r.throughput:.2f} req/s, "
                f"TTFT={r.avg_ttft:.3f}s, "
                f"ITL={r.avg_itl*1000:.1f}ms, "
                f"SLA={r.sla_attainment_rate:.1f}%, "
                f"Time={result.wall_time:.1f}s"
            )

        # Save results
        self._save_results(all_results)

        # Print summary
        self._print_summary(all_results)

        return all_results

    def _save_results(self, results: List[SearchResult]):
        """Save search results to JSON."""
        output_path = self.output_dir / "policy_search_results.json"
        data = [r.to_dict() for r in results]
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nResults saved to {output_path}")

    def _print_summary(self, results: List[SearchResult]):
        """Print summary of search results sorted by SLA attainment."""
        print(f"\n{'='*80}")
        print("POLICY SEARCH SUMMARY (sorted by SLA attainment)")
        print(f"{'='*80}")
        print(
            f"{'Config':<30} {'SLA%':>6} {'Throughput':>10} "
            f"{'TTFT(s)':>8} {'ITL(ms)':>9} {'E2E(s)':>8}"
        )
        print("-" * 80)

        sorted_results = sorted(
            results,
            key=lambda r: (r.results.sla_attainment_rate, r.results.throughput),
            reverse=True,
        )

        for sr in sorted_results:
            r = sr.results
            print(
                f"{sr.config.label():<30} "
                f"{r.sla_attainment_rate:>5.1f}% "
                f"{r.throughput:>9.2f} "
                f"{r.avg_ttft:>8.3f} "
                f"{r.avg_itl*1000:>9.1f} "
                f"{r.avg_e2e_latency:>8.3f}"
            )

        print(f"{'='*80}")

        # Highlight best
        best = sorted_results[0]
        print(f"\nBest config: {best.config.label()}")
        print(
            f"  SLA={best.results.sla_attainment_rate:.1f}%, "
            f"Throughput={best.results.throughput:.2f} req/s"
        )
