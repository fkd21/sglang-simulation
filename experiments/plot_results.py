"""Plot partial offload experiment results.

Reads JSON results from simulation/experiments/results/ and generates
PNG charts to simulation/experiments/results/plots/.

Usage:
    python -m simulation.experiments.plot_results
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent / "results"
PLOTS_DIR = RESULTS_DIR / "plots"


def load_results(name: str) -> list:
    """Load latest experiment results matching the prefix name."""
    path = RESULTS_DIR / f"{name}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    candidates = sorted(RESULTS_DIR.glob(f"{name}_*.json"))
    if not candidates:
        print(f"Warning: no files matching {name}* found, skipping")
        return []
    path = candidates[-1]
    print(f"  Loading {path.name}")
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Experiment 1: SLO Target Sensitivity
# ---------------------------------------------------------------------------

def plot_slo_sensitivity(data: list):
    """Line chart: x=SLO target, y=TTFT/ITL, secondary axis for num_offloaded."""
    if not data:
        return

    baseline = [d for d in data if d.get("slo_target") is None]
    offload = [d for d in data if d.get("slo_target") is not None]
    offload.sort(key=lambda d: d["slo_target"])

    slo_vals = [d["slo_target"] for d in offload]
    ttft_avg = [d["avg_ttft"] for d in offload]
    ttft_p99 = [d["p99_ttft"] for d in offload]
    itl_avg = [d["avg_itl"] * 1000 for d in offload]  # ms
    num_off = [d["num_offloaded"] for d in offload]

    bl_ttft = baseline[0]["avg_ttft"] if baseline else None
    bl_itl = baseline[0]["avg_itl"] * 1000 if baseline else None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: TTFT
    ax1.plot(slo_vals, ttft_avg, "o-", color="tab:blue", label="Avg TTFT (offload)")
    ax1.plot(slo_vals, ttft_p99, "s--", color="tab:cyan", label="P99 TTFT (offload)")
    if bl_ttft is not None:
        ax1.axhline(bl_ttft, color="tab:red", linestyle=":", label=f"Baseline avg TTFT ({bl_ttft:.3f}s)")
    ax1.set_xlabel("SLO Target (s)")
    ax1.set_ylabel("TTFT (s)")
    ax1.set_title("TTFT vs SLO Target")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax1b = ax1.twinx()
    ax1b.bar(slo_vals, num_off, width=0.05, alpha=0.3, color="tab:orange", label="Num Offloaded")
    ax1b.set_ylabel("Num Offloaded", color="tab:orange")
    ax1b.tick_params(axis="y", labelcolor="tab:orange")

    # Right: ITL + SLA
    sla = [d["sla_attainment_rate"] for d in offload]
    bl_sla = baseline[0]["sla_attainment_rate"] if baseline else None

    ax2.plot(slo_vals, itl_avg, "o-", color="tab:green", label="Avg ITL (offload)")
    if bl_itl is not None:
        ax2.axhline(bl_itl, color="tab:red", linestyle=":", label=f"Baseline avg ITL ({bl_itl:.1f}ms)")
    ax2.set_xlabel("SLO Target (s)")
    ax2.set_ylabel("ITL (ms)")
    ax2.set_title("ITL & SLA Attainment vs SLO Target")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, alpha=0.3)

    ax2b = ax2.twinx()
    ax2b.plot(slo_vals, sla, "D-", color="tab:purple", label="SLA Attainment (offload)")
    if bl_sla is not None:
        ax2b.axhline(bl_sla, color="tab:purple", linestyle=":", alpha=0.5, label=f"Baseline SLA ({bl_sla:.1f}%)")
    ax2b.set_ylabel("SLA Attainment (%)", color="tab:purple")
    ax2b.tick_params(axis="y", labelcolor="tab:purple")
    ax2b.legend(loc="upper right", fontsize=8)

    fig.suptitle("Experiment 1: SLO Target Sensitivity", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "exp1_slo_sensitivity.png", dpi=150)
    plt.close(fig)
    print("  Saved exp1_slo_sensitivity.png")


# ---------------------------------------------------------------------------
# Experiment 2: Instance Configuration
# ---------------------------------------------------------------------------

def plot_instance_config(data: list):
    """Grouped bar chart: baseline vs offload for each P:D config."""
    if not data:
        return

    configs = []
    for d in data:
        if d["pd_config"] not in configs:
            configs.append(d["pd_config"])

    baseline = {d["pd_config"]: d for d in data if d["mode"] == "baseline"}
    offload = {d["pd_config"]: d for d in data if d["mode"] == "offload"}

    x = np.arange(len(configs))
    width = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # TTFT
    ax = axes[0]
    bl_vals = [baseline[c]["avg_ttft"] for c in configs]
    off_vals = [offload[c]["avg_ttft"] for c in configs]
    ax.bar(x - width / 2, bl_vals, width, label="Baseline", color="tab:blue", alpha=0.8)
    ax.bar(x + width / 2, off_vals, width, label="Offload", color="tab:orange", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45)
    ax.set_ylabel("Avg TTFT (s)")
    ax.set_title("Avg TTFT")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # ITL
    ax = axes[1]
    bl_vals = [baseline[c]["avg_itl"] * 1000 for c in configs]
    off_vals = [offload[c]["avg_itl"] * 1000 for c in configs]
    ax.bar(x - width / 2, bl_vals, width, label="Baseline", color="tab:blue", alpha=0.8)
    ax.bar(x + width / 2, off_vals, width, label="Offload", color="tab:orange", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45)
    ax.set_ylabel("Avg ITL (ms)")
    ax.set_title("Avg ITL")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # Throughput
    ax = axes[2]
    bl_vals = [baseline[c]["throughput"] for c in configs]
    off_vals = [offload[c]["throughput"] for c in configs]
    ax.bar(x - width / 2, bl_vals, width, label="Baseline", color="tab:blue", alpha=0.8)
    ax.bar(x + width / 2, off_vals, width, label="Offload", color="tab:orange", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45)
    ax.set_ylabel("Throughput (req/s)")
    ax.set_title("Throughput")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Experiment 2: Instance Configuration (Baseline vs Offload)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "exp2_instance_config.png", dpi=150)
    plt.close(fig)
    print("  Saved exp2_instance_config.png")


# ---------------------------------------------------------------------------
# Experiment 3: Context Length Distribution
# ---------------------------------------------------------------------------

def plot_context_length(data: list):
    """Grouped bar chart + num_offloaded overlay for each workload type."""
    if not data:
        return

    workloads = []
    for d in data:
        if d["workload"] not in workloads:
            workloads.append(d["workload"])

    baseline = {d["workload"]: d for d in data if d["mode"] == "baseline"}
    offload = {d["workload"]: d for d in data if d["mode"] == "offload"}

    x = np.arange(len(workloads))
    width = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # TTFT
    ax = axes[0]
    bl_vals = [baseline[w]["avg_ttft"] for w in workloads]
    off_vals = [offload[w]["avg_ttft"] for w in workloads]
    ax.bar(x - width / 2, bl_vals, width, label="Baseline", color="tab:blue", alpha=0.8)
    ax.bar(x + width / 2, off_vals, width, label="Offload", color="tab:orange", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(workloads, rotation=45)
    ax.set_ylabel("Avg TTFT (s)")
    ax.set_title("Avg TTFT by Context Distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # P99 TTFT
    ax = axes[1]
    bl_vals = [baseline[w]["p99_ttft"] for w in workloads]
    off_vals = [offload[w]["p99_ttft"] for w in workloads]
    ax.bar(x - width / 2, bl_vals, width, label="Baseline", color="tab:blue", alpha=0.8)
    ax.bar(x + width / 2, off_vals, width, label="Offload", color="tab:orange", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(workloads, rotation=45)
    ax.set_ylabel("P99 TTFT (s)")
    ax.set_title("P99 TTFT by Context Distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # Num offloaded
    ax = axes[2]
    off_count = [offload[w]["num_offloaded"] for w in workloads]
    total = [offload[w]["total_requests"] for w in workloads]
    pct = [o / t * 100 if t > 0 else 0 for o, t in zip(off_count, total)]
    ax.bar(x, pct, 0.6, color="tab:green", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(workloads, rotation=45)
    ax.set_ylabel("Offloaded (%)")
    ax.set_title("% Requests Offloaded")
    ax.grid(True, alpha=0.3, axis="y")
    for i, (c, p) in enumerate(zip(off_count, pct)):
        ax.text(i, p + 0.5, f"{c}", ha="center", fontsize=8)

    fig.suptitle("Experiment 3: Context Length Distribution", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "exp3_context_length.png", dpi=150)
    plt.close(fig)
    print("  Saved exp3_context_length.png")


# ---------------------------------------------------------------------------
# Experiment 4: LP Window Size
# ---------------------------------------------------------------------------

def plot_window_size(data: list):
    """Line chart: x=window size, y=TTFT and num_offloaded."""
    if not data:
        return

    baseline = [d for d in data if d["window_size"] == 0]
    offload = sorted([d for d in data if d["window_size"] > 0], key=lambda d: d["window_size"])

    ws = [d["window_size"] for d in offload]
    ttft_avg = [d["avg_ttft"] for d in offload]
    ttft_p99 = [d["p99_ttft"] for d in offload]
    num_off = [d["num_offloaded"] for d in offload]
    throughput = [d["throughput"] for d in offload]

    bl_ttft = baseline[0]["avg_ttft"] if baseline else None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # TTFT
    ax1.plot(ws, ttft_avg, "o-", color="tab:blue", label="Avg TTFT")
    ax1.plot(ws, ttft_p99, "s--", color="tab:cyan", label="P99 TTFT")
    if bl_ttft is not None:
        ax1.axhline(bl_ttft, color="tab:red", linestyle=":", label=f"Baseline ({bl_ttft:.3f}s)")
    ax1.set_xlabel("LP Window Size")
    ax1.set_ylabel("TTFT (s)")
    ax1.set_title("TTFT vs Window Size")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Num offloaded + Throughput
    ax2.bar(ws, num_off, width=0.8, alpha=0.4, color="tab:orange", label="Num Offloaded")
    ax2.set_xlabel("LP Window Size")
    ax2.set_ylabel("Num Offloaded", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax2.grid(True, alpha=0.3)

    ax2b = ax2.twinx()
    ax2b.plot(ws, throughput, "D-", color="tab:green", label="Throughput")
    if baseline:
        ax2b.axhline(baseline[0]["throughput"], color="tab:green", linestyle=":", alpha=0.5)
    ax2b.set_ylabel("Throughput (req/s)", color="tab:green")
    ax2b.tick_params(axis="y", labelcolor="tab:green")

    ax2.legend(loc="upper left", fontsize=8)
    ax2b.legend(loc="upper right", fontsize=8)
    ax2.set_title("Offloaded Requests & Throughput")

    fig.suptitle("Experiment 4: LP Window Size", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "exp4_window_size.png", dpi=150)
    plt.close(fig)
    print("  Saved exp4_window_size.png")


# ---------------------------------------------------------------------------
# Experiment 5: Load Level
# ---------------------------------------------------------------------------

def plot_load_level(data: list):
    """Line chart: x=arrival rate, y=TTFT/SLA, baseline vs offload."""
    if not data:
        return

    baseline = sorted([d for d in data if d["mode"] == "baseline"], key=lambda d: d["arrival_rate"])
    offload = sorted([d for d in data if d["mode"] == "offload"], key=lambda d: d["arrival_rate"])

    rates = [d["arrival_rate"] for d in baseline]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Avg TTFT
    ax = axes[0]
    ax.plot(rates, [d["avg_ttft"] for d in baseline], "o-", color="tab:blue", label="Baseline")
    ax.plot(rates, [d["avg_ttft"] for d in offload], "s-", color="tab:orange", label="Offload")
    ax.set_xlabel("Arrival Rate (req/s)")
    ax.set_ylabel("Avg TTFT (s)")
    ax.set_title("Avg TTFT vs Load")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # P99 TTFT
    ax = axes[1]
    ax.plot(rates, [d["p99_ttft"] for d in baseline], "o-", color="tab:blue", label="Baseline")
    ax.plot(rates, [d["p99_ttft"] for d in offload], "s-", color="tab:orange", label="Offload")
    ax.set_xlabel("Arrival Rate (req/s)")
    ax.set_ylabel("P99 TTFT (s)")
    ax.set_title("P99 TTFT vs Load")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # SLA + Offloaded count
    ax = axes[2]
    ax.plot(rates, [d["sla_attainment_rate"] for d in baseline], "o-", color="tab:blue", label="Baseline SLA")
    ax.plot(rates, [d["sla_attainment_rate"] for d in offload], "s-", color="tab:orange", label="Offload SLA")
    ax.set_xlabel("Arrival Rate (req/s)")
    ax.set_ylabel("SLA Attainment (%)")
    ax.set_title("SLA Attainment & Offloaded Count")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax_b = ax.twinx()
    ax_b.bar(rates, [d["num_offloaded"] for d in offload], width=0.3, alpha=0.3, color="tab:green", label="Offloaded")
    ax_b.set_ylabel("Num Offloaded", color="tab:green")
    ax_b.tick_params(axis="y", labelcolor="tab:green")
    ax_b.legend(loc="upper right", fontsize=8)

    fig.suptitle("Experiment 5: Load Level (Arrival Rate)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "exp5_load_level.png", dpi=150)
    plt.close(fig)
    print("  Saved exp5_load_level.png")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def plot_summary_table(all_data: dict):
    """Generate a summary comparison table as an image."""
    rows = []

    # From exp2 (instance config), extract 2P4D comparison
    exp2 = all_data.get("exp2_instance_config", [])
    bl = next((d for d in exp2 if d.get("pd_config") == "2P4D" and d.get("mode") == "baseline"), None)
    off = next((d for d in exp2 if d.get("pd_config") == "2P4D" and d.get("mode") == "offload"), None)
    if bl and off:
        rows.append({
            "Metric": "Avg TTFT (s)",
            "Baseline": f"{bl['avg_ttft']:.4f}",
            "Offload": f"{off['avg_ttft']:.4f}",
            "Delta": f"{off['avg_ttft'] - bl['avg_ttft']:+.4f}",
        })
        rows.append({
            "Metric": "P99 TTFT (s)",
            "Baseline": f"{bl['p99_ttft']:.4f}",
            "Offload": f"{off['p99_ttft']:.4f}",
            "Delta": f"{off['p99_ttft'] - bl['p99_ttft']:+.4f}",
        })
        rows.append({
            "Metric": "Avg ITL (ms)",
            "Baseline": f"{bl['avg_itl']*1000:.2f}",
            "Offload": f"{off['avg_itl']*1000:.2f}",
            "Delta": f"{(off['avg_itl'] - bl['avg_itl'])*1000:+.2f}",
        })
        rows.append({
            "Metric": "Throughput (req/s)",
            "Baseline": f"{bl['throughput']:.2f}",
            "Offload": f"{off['throughput']:.2f}",
            "Delta": f"{off['throughput'] - bl['throughput']:+.2f}",
        })
        rows.append({
            "Metric": "SLA Attainment (%)",
            "Baseline": f"{bl['sla_attainment_rate']:.1f}",
            "Offload": f"{off['sla_attainment_rate']:.1f}",
            "Delta": f"{off['sla_attainment_rate'] - bl['sla_attainment_rate']:+.1f}",
        })
        rows.append({
            "Metric": "Num Offloaded",
            "Baseline": "0",
            "Offload": str(off["num_offloaded"]),
            "Delta": f"+{off['num_offloaded']}",
        })

    if not rows:
        return

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")
    col_labels = ["Metric", "Baseline", "Offload (SLO=1.0)", "Delta"]
    table_data = [[r["Metric"], r["Baseline"], r["Offload"], r["Delta"]] for r in rows]
    table = ax.table(cellText=table_data, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Style header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")

    fig.suptitle("Summary: 2P4D Azure Trace (Baseline vs Partial Offload)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "summary_table.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved summary_table.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating plots to {PLOTS_DIR}")

    # Try loading combined results first (latest timestamped version)
    combined_candidates = sorted(RESULTS_DIR.glob("all_experiments*.json"))
    if combined_candidates:
        with open(combined_candidates[-1]) as f:
            all_data = json.load(f)
    else:
        all_data = {}
        for name in ["exp1_slo_sensitivity", "exp2_instance_config",
                      "exp3_context_length", "exp4_window_size", "exp5_load_level"]:
            data = load_results(name)
            if data:
                all_data[name] = data

    if not all_data:
        print("No results found. Run experiments first.")
        sys.exit(1)

    if "exp1_slo_sensitivity" in all_data:
        plot_slo_sensitivity(all_data["exp1_slo_sensitivity"])

    if "exp2_instance_config" in all_data:
        plot_instance_config(all_data["exp2_instance_config"])

    if "exp3_context_length" in all_data:
        plot_context_length(all_data["exp3_context_length"])

    if "exp4_window_size" in all_data:
        plot_window_size(all_data["exp4_window_size"])

    if "exp5_load_level" in all_data:
        plot_load_level(all_data["exp5_load_level"])

    plot_summary_table(all_data)

    print("\nDone!")


if __name__ == "__main__":
    main()
