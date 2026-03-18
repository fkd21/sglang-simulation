"""Plot role switching policy experiment results.

Reads JSON results from experiments/results/ and generates
PNG charts to experiments/results/plots/.

Usage:
    python -m experiments.plot_switching_results
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
# Plot 1: Baseline Policy Comparison
# ---------------------------------------------------------------------------

def plot_baseline_comparison():
    """Bar charts comparing no-switch vs alpha vs v1 policies."""
    data = load_results("switching_exp1_baseline")
    if not data:
        return

    policies = ["none", "alpha", "v1"]
    policy_labels = ["No Switching", "Alpha Policy", "V1 Policy"]
    colors = ["tab:gray", "tab:blue", "tab:green"]

    metrics = {p: {} for p in policies}
    for d in data:
        policy = d.get("policy", "none")
        if policy in metrics:
            metrics[policy] = {
                "avg_e2e_latency": d.get("avg_e2e_latency", 0),
                "p99_e2e_latency": d.get("p99_e2e_latency", 0),
                "throughput": d.get("throughput", 0),
                "sla_attainment_rate": d.get("sla_attainment_rate", 0),
                "num_switches": d.get("num_switches", 0),
                "avg_ttft": d.get("avg_ttft", 0),
            }

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    x = np.arange(len(policies))
    width = 0.6

    def bar_with_labels(ax, values, ylabel, title):
        bars = ax.bar(x, values, width, color=colors)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(policy_labels)
        ax.grid(True, alpha=0.3, axis="y")
        for bar in bars:
            h = bar.get_height()
            fmt = f"{h:.3f}" if h < 100 else f"{h:.1f}"
            ax.text(bar.get_x() + bar.get_width() / 2., h, fmt,
                    ha="center", va="bottom", fontsize=9)

    bar_with_labels(axes[0, 0],
                    [metrics[p].get("avg_e2e_latency", 0) for p in policies],
                    "Avg E2E Latency (s)", "Average E2E Latency")
    bar_with_labels(axes[0, 1],
                    [metrics[p].get("p99_e2e_latency", 0) for p in policies],
                    "P99 E2E Latency (s)", "P99 E2E Latency")
    bar_with_labels(axes[0, 2],
                    [metrics[p].get("avg_ttft", 0) for p in policies],
                    "Avg TTFT (s)", "Average TTFT")
    bar_with_labels(axes[1, 0],
                    [metrics[p].get("throughput", 0) for p in policies],
                    "Throughput (req/s)", "Request Throughput")
    bar_with_labels(axes[1, 1],
                    [metrics[p].get("sla_attainment_rate", 0) for p in policies],
                    "SLA Attainment (%)", "SLA Attainment Rate")
    bar_with_labels(axes[1, 2],
                    [metrics[p].get("num_switches", 0) for p in policies],
                    "Number of Role Switches", "Role Switching Frequency")

    fig.suptitle("Exp1: Baseline Policy Comparison (2P4D, Azure Trace)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / "switching_exp1_baseline.png", dpi=150)
    plt.close(fig)
    print("  Saved switching_exp1_baseline.png")


# ---------------------------------------------------------------------------
# Plot 2: Instance Configuration Scaling
# ---------------------------------------------------------------------------

def plot_instance_config_scaling():
    """Performance vs P:D ratio for all policies."""
    data = load_results("switching_exp2_instance_config")
    if not data:
        return

    configs = ["1P2D", "2P4D", "2P6D", "4P4D"]
    policies = ["none", "alpha", "v1"]
    policy_labels = ["No Switching", "Alpha", "V1"]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    x = np.arange(len(configs))
    width = 0.25

    for i, (policy, label) in enumerate(zip(policies, policy_labels)):
        policy_data = [d for d in data if d.get("policy") == policy]
        sorted_data = []
        for cfg in configs:
            matching = [d for d in policy_data if d.get("pd_config") == cfg]
            sorted_data.append(matching[0] if matching else {})

        offset = (i - 1) * width
        ax1.bar(x + offset, [d.get("avg_e2e_latency", 0) for d in sorted_data], width, label=label)
        ax2.bar(x + offset, [d.get("p99_e2e_latency", 0) for d in sorted_data], width, label=label)
        ax3.bar(x + offset, [d.get("throughput", 0) for d in sorted_data], width, label=label)
        ax4.bar(x + offset, [d.get("sla_attainment_rate", 0) for d in sorted_data], width, label=label)

    for ax, ylabel, title in [
        (ax1, "Avg E2E Latency (s)", "Average Latency"),
        (ax2, "P99 E2E Latency (s)", "P99 Latency"),
        (ax3, "Throughput (req/s)", "Throughput"),
        (ax4, "SLA Attainment (%)", "SLA Attainment"),
    ]:
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(configs)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Exp2: Instance Configuration Scaling (Azure Trace)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()

    fig.savefig(PLOTS_DIR / "switching_exp2_instance_config.png", dpi=150)
    plt.close(fig)
    print("  Saved switching_exp2_instance_config.png")


# ---------------------------------------------------------------------------
# Plot 3: Load Level Sensitivity
# ---------------------------------------------------------------------------

def plot_load_sensitivity():
    """Line plots of latency vs arrival rate for each policy."""
    data = load_results("switching_exp3_load_level")
    if not data:
        return

    policies = ["none", "alpha", "v1"]
    policy_labels = ["No Switching", "Alpha", "V1"]
    policy_colors = {"none": "tab:gray", "alpha": "tab:blue", "v1": "tab:green"}

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    for policy, label in zip(policies, policy_labels):
        pd = sorted([d for d in data if d.get("policy") == policy],
                     key=lambda d: d.get("arrival_rate", 0))
        rates = [d.get("arrival_rate", 0) for d in pd]
        color = policy_colors[policy]

        ax1.plot(rates, [d.get("avg_e2e_latency", 0) for d in pd], "o-", label=label, color=color)
        ax2.plot(rates, [d.get("p99_e2e_latency", 0) for d in pd], "o-", label=label, color=color)
        ax3.plot(rates, [d.get("throughput", 0) for d in pd], "o-", label=label, color=color)
        ax4.plot(rates, [d.get("sla_attainment_rate", 0) for d in pd], "o-", label=label, color=color)

    for ax, xlabel, ylabel, title in [
        (ax1, "Arrival Rate (req/s)", "Avg E2E Latency (s)", "Average Latency"),
        (ax2, "Arrival Rate (req/s)", "P99 E2E Latency (s)", "P99 Latency"),
        (ax3, "Arrival Rate (req/s)", "Throughput (req/s)", "Throughput"),
        (ax4, "Arrival Rate (req/s)", "SLA Attainment (%)", "SLA Attainment"),
    ]:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Exp3: Load Level Sensitivity (2P4D, Synthetic Workload)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()

    fig.savefig(PLOTS_DIR / "switching_exp3_load_level.png", dpi=150)
    plt.close(fig)
    print("  Saved switching_exp3_load_level.png")


# ---------------------------------------------------------------------------
# Plot 4: Switching + Offload Interaction
# ---------------------------------------------------------------------------

def plot_interaction():
    """Grouped bars showing switching+offload combinations."""
    data = load_results("switching_exp4_interaction")
    if not data:
        return

    policies = ["none", "alpha", "v1"]
    policy_labels = ["No Switching", "Alpha", "V1"]

    metrics = {}
    for d in data:
        policy = d.get("policy", "none")
        offload = d.get("offload_enabled", False)
        metrics[(policy, offload)] = d

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    x = np.arange(len(policies))
    width = 0.35

    for ax, key, ylabel, title in [
        (ax1, "avg_e2e_latency", "Avg E2E Latency (s)", "Average Latency"),
        (ax2, "p99_e2e_latency", "P99 E2E Latency (s)", "P99 Latency"),
        (ax3, "throughput", "Throughput (req/s)", "Throughput"),
        (ax4, "sla_attainment_rate", "SLA Attainment (%)", "SLA Attainment"),
    ]:
        no_off = [metrics.get((p, False), {}).get(key, 0) for p in policies]
        with_off = [metrics.get((p, True), {}).get(key, 0) for p in policies]
        ax.bar(x - width / 2, no_off, width, label="No Offload", color="tab:blue", alpha=0.8)
        ax.bar(x + width / 2, with_off, width, label="With Offload", color="tab:orange", alpha=0.8)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(policy_labels)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Exp4: Switching + Offload Interaction (2P4D, Azure Trace)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()

    fig.savefig(PLOTS_DIR / "switching_exp4_interaction.png", dpi=150)
    plt.close(fig)
    print("  Saved switching_exp4_interaction.png")


# ---------------------------------------------------------------------------
# Plot 5: Switch Timeline
# ---------------------------------------------------------------------------

def plot_switch_timeline():
    """Timeline showing when switches occurred."""
    data = load_results("switching_exp1_baseline")
    if not data:
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    from matplotlib.patches import Patch

    for idx, (policy_name, ax) in enumerate(zip(["alpha", "v1"], axes)):
        policy_data = [d for d in data if d.get("policy") == policy_name and d.get("num_switches", 0) > 0]
        if policy_data:
            switches = policy_data[0].get("role_switches", [])
            if switches:
                times = [s.get("time", 0) for s in switches]
                labels = [s.get("instance_id", "?") for s in switches]
                to_roles = [s.get("to_role", "?") for s in switches]
                colors = ["tab:blue" if r == "PREFILL" else "tab:orange" for r in to_roles]
                ax.scatter(times, labels, c=colors, s=100, alpha=0.7, edgecolors="black", linewidth=0.5)
                ax.legend(handles=[
                    Patch(facecolor="tab:blue", edgecolor="black", label="-> Prefill"),
                    Patch(facecolor="tab:orange", edgecolor="black", label="-> Decode"),
                ], loc="upper right")
            else:
                ax.text(0.5, 0.5, "No switches", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, f"No {policy_name} data", ha="center", va="center", transform=ax.transAxes)
        ax.set_ylabel("Instance ID")
        ax.set_title(f"{policy_name.upper()} Policy: Role Switch Timeline")
        ax.grid(True, alpha=0.3)

    axes[1].set_xlabel("Simulation Time (s)")
    fig.suptitle("Role Switch Timeline", fontsize=13, fontweight="bold")
    fig.tight_layout()

    fig.savefig(PLOTS_DIR / "switching_timeline.png", dpi=150)
    plt.close(fig)
    print("  Saved switching_timeline.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Generate all plots."""
    print("Generating switching policy experiment plots...")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n1. Baseline comparison...")
    plot_baseline_comparison()

    print("\n2. Instance config scaling...")
    plot_instance_config_scaling()

    print("\n3. Load sensitivity...")
    plot_load_sensitivity()

    print("\n4. Switching + offload interaction...")
    plot_interaction()

    print("\n5. Switch timeline...")
    plot_switch_timeline()

    print(f"\nAll plots saved to {PLOTS_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
