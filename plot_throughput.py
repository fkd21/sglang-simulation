import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

INPUT = "BurstGPT_20rps.csv"
OUTPUT = "throughput_plot.png"
BIN_SIZE = 60  # seconds per bin

print("Loading data...")
df = pd.read_csv(INPUT)
print(f"Total requests: {len(df)}, duration: {df['Timestamp'].iloc[-1]:.1f} s")

# Bin by time window
df["bin"] = (df["Timestamp"] // BIN_SIZE).astype(int)
bins = df.groupby("bin").agg(
    input_tokens=("Request tokens", "sum"),
    output_tokens=("Response tokens", "sum"),
).reset_index()

# Convert to throughput (tokens/sec)
bins["time_min"] = bins["bin"] * BIN_SIZE / 60  # x-axis in minutes
bins["input_tps"]  = bins["input_tokens"]  / BIN_SIZE
bins["output_tps"] = bins["output_tokens"] / BIN_SIZE

print(f"Number of bins ({BIN_SIZE}s each): {len(bins)}")
print(f"Avg input  throughput: {bins['input_tps'].mean():.1f} tokens/s")
print(f"Avg output throughput: {bins['output_tps'].mean():.1f} tokens/s")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

ax1, ax2 = axes

ax1.plot(bins["time_min"], bins["input_tps"], color="#2196F3", linewidth=0.6, alpha=0.85)
ax1.set_ylabel("Input Throughput\n(tokens/s)", fontsize=11)
ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax1.grid(True, linestyle="--", alpha=0.4)
ax1.set_title("BurstGPT (20 req/s compressed) — Token Throughput over Time", fontsize=13)

ax2.plot(bins["time_min"], bins["output_tps"], color="#4CAF50", linewidth=0.6, alpha=0.85)
ax2.set_ylabel("Output Throughput\n(tokens/s)", fontsize=11)
ax2.set_xlabel("Time (minutes)", fontsize=11)
ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax2.grid(True, linestyle="--", alpha=0.4)

# Annotate averages
ax1.axhline(bins["input_tps"].mean(),  color="#2196F3", linestyle="--", linewidth=1.2,
            label=f"mean = {bins['input_tps'].mean():,.0f} tok/s")
ax2.axhline(bins["output_tps"].mean(), color="#4CAF50", linestyle="--", linewidth=1.2,
            label=f"mean = {bins['output_tps'].mean():,.0f} tok/s")
ax1.legend(fontsize=10)
ax2.legend(fontsize=10)

plt.tight_layout()
plt.savefig(OUTPUT, dpi=150, bbox_inches="tight")
print(f"Saved to {OUTPUT}")
