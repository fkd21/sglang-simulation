import pandas as pd
import numpy as np

INPUT = "BurstGPT_cleaned.csv"
OUTPUT = "BurstGPT_20rps.csv"

print("Loading data...")
df = pd.read_csv(INPUT)
print(f"Total requests: {len(df)}")

t_first = df["Timestamp"].iloc[0]
t_last  = df["Timestamp"].iloc[-1]
original_duration = t_last - t_first
original_rate = len(df) / original_duration
print(f"Original duration : {original_duration:.2f} s")
print(f"Original avg rate : {original_rate:.4f} req/s")

# Target: 20 req/s => new total duration
target_rate = 20.0
new_duration = len(df) / target_rate
compression_factor = original_duration / new_duration
print(f"Compression factor: {compression_factor:.4f}x")
print(f"New duration      : {new_duration:.2f} s")

# Compress: shift to zero, scale, keep fractional seconds
df["Timestamp"] = (df["Timestamp"] - t_first) / compression_factor
df["Timestamp"] = df["Timestamp"].round(6)

# Verify
actual_rate = len(df) / df["Timestamp"].iloc[-1]
print(f"Actual avg rate after compression: {actual_rate:.4f} req/s")

df.to_csv(OUTPUT, index=False)
print(f"Saved to {OUTPUT}")
print(f"\nCompressed dataset total duration: {df['Timestamp'].iloc[-1]:.4f} seconds")
