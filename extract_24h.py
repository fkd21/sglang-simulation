import pandas as pd

INPUT  = "BurstGPT_20rps.csv"
OUTPUT = "BurstGPT_20rps_24h.csv"
CUTOFF = 24 * 3600  # 86400 seconds

print("Loading data...")
df = pd.read_csv(INPUT)
print(f"Total requests: {len(df)}")

df_24h = df[df["Timestamp"] <= CUTOFF].copy()
print(f"Requests in first 24h: {len(df_24h)}")
print(f"Actual end timestamp : {df_24h['Timestamp'].iloc[-1]:.4f} s")
print(f"Avg arrival rate     : {len(df_24h) / df_24h['Timestamp'].iloc[-1]:.4f} req/s")

df_24h.to_csv(OUTPUT, index=False)
print(f"Saved to {OUTPUT}")
