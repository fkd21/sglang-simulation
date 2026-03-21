"""Extract first 24 hours of trace data from week-long Azure trace."""

import pandas as pd
from datetime import timedelta

print("Loading trace file...")
df = pd.read_csv('AzureLLMInferenceTrace_code_1week.csv')

print(f"Total rows: {len(df)}")

# Parse timestamp column
df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])

# Get the start time
start_time = df['TIMESTAMP'].min()
end_time_24h = start_time + timedelta(hours=24)

print(f"Start time: {start_time}")
print(f"End time (24h): {end_time_24h}")

# Filter to first 24 hours
df_24h = df[df['TIMESTAMP'] <= end_time_24h].copy()

print(f"Rows in first 24 hours: {len(df_24h)}")

# Save to new file
output_file = 'azure_code_24h.csv'
df_24h.to_csv(output_file, index=False)

print(f"\nSaved {len(df_24h)} rows to {output_file}")
print(f"Time range: {df_24h['TIMESTAMP'].min()} to {df_24h['TIMESTAMP'].max()}")
print(f"Duration: {df_24h['TIMESTAMP'].max() - df_24h['TIMESTAMP'].min()}")
