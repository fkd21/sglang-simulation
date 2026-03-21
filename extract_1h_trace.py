"""Extract first 1 hour of trace data for quick testing."""

import pandas as pd
from datetime import timedelta

print("Loading 24h trace file...")
df = pd.read_csv('azure_code_24h.csv')

print(f"Total rows in 24h trace: {len(df)}")

# Parse timestamp column
df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])

# Get the start time
start_time = df['TIMESTAMP'].min()
end_time_1h = start_time + timedelta(hours=1)

print(f"Start time: {start_time}")
print(f"End time (1h): {end_time_1h}")

# Filter to first 1 hour
df_1h = df[df['TIMESTAMP'] <= end_time_1h].copy()

print(f"Rows in first 1 hour: {len(df_1h)}")

# Save to new file
output_file = 'azure_code_1h.csv'
df_1h.to_csv(output_file, index=False)

print(f"\nSaved {len(df_1h)} rows to {output_file}")
print(f"Time range: {df_1h['TIMESTAMP'].min()} to {df_1h['TIMESTAMP'].max()}")
print(f"Duration: {df_1h['TIMESTAMP'].max() - df_1h['TIMESTAMP'].min()}")
