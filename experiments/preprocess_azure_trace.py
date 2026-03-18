"""Preprocessing utilities for Azure trace datasets.

Provides functions to extract time windows from large Azure trace files.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


def extract_time_window(
    input_path: str,
    output_path: str,
    duration_seconds: float,
    start_offset: float = 0.0,
) -> Path:
    """Extract requests within a time window from Azure trace CSV.

    Args:
        input_path: Path to input Azure trace CSV file.
        output_path: Path to output CSV file.
        duration_seconds: Duration of the time window in seconds.
        start_offset: Start offset in seconds from the first request (default: 0.0).

    Returns:
        Path to the output file.

    Example:
        # Extract first 1 hour (3600 seconds) from week-long trace
        extract_time_window(
            "AzureLLMInferenceTrace_code_1week.csv",
            "azure_code_1hour.csv",
            duration_seconds=3600.0
        )

        # Extract second hour (skip first 3600 seconds, then take 3600 seconds)
        extract_time_window(
            "AzureLLMInferenceTrace_code_1week.csv",
            "azure_code_hour2.csv",
            duration_seconds=3600.0,
            start_offset=3600.0
        )
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Reading Azure trace from {input_path}...")
    df = pd.read_csv(input_path)

    # Check required columns
    required_cols = ["TIMESTAMP", "ContextTokens", "GeneratedTokens"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV must contain columns: {required_cols}")

    # Parse timestamps
    print("Parsing timestamps...")
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])

    # Calculate relative time from first request
    first_timestamp = df["TIMESTAMP"].min()
    df["relative_time"] = (df["TIMESTAMP"] - first_timestamp).dt.total_seconds()

    # Filter by time window
    start_time = start_offset
    end_time = start_offset + duration_seconds

    print(f"Extracting window: [{start_time:.1f}s, {end_time:.1f}s)")
    mask = (df["relative_time"] >= start_time) & (df["relative_time"] < end_time)
    df_filtered = df[mask].copy()

    # Drop the relative_time column (only used for filtering)
    df_filtered = df_filtered[required_cols]

    # Save to output file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_filtered.to_csv(output_path, index=False)

    print(f"Extracted {len(df_filtered)} requests from {len(df)} total")
    print(f"Saved to {output_path}")

    return output_path


def get_trace_stats(trace_path: str) -> dict:
    """Get basic statistics about an Azure trace file.

    Args:
        trace_path: Path to Azure trace CSV file.

    Returns:
        Dictionary with trace statistics.
    """
    trace_path = Path(trace_path)
    if not trace_path.exists():
        raise FileNotFoundError(f"Trace file not found: {trace_path}")

    print(f"Reading {trace_path}...")
    df = pd.read_csv(trace_path)

    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
    first_timestamp = df["TIMESTAMP"].min()
    last_timestamp = df["TIMESTAMP"].max()
    duration_seconds = (last_timestamp - first_timestamp).total_seconds()

    stats = {
        "num_requests": len(df),
        "first_timestamp": str(first_timestamp),
        "last_timestamp": str(last_timestamp),
        "duration_seconds": duration_seconds,
        "duration_hours": duration_seconds / 3600.0,
        "context_tokens": {
            "min": int(df["ContextTokens"].min()),
            "max": int(df["ContextTokens"].max()),
            "mean": float(df["ContextTokens"].mean()),
            "median": float(df["ContextTokens"].median()),
        },
        "generated_tokens": {
            "min": int(df["GeneratedTokens"].min()),
            "max": int(df["GeneratedTokens"].max()),
            "mean": float(df["GeneratedTokens"].mean()),
            "median": float(df["GeneratedTokens"].median()),
        },
    }

    return stats


def main():
    """CLI for extracting time windows from Azure traces."""
    # Default: Extract first 1 hour from week-long code trace
    input_file = Path(__file__).resolve().parent.parent.parent / "AzurePublicDataset" / "data" / "AzureLLMInferenceTrace_code.csv"
    output_file = Path(__file__).resolve().parent / "data" / "azure_code_1hour.csv"

    if not input_file.exists():
        # Try alternative path
        input_file = Path(__file__).resolve().parent.parent.parent / "AzureLLMInferenceTrace_code_1week.csv"
        if not input_file.exists():
            print(f"Error: Could not find Azure week trace at:")
            print(f"  {input_file}")
            print("\nPlease update the input_file path in this script or provide the file.")
            sys.exit(1)

    print("=" * 70)
    print("Azure Trace Preprocessing")
    print("=" * 70)

    # First, show trace stats
    print("\n1. Trace Statistics:")
    print("-" * 70)
    stats = get_trace_stats(str(input_file))
    print(f"  Total requests: {stats['num_requests']:,}")
    print(f"  Duration: {stats['duration_hours']:.2f} hours ({stats['duration_seconds']:.0f} seconds)")
    print(f"  First request: {stats['first_timestamp']}")
    print(f"  Last request: {stats['last_timestamp']}")
    print(f"  Context tokens: min={stats['context_tokens']['min']}, max={stats['context_tokens']['max']}, median={stats['context_tokens']['median']:.0f}")
    print(f"  Generated tokens: min={stats['generated_tokens']['min']}, max={stats['generated_tokens']['max']}, median={stats['generated_tokens']['median']:.0f}")

    # Extract first 1 hour
    print("\n2. Extracting Time Window:")
    print("-" * 70)
    extract_time_window(
        input_path=str(input_file),
        output_path=str(output_file),
        duration_seconds=3600.0,
        start_offset=0.0,
    )

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
