"""Trace subsampling utility for Azure LLM inference traces (v2 with 5x5 binning).

Subsamples traces while preserving temporal distribution across request categories.
Uses systematic sampling to maintain arrival patterns and burstiness within time windows.
Categorizes requests into 25 bins (5x5 input/output with distribution 30%, 20%, 20%, 20%, 10%)
and maintains the distribution within configurable time windows.

This v2 version uses finer-grained binning and vectorized operations for improved performance.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from multiprocessing import Pool, cpu_count
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _process_chunk_for_percentiles(args):
    """Helper function to process a chunk in parallel."""
    chunk_data, dtype_dict = args
    df = pd.read_csv(
        chunk_data,
        usecols=['ContextTokens', 'GeneratedTokens'],
        dtype=dtype_dict
    )
    return df['ContextTokens'].values, df['GeneratedTokens'].values


def compute_percentile_boundaries(
    input_path: str,
    chunk_size: int = 100000,
    n_workers: int = 1
) -> Dict[str, Tuple[float, float, float, float]]:
    """Compute 30th, 50th, 70th, and 90th percentiles for input/output tokens using streaming.

    Args:
        input_path: Path to CSV trace file
        chunk_size: Number of rows to read per chunk for memory efficiency
        n_workers: Number of parallel workers (default: 1 for sequential)

    Returns:
        Dictionary with boundaries:
        {
            'input': (p30, p50, p70, p90),
            'output': (p30, p50, p70, p90)
        }
    """
    print(f"\n[Pass 0] Computing percentile boundaries (using {n_workers} workers)...")

    input_tokens = []
    output_tokens = []

    # Read in chunks and accumulate token values
    dtype_dict = {'ContextTokens': 'int32', 'GeneratedTokens': 'int32'}
    chunks_processed = 0

    for chunk in pd.read_csv(
        input_path,
        usecols=['ContextTokens', 'GeneratedTokens'],
        dtype=dtype_dict,
        chunksize=chunk_size
    ):
        input_tokens.extend(chunk['ContextTokens'].values)
        output_tokens.extend(chunk['GeneratedTokens'].values)
        chunks_processed += 1

        if chunks_processed % 10 == 0:
            rows_processed = chunks_processed * chunk_size
            print(f"  Processed {rows_processed:,} rows...")

    # Compute percentiles
    input_tokens = np.array(input_tokens)
    output_tokens = np.array(output_tokens)

    input_p30, input_p50, input_p70, input_p90 = np.percentile(input_tokens, [30, 50, 70, 90])
    output_p30, output_p50, output_p70, output_p90 = np.percentile(output_tokens, [30, 50, 70, 90])

    boundaries = {
        'input': (float(input_p30), float(input_p50), float(input_p70), float(input_p90)),
        'output': (float(output_p30), float(output_p50), float(output_p70), float(output_p90))
    }

    print(f"\n  Percentile boundaries computed:")
    print(f"    Input tokens:  p30={input_p30:.0f}, p50={input_p50:.0f}, p70={input_p70:.0f}, p90={input_p90:.0f}")
    print(f"    Output tokens: p30={output_p30:.0f}, p50={output_p50:.0f}, p70={output_p70:.0f}, p90={output_p90:.0f}")
    print(f"    Total requests analyzed: {len(input_tokens):,}")

    return boundaries


def categorize_request(
    context_tokens: int,
    generated_tokens: int,
    input_boundaries: Tuple[float, float, float, float],
    output_boundaries: Tuple[float, float, float, float]
) -> Tuple[int, int]:
    """Categorize a request into input/output bins (0-4 for 5 bins).

    Args:
        context_tokens: Input token count
        generated_tokens: Output token count
        input_boundaries: (p30, p50, p70, p90) for input tokens
        output_boundaries: (p30, p50, p70, p90) for output tokens

    Returns:
        Tuple of (input_bin, output_bin) where each is 0, 1, 2, 3, or 4
        - Bin 0: 0-30th percentile (30% of data)
        - Bin 1: 30-50th percentile (20% of data)
        - Bin 2: 50-70th percentile (20% of data)
        - Bin 3: 70-90th percentile (20% of data)
        - Bin 4: 90-100th percentile (10% of data)
    """
    # Input categorization
    if context_tokens <= input_boundaries[0]:
        input_bin = 0
    elif context_tokens <= input_boundaries[1]:
        input_bin = 1
    elif context_tokens <= input_boundaries[2]:
        input_bin = 2
    elif context_tokens <= input_boundaries[3]:
        input_bin = 3
    else:
        input_bin = 4

    # Output categorization
    if generated_tokens <= output_boundaries[0]:
        output_bin = 0
    elif generated_tokens <= output_boundaries[1]:
        output_bin = 1
    elif generated_tokens <= output_boundaries[2]:
        output_bin = 2
    elif generated_tokens <= output_boundaries[3]:
        output_bin = 3
    else:
        output_bin = 4

    return input_bin, output_bin


def categorize_requests_vectorized(
    context_tokens_arr: np.ndarray,
    generated_tokens_arr: np.ndarray,
    input_boundaries: Tuple[float, float, float, float],
    output_boundaries: Tuple[float, float, float, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized categorization of requests using numpy.searchsorted.

    This is significantly faster than calling categorize_request() in a loop.

    Args:
        context_tokens_arr: Array of input token counts
        generated_tokens_arr: Array of output token counts
        input_boundaries: (p30, p50, p70, p90) for input tokens
        output_boundaries: (p30, p50, p70, p90) for output tokens

    Returns:
        Tuple of (input_bins, output_bins) where each is a numpy array of bin indices (0-4)
    """
    # searchsorted returns the index where the value would be inserted
    # For boundaries [p30, p50, p70, p90], searchsorted returns 0-4 which maps perfectly to our bins
    input_bins = np.searchsorted(input_boundaries, context_tokens_arr, side='right')
    output_bins = np.searchsorted(output_boundaries, generated_tokens_arr, side='right')

    return input_bins, output_bins


def get_minute_bin(timestamp_str: str, start_time: Optional[datetime] = None) -> Tuple[int, datetime]:
    """Parse timestamp and return minute index from trace start.

    Args:
        timestamp_str: Timestamp string from CSV
        start_time: Start time of trace (computed from first row if None)

    Returns:
        Tuple of (minute_bin, parsed_timestamp)
    """
    # Handle Azure's 7-digit microseconds (Python supports 6)
    if '.' in timestamp_str and '+' in timestamp_str:
        parts = timestamp_str.split('.')
        microsec_and_tz = parts[1].split('+')
        microseconds = microsec_and_tz[0][:6]  # Truncate to 6 digits
        timezone = microsec_and_tz[1]
        timestamp_str = f"{parts[0]}.{microseconds}+{timezone}"

    timestamp = pd.to_datetime(timestamp_str)

    if start_time is None:
        return 0, timestamp

    seconds_from_start = (timestamp - start_time).total_seconds()
    minute_bin = int(seconds_from_start // 60)

    return minute_bin, timestamp


def round_randomly(value: float, rng: np.random.Generator) -> int:
    """Randomly round a float to maintain expected value.

    Example: 5.7 -> 6 with 70% probability, 5 with 30% probability

    Args:
        value: Float value to round
        rng: Numpy random generator

    Returns:
        Randomly rounded integer
    """
    floor_val = int(value)
    remainder = value - floor_val
    return floor_val + (1 if rng.random() < remainder else 0)


def _process_chunk_pass1(args):
    """Process a chunk for Pass 1 (counting) in parallel using vectorized operations."""
    chunk_df, start_timestamp, input_boundaries, output_boundaries = args

    local_counts = defaultdict(int)

    # Vectorized categorization
    context_tokens_arr = chunk_df['ContextTokens'].values
    generated_tokens_arr = chunk_df['GeneratedTokens'].values
    input_cats, output_cats = categorize_requests_vectorized(
        context_tokens_arr, generated_tokens_arr,
        input_boundaries, output_boundaries
    )

    # Process timestamps (vectorized pandas operations)
    timestamps = pd.to_datetime(chunk_df['TIMESTAMP'], format='mixed')
    seconds_from_start = (timestamps - start_timestamp).dt.total_seconds()
    minute_bins = (seconds_from_start // 60).astype(int).values

    # Count using numpy operations
    for i in range(len(chunk_df)):
        bin_key = (minute_bins[i], input_cats[i], output_cats[i])
        local_counts[bin_key] += 1

    return local_counts


def subsample_trace(
    input_path: str,
    output_path: str,
    sample_rate: float = 0.5,
    time_window_seconds: float = 60.0,
    seed: Optional[int] = None,
    generate_plots: bool = True,
    n_workers: int = 1,
) -> Dict[str, Any]:
    """Subsample Azure trace while preserving temporal distribution.

    Args:
        input_path: Path to input CSV trace file
        output_path: Path for output subsampled CSV
        sample_rate: Target sampling rate (default: 0.5 for 50%)
        time_window_seconds: Time granularity for temporal binning (default: 60.0)
        seed: Random seed for reproducibility (default: None)
        generate_plots: Whether to generate validation plots (default: True)
        n_workers: Number of parallel workers (default: 1 for sequential)

    Returns:
        Dictionary with statistics:
        - original_count: Total requests in original trace
        - subsampled_count: Total requests in subsampled trace
        - actual_rate: Actual sampling rate achieved
        - bin_boundaries: Dict with input/output percentile boundaries
        - processing_time: Time taken to process
    """
    start_time_proc = time.time()

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize random number generator
    rng = np.random.default_rng(seed)

    print("=" * 70)
    print(f"Trace Subsampling Utility (Parallel)")
    print("=" * 70)
    print(f"Input:  {input_path.name}")
    print(f"Output: {output_path.name}")
    print(f"Target sampling rate: {sample_rate:.1%}")
    print(f"Time window: {time_window_seconds:.0f} seconds")
    print(f"Random seed: {seed}")
    print(f"Workers: {n_workers}")

    # Step 1: Compute percentile boundaries
    boundaries = compute_percentile_boundaries(str(input_path), n_workers=n_workers)
    input_boundaries = boundaries['input']
    output_boundaries = boundaries['output']

    # Step 2: Pass 1 - Count requests per bin (with parallel processing)
    print(f"\n[Pass 1] Counting requests per time-category bin (parallel)...")

    # First, read first chunk to get start_timestamp
    dtype_dict = {'ContextTokens': 'int32', 'GeneratedTokens': 'int32'}
    chunk_size = 100000

    first_chunk = pd.read_csv(input_path, dtype=dtype_dict, nrows=1)
    _, start_timestamp = get_minute_bin(first_chunk.iloc[0]['TIMESTAMP'], None)
    print(f"  Start timestamp: {start_timestamp}")

    # Now process all chunks in parallel
    bin_counts = defaultdict(int)
    rows_processed = 0

    if n_workers > 1:
        # Parallel processing
        chunks = []
        for chunk in pd.read_csv(input_path, dtype=dtype_dict, chunksize=chunk_size):
            chunks.append((chunk, start_timestamp, input_boundaries, output_boundaries))

        print(f"  Processing {len(chunks)} chunks with {n_workers} workers...")

        with Pool(processes=n_workers) as pool:
            chunk_results = pool.map(_process_chunk_pass1, chunks)

        # Merge results
        for local_counts in chunk_results:
            for bin_key, count in local_counts.items():
                bin_counts[bin_key] += count
                rows_processed += count

        print(f"  Parallel processing complete!")
    else:
        # Sequential processing (vectorized)
        for chunk in pd.read_csv(input_path, dtype=dtype_dict, chunksize=chunk_size):
            # Vectorized categorization
            context_tokens_arr = chunk['ContextTokens'].values
            generated_tokens_arr = chunk['GeneratedTokens'].values
            input_cats, output_cats = categorize_requests_vectorized(
                context_tokens_arr, generated_tokens_arr,
                input_boundaries, output_boundaries
            )

            # Vectorized timestamp processing
            timestamps = pd.to_datetime(chunk['TIMESTAMP'], format='mixed')
            seconds_from_start = (timestamps - start_timestamp).dt.total_seconds()
            minute_bins = (seconds_from_start // 60).astype(int).values

            # Count bins
            for i in range(len(chunk)):
                bin_key = (minute_bins[i], input_cats[i], output_cats[i])
                bin_counts[bin_key] += 1

            rows_processed += len(chunk)
            if rows_processed % 1000000 == 0:
                print(f"  Processed {rows_processed:,} rows...")

    original_count = rows_processed
    print(f"  Total requests: {original_count:,}")

    # Step 3: Determine target counts per bin
    print(f"\n[Pass 1.5] Determining target sample counts per bin...")

    bin_targets = {}
    total_target = 0

    for bin_key, count in bin_counts.items():
        target = round_randomly(count * sample_rate, rng)
        bin_targets[bin_key] = target
        total_target += target

    print(f"  Target total: {total_target:,} requests ({total_target/original_count:.1%})")

    # Print category distribution
    print(f"\n  Category distribution (original):")
    category_totals = defaultdict(int)
    for (minute_bin, input_cat, output_cat), count in bin_counts.items():
        category_totals[(input_cat, output_cat)] += count

    # 5x5 grid with percentile labels
    labels = ['0-30%', '30-50%', '50-70%', '70-90%', '90-100%']
    print(f"  {'':>8} | {'0-30%':>10} | {'30-50%':>10} | {'50-70%':>10} | {'70-90%':>10} | {'90-100%':>10}")
    print(f"  {'-'*8}|{'-'*12}|{'-'*12}|{'-'*12}|{'-'*12}|{'-'*12}")
    for input_cat, input_label in enumerate(labels):
        row_str = f"  {input_label:>8} |"
        for output_cat in range(5):
            count = category_totals.get((input_cat, output_cat), 0)
            row_str += f" {count:>10,} |"
        print(row_str)

    # Step 4: Pass 2 - Sample and write
    print(f"\n[Pass 2] Sampling requests and writing to output...")

    bin_selected = defaultdict(int)  # Track how many selected per bin
    bin_seen = defaultdict(int)  # Track how many seen per bin
    start_timestamp = None
    rows_processed = 0
    rows_written = 0

    # Initialize systematic sampling: track next selection position for each bin
    bin_next_select = {}  # Maps bin_key -> next position to select (float)
    for bin_key, target_count in bin_targets.items():
        if target_count > 0:
            total_count = bin_counts[bin_key]
            sample_interval = total_count / target_count
            # Random start offset in [0, sample_interval) for each bin
            bin_next_select[bin_key] = rng.uniform(0, sample_interval)

    # Open output file for writing
    with open(output_path, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['TIMESTAMP', 'ContextTokens', 'GeneratedTokens'])

        for chunk in pd.read_csv(input_path, dtype=dtype_dict, chunksize=chunk_size):
            # Vectorized preprocessing of chunk
            timestamp_strs = chunk['TIMESTAMP'].values
            context_tokens_arr = chunk['ContextTokens'].values
            generated_tokens_arr = chunk['GeneratedTokens'].values

            # Vectorized timestamp parsing
            if start_timestamp is None:
                timestamps = pd.to_datetime(chunk['TIMESTAMP'], format='mixed')
                start_timestamp = timestamps.iloc[0]
                seconds_from_start = (timestamps - start_timestamp).dt.total_seconds()
            else:
                timestamps = pd.to_datetime(chunk['TIMESTAMP'], format='mixed')
                seconds_from_start = (timestamps - start_timestamp).dt.total_seconds()

            minute_bins = (seconds_from_start // 60).astype(int).values

            # Vectorized categorization
            input_cats, output_cats = categorize_requests_vectorized(
                context_tokens_arr, generated_tokens_arr,
                input_boundaries, output_boundaries
            )

            # Process each row (systematic sampling must be sequential)
            for i in range(len(chunk)):
                timestamp_str = timestamp_strs[i]
                context_tokens = context_tokens_arr[i]
                generated_tokens = generated_tokens_arr[i]
                minute_bin = minute_bins[i]
                input_cat = input_cats[i]
                output_cat = output_cats[i]

                bin_key = (minute_bin, input_cat, output_cat)

                # Track that we've seen this request
                bin_seen[bin_key] += 1

                # Systematic sampling: select at regular intervals
                target_count = bin_targets.get(bin_key, 0)

                if bin_key in bin_next_select and target_count > 0:
                    bin_position = bin_seen[bin_key] - 1  # 0-indexed position (incremented above)
                    next_select = bin_next_select[bin_key]

                    # Check if this position should be selected
                    if bin_position >= next_select:
                        # Select this request
                        writer.writerow([timestamp_str, context_tokens, generated_tokens])
                        bin_selected[bin_key] += 1
                        rows_written += 1

                        # Calculate next selection position
                        total_count = bin_counts[bin_key]
                        sample_interval = total_count / target_count
                        bin_next_select[bin_key] += sample_interval

            rows_processed += len(chunk)
            if rows_processed % 1000000 == 0:
                print(f"  Processed {rows_processed:,} rows, wrote {rows_written:,} rows...")

    actual_rate = rows_written / original_count

    print(f"\n  Subsampling complete!")
    print(f"    Original:   {original_count:,} requests")
    print(f"    Subsampled: {rows_written:,} requests")
    print(f"    Actual rate: {actual_rate:.2%}")

    # Step 5: Generate validation plots
    if generate_plots:
        print(f"\n[Step 3] Generating validation plots...")
        plot_dir = output_path.parent
        generate_validation_plots(
            str(input_path),
            str(output_path),
            str(plot_dir),
            time_window_seconds=300.0  # 5-minute windows for smoother plots
        )

    processing_time = time.time() - start_time_proc
    print(f"\n{'='*70}")
    print(f"Total processing time: {processing_time:.1f} seconds")
    print(f"{'='*70}\n")

    return {
        'original_count': original_count,
        'subsampled_count': rows_written,
        'actual_rate': actual_rate,
        'bin_boundaries': boundaries,
        'processing_time': processing_time,
    }


def generate_validation_plots(
    original_path: str,
    subsampled_path: str,
    output_dir: str,
    time_window_seconds: float = 300.0
) -> None:
    """Generate validation plots comparing original vs subsampled traces.

    Creates:
    - Input throughput over time (original vs subsampled)
    - Output throughput over time (original vs subsampled)
    - Request count over time (original vs subsampled vs expected)

    Args:
        original_path: Path to original trace CSV
        subsampled_path: Path to subsampled trace CSV
        output_dir: Directory to save plots
        time_window_seconds: Time window for throughput aggregation
    """
    output_dir = Path(output_dir)
    subsampled_name = Path(subsampled_path).stem

    # Use Agg backend for headless environments
    plt.switch_backend('Agg')

    print(f"  Computing throughput time series...")

    # Helper function to compute throughput over time
    def compute_throughput(trace_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (time_bins_hours, input_throughput, output_throughput)."""
        time_bins = defaultdict(lambda: {'input': 0, 'output': 0})
        start_time = None

        chunk_size = 100000
        dtype_dict = {'ContextTokens': 'int32', 'GeneratedTokens': 'int32'}

        for chunk in pd.read_csv(trace_path, dtype=dtype_dict, chunksize=chunk_size):
            for _, row in chunk.iterrows():
                timestamp_str = row['TIMESTAMP']

                # Parse timestamp
                if '.' in timestamp_str and '+' in timestamp_str:
                    parts = timestamp_str.split('.')
                    microsec_and_tz = parts[1].split('+')
                    microseconds = microsec_and_tz[0][:6]
                    timezone = microsec_and_tz[1]
                    timestamp_str = f"{parts[0]}.{microseconds}+{timezone}"

                timestamp = pd.to_datetime(timestamp_str)

                if start_time is None:
                    start_time = timestamp

                # Get time bin
                seconds_from_start = (timestamp - start_time).total_seconds()
                bin_idx = int(seconds_from_start // time_window_seconds)

                # Accumulate tokens
                time_bins[bin_idx]['input'] += row['ContextTokens']
                time_bins[bin_idx]['output'] += row['GeneratedTokens']

        # Convert to arrays
        max_bin = max(time_bins.keys()) if time_bins else 0
        time_hours = []
        input_tp = []
        output_tp = []

        for bin_idx in range(max_bin + 1):
            time_hours.append((bin_idx * time_window_seconds) / 3600.0)  # Convert to hours
            input_tp.append(time_bins[bin_idx]['input'] / time_window_seconds)
            output_tp.append(time_bins[bin_idx]['output'] / time_window_seconds)

        return np.array(time_hours), np.array(input_tp), np.array(output_tp)

    # Compute throughput for both traces
    print(f"    Processing original trace...")
    orig_time, orig_input_tp, orig_output_tp = compute_throughput(original_path)

    print(f"    Processing subsampled trace...")
    sub_time, sub_input_tp, sub_output_tp = compute_throughput(subsampled_path)

    # Plot 1: Input Throughput
    print(f"    Generating input throughput plot...")
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(orig_time, orig_input_tp, label='Original', linewidth=2, alpha=0.7, color='#1f77b4')
    ax.plot(sub_time, sub_input_tp, label='Subsampled', linewidth=2, alpha=0.7,
            linestyle='--', color='#ff7f0e')

    ax.set_xlabel('Time (hours from start)', fontsize=12)
    ax.set_ylabel('Input Throughput (tokens/second)', fontsize=12)
    ax.set_title('Input Throughput Comparison: Original vs Subsampled', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / f"{subsampled_name}_input_throughput.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {output_file.name}")

    # Plot 2: Output Throughput
    print(f"    Generating output throughput plot...")
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(orig_time, orig_output_tp, label='Original', linewidth=2, alpha=0.7, color='#1f77b4')
    ax.plot(sub_time, sub_output_tp, label='Subsampled', linewidth=2, alpha=0.7,
            linestyle='--', color='#ff7f0e')

    ax.set_xlabel('Time (hours from start)', fontsize=12)
    ax.set_ylabel('Output Throughput (tokens/second)', fontsize=12)
    ax.set_title('Output Throughput Comparison: Original vs Subsampled', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / f"{subsampled_name}_output_throughput.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {output_file.name}")

    # Plot 3: Request Count Over Time
    print(f"    Generating request count plot...")

    def compute_request_counts(trace_path: str, window_sec: float = 60.0) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (time_bins_hours, request_counts)."""
        time_bins = defaultdict(int)
        start_time = None

        chunk_size = 100000
        dtype_dict = {'ContextTokens': 'int32', 'GeneratedTokens': 'int32'}

        for chunk in pd.read_csv(trace_path, dtype=dtype_dict, chunksize=chunk_size):
            for _, row in chunk.iterrows():
                timestamp_str = row['TIMESTAMP']

                # Parse timestamp (same logic as compute_throughput)
                if '.' in timestamp_str and '+' in timestamp_str:
                    parts = timestamp_str.split('.')
                    microsec_and_tz = parts[1].split('+')
                    microseconds = microsec_and_tz[0][:6]
                    timezone = microsec_and_tz[1]
                    timestamp_str = f"{parts[0]}.{microseconds}+{timezone}"

                timestamp = pd.to_datetime(timestamp_str)

                if start_time is None:
                    start_time = timestamp

                # Get time bin
                seconds_from_start = (timestamp - start_time).total_seconds()
                bin_idx = int(seconds_from_start // window_sec)

                # Count requests
                time_bins[bin_idx] += 1

        # Convert to arrays
        max_bin = max(time_bins.keys()) if time_bins else 0
        time_hours = []
        counts = []

        for bin_idx in range(max_bin + 1):
            time_hours.append((bin_idx * window_sec) / 3600.0)  # Convert to hours
            counts.append(time_bins[bin_idx])

        return np.array(time_hours), np.array(counts)

    # Compute request counts (using 1-minute windows for this plot)
    print(f"    Processing original trace for request counts...")
    orig_count_time, orig_counts = compute_request_counts(original_path, window_sec=60.0)

    print(f"    Processing subsampled trace for request counts...")
    sub_count_time, sub_counts = compute_request_counts(subsampled_path, window_sec=60.0)

    # Calculate expected counts based on sample rate
    # Better: calculate from actual data
    total_orig = np.sum(orig_counts)
    total_sub = np.sum(sub_counts)
    actual_sample_rate = total_sub / total_orig if total_orig > 0 else 0

    expected_counts = orig_counts * actual_sample_rate

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(orig_count_time, orig_counts, label='Original', linewidth=2, alpha=0.7, color='#1f77b4')
    ax.plot(sub_count_time, sub_counts, label='Subsampled', linewidth=2, alpha=0.7,
            linestyle='--', color='#ff7f0e')
    ax.plot(orig_count_time, expected_counts, label=f'Expected ({actual_sample_rate:.1%} of original)',
            linewidth=1.5, alpha=0.5, linestyle=':', color='#2ca02c')

    ax.set_xlabel('Time (hours from start)', fontsize=12)
    ax.set_ylabel('Request Count (per minute)', fontsize=12)
    ax.set_title('Request Count Comparison: Original vs Subsampled', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / f"{subsampled_name}_request_count.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {output_file.name}")


def main():
    """CLI for trace subsampling."""
    parser = argparse.ArgumentParser(
        description='Subsample Azure LLM trace while preserving temporal distribution',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input trace CSV file'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path for output subsampled CSV file'
    )

    parser.add_argument(
        '--rate',
        type=float,
        default=0.5,
        help='Target sampling rate (0.0 to 1.0)'
    )

    parser.add_argument(
        '--time-window',
        type=float,
        default=60.0,
        help='Time window in seconds for temporal binning'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip generating validation plots'
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=-1,
        help='Number of parallel workers (default: -1 for auto (half of CPUs), 1 for sequential)'
    )

    args = parser.parse_args()

    # Validate arguments
    if not 0.0 < args.rate <= 1.0:
        print(f"Error: Sampling rate must be between 0 and 1, got {args.rate}")
        sys.exit(1)

    if args.time_window <= 0:
        print(f"Error: Time window must be positive, got {args.time_window}")
        sys.exit(1)

    # Determine number of workers
    n_workers = args.workers
    if n_workers == -1:
        # Use half of available CPUs by default for better performance
        n_workers = max(1, cpu_count() // 2)
        print(f"Auto-detected {cpu_count()} CPUs, using {n_workers} workers")
    elif n_workers < 1:
        print(f"Error: Number of workers must be positive or -1, got {args.workers}")
        sys.exit(1)

    # Run subsampling
    try:
        stats = subsample_trace(
            input_path=args.input,
            output_path=args.output,
            sample_rate=args.rate,
            time_window_seconds=args.time_window,
            seed=args.seed,
            generate_plots=not args.no_plots,
            n_workers=n_workers,
        )

        print(f"Success! Subsampled trace saved to: {args.output}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
