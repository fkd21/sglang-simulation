"""Trace subsampling utility for Azure LLM inference traces.

Subsamples traces while preserving temporal distribution across request categories.
Categorizes requests into 9 bins (3x3 input/output: short/medium/long) and maintains
the distribution within configurable time windows.
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
) -> Dict[str, Tuple[float, float]]:
    """Compute 33rd and 66th percentiles for input/output tokens using streaming.

    Args:
        input_path: Path to CSV trace file
        chunk_size: Number of rows to read per chunk for memory efficiency
        n_workers: Number of parallel workers (default: 1 for sequential)

    Returns:
        Dictionary with boundaries:
        {
            'input': (p33, p66),
            'output': (p33, p66)
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

    input_p33, input_p66 = np.percentile(input_tokens, [33, 66])
    output_p33, output_p66 = np.percentile(output_tokens, [33, 66])

    boundaries = {
        'input': (float(input_p33), float(input_p66)),
        'output': (float(output_p33), float(output_p66))
    }

    print(f"\n  Percentile boundaries computed:")
    print(f"    Input tokens:  p33={input_p33:.0f}, p66={input_p66:.0f}")
    print(f"    Output tokens: p33={output_p33:.0f}, p66={output_p66:.0f}")
    print(f"    Total requests analyzed: {len(input_tokens):,}")

    return boundaries


def categorize_request(
    context_tokens: int,
    generated_tokens: int,
    input_boundaries: Tuple[float, float],
    output_boundaries: Tuple[float, float]
) -> Tuple[int, int]:
    """Categorize a request into input/output bins (0=short, 1=medium, 2=long).

    Args:
        context_tokens: Input token count
        generated_tokens: Output token count
        input_boundaries: (p33, p66) for input tokens
        output_boundaries: (p33, p66) for output tokens

    Returns:
        Tuple of (input_bin, output_bin) where each is 0, 1, or 2
    """
    # Input categorization
    if context_tokens <= input_boundaries[0]:
        input_bin = 0  # Short
    elif context_tokens <= input_boundaries[1]:
        input_bin = 1  # Medium
    else:
        input_bin = 2  # Long

    # Output categorization
    if generated_tokens <= output_boundaries[0]:
        output_bin = 0  # Short
    elif generated_tokens <= output_boundaries[1]:
        output_bin = 1  # Medium
    else:
        output_bin = 2  # Long

    return input_bin, output_bin


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
    """Process a chunk for Pass 1 (counting) in parallel."""
    chunk_df, start_timestamp, input_boundaries, output_boundaries = args

    local_counts = defaultdict(int)

    for _, row in chunk_df.iterrows():
        timestamp_str = row['TIMESTAMP']
        context_tokens = row['ContextTokens']
        generated_tokens = row['GeneratedTokens']

        # Get minute bin
        minute_bin, _ = get_minute_bin(timestamp_str, start_timestamp)

        # Categorize request
        input_cat, output_cat = categorize_request(
            context_tokens, generated_tokens,
            input_boundaries, output_boundaries
        )

        # Increment counter
        bin_key = (minute_bin, input_cat, output_cat)
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
        # Sequential processing
        for chunk in pd.read_csv(input_path, dtype=dtype_dict, chunksize=chunk_size):
            for _, row in chunk.iterrows():
                timestamp_str = row['TIMESTAMP']
                context_tokens = row['ContextTokens']
                generated_tokens = row['GeneratedTokens']

                minute_bin, _ = get_minute_bin(timestamp_str, start_timestamp)

                input_cat, output_cat = categorize_request(
                    context_tokens, generated_tokens,
                    input_boundaries, output_boundaries
                )

                bin_key = (minute_bin, input_cat, output_cat)
                bin_counts[bin_key] += 1

                rows_processed += 1
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

    print(f"  {'':>6} | {'Short':>10} | {'Medium':>10} | {'Long':>10}")
    print(f"  {'-'*6}|{'-'*12}|{'-'*12}|{'-'*12}")
    for input_cat, input_label in enumerate(['Short', 'Medium', 'Long']):
        row_str = f"  {input_label:>6} |"
        for output_cat in range(3):
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

    # Open output file for writing
    with open(output_path, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['TIMESTAMP', 'ContextTokens', 'GeneratedTokens'])

        for chunk in pd.read_csv(input_path, dtype=dtype_dict, chunksize=chunk_size):
            for _, row in chunk.iterrows():
                timestamp_str = row['TIMESTAMP']
                context_tokens = row['ContextTokens']
                generated_tokens = row['GeneratedTokens']

                # Get minute bin
                if start_timestamp is None:
                    minute_bin, start_timestamp = get_minute_bin(timestamp_str, None)
                else:
                    minute_bin, _ = get_minute_bin(timestamp_str, start_timestamp)

                # Categorize request
                input_cat, output_cat = categorize_request(
                    context_tokens, generated_tokens,
                    input_boundaries, output_boundaries
                )

                bin_key = (minute_bin, input_cat, output_cat)

                # Track that we've seen this request
                bin_seen[bin_key] += 1

                # Check if we should select this request
                selected_count = bin_selected[bin_key]
                target_count = bin_targets.get(bin_key, 0)

                # Reservoir sampling: select with probability remaining_needed / remaining_total
                if selected_count < target_count:
                    remaining_needed = target_count - selected_count
                    remaining_total = bin_counts[bin_key] - bin_seen[bin_key] + 1  # +1 for current

                    select_probability = remaining_needed / max(remaining_total, 1)

                    if rng.random() < select_probability:
                        writer.writerow([timestamp_str, context_tokens, generated_tokens])
                        bin_selected[bin_key] += 1
                        rows_written += 1

                rows_processed += 1
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
        default=1,
        help='Number of parallel workers (default: 1 for sequential, -1 for all CPUs)'
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
        n_workers = cpu_count()
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
