#!/usr/bin/env python3
"""Generate validation plots for subsampled traces."""

import sys
from utils.trace_subsampler import generate_validation_plots

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python generate_plots.py <original_csv> <subsampled_csv> [output_dir] [time_window]")
        sys.exit(1)

    original_path = sys.argv[1]
    subsampled_path = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else '.'
    time_window = float(sys.argv[4]) if len(sys.argv) > 4 else 300.0

    print(f"Generating validation plots...")
    print(f"  Original: {original_path}")
    print(f"  Subsampled: {subsampled_path}")
    print(f"  Output dir: {output_dir}")
    print(f"  Time window: {time_window}s")

    generate_validation_plots(
        original_path,
        subsampled_path,
        output_dir,
        time_window_seconds=time_window
    )

    print("Done!")
