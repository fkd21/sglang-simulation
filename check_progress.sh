#!/bin/bash
# Script to check experiment progress

echo "=========================================="
echo "Experiment 7 Progress Check"
echo "=========================================="
echo ""

# Check if job is running
echo "Checking srun job status..."
squeue -u $USER | grep -E "JOBID|bcjw-delta-cpu" || echo "No active jobs found in queue"
echo ""

# Check output file
OUTPUT_FILE="/tmp/claude/-work-nvme-bcjw-kfu2-sglang-simulation/tasks/b817d67.output"
if [ -f "$OUTPUT_FILE" ]; then
    echo "Last 20 lines of output:"
    echo "----------------------------------------"
    tail -20 "$OUTPUT_FILE"
    echo "----------------------------------------"
else
    echo "Output file not found: $OUTPUT_FILE"
fi

echo ""
echo "Results directory:"
ls -lh experiments/results/ 2>/dev/null || echo "Results directory not found yet"
