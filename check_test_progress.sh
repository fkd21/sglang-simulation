#!/bin/bash
# Check test run progress

echo "=========================================="
echo "Test Run Progress Check"
echo "=========================================="
echo ""

# Check if job is running
echo "Job Status:"
squeue -u $USER | grep -E "JOBID|cpu" || echo "  No active jobs"
echo ""

# Check output file
OUTPUT_FILE="/tmp/claude/-work-nvme-bcjw-kfu2-sglang-simulation/tasks/b2b558d.output"
if [ -f "$OUTPUT_FILE" ]; then
    echo "Recent Output (last 30 lines):"
    echo "----------------------------------------"
    tail -30 "$OUTPUT_FILE"
    echo "----------------------------------------"
    echo ""
    echo "Total lines: $(wc -l < "$OUTPUT_FILE")"
else
    echo "Output file not found yet: $OUTPUT_FILE"
fi
