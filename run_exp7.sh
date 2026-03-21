#!/bin/bash
# Script to run Experiment 7 on CPU node

echo "=========================================="
echo "Running Experiment 7: Switching + Offload at 1P7D"
echo "Trace: 24-hour Azure code trace (~2.5M requests)"
echo "Configurations: 12 (4 policies x 3 offload modes)"
echo "=========================================="
echo ""

# Navigate to project directory
cd /work/nvme/bcjw/kfu2/sglang-simulation

# Check trace file
if [ ! -f "azure_code_24h.csv" ]; then
    echo "ERROR: azure_code_24h.csv not found!"
    exit 1
fi

echo "Trace file found: $(ls -lh azure_code_24h.csv | awk '{print $5}')"
echo ""

# Run experiment
echo "Starting experiment at $(date)..."
python experiments/run_experiment7_only.py

# Check results
echo ""
echo "=========================================="
echo "Experiment completed at $(date)"
echo "=========================================="

if [ -f "experiments/results/exp7_switching_offload_1p7d.json" ]; then
    echo "✓ Results saved to experiments/results/exp7_switching_offload_1p7d.json"
    echo ""
    echo "Result file size: $(ls -lh experiments/results/exp7_switching_offload_1p7d.json | awk '{print $5}')"
else
    echo "✗ Results file not found!"
fi
