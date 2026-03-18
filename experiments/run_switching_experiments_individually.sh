#!/bin/bash
# Run switching policy experiments one at a time to avoid memory issues
# Each experiment runs in a separate Python process

set -e

cd "$(dirname "$0")"

echo "=========================================="
echo "Running Switching Policy Experiments"
echo "One experiment at a time to manage memory"
echo "=========================================="

# Run each experiment individually
python3 -c "
import sys
sys.path.insert(0, '../..')
from simulation.experiments.run_switching_policy_experiments import *

print('\n' + '='*70)
print('EXPERIMENT 1: Baseline Comparison')
print('='*70)
r1 = experiment_baseline_comparison()
save_experiment('switching_exp1_baseline', r1)
print('Experiment 1 complete!')
"

python3 -c "
import sys
sys.path.insert(0, '../..')
from simulation.experiments.run_switching_policy_experiments import *

print('\n' + '='*70)
print('EXPERIMENT 2: Switching + Offload Interaction')
print('='*70)
r2 = experiment_switching_offload_interaction()
save_experiment('switching_exp2_interaction', r2)
print('Experiment 2 complete!')
"

python3 -c "
import sys
sys.path.insert(0, '../..')
from simulation.experiments.run_switching_policy_experiments import *

print('\n' + '='*70)
print('EXPERIMENT 3: Instance Configuration Scaling')
print('='*70)
r3 = experiment_instance_config_scaling()
save_experiment('switching_exp3_instance_config', r3)
print('Experiment 3 complete!')
"

python3 -c "
import sys
sys.path.insert(0, '../..')
from simulation.experiments.run_switching_policy_experiments import *

print('\n' + '='*70)
print('EXPERIMENT 4: Load Level Sensitivity')
print('='*70)
r4 = experiment_load_level()
save_experiment('switching_exp4_load_level', r4)
print('Experiment 4 complete!')
"

# Combine all results
python3 -c "
import json
from pathlib import Path

RESULTS_DIR = Path('results')
all_results = {}

for exp_name in ['exp1_baseline', 'exp2_interaction', 'exp3_instance_config', 'exp4_load_level']:
    path = RESULTS_DIR / f'switching_{exp_name}.json'
    if path.exists():
        with open(path) as f:
            all_results[exp_name] = json.load(f)

combined_path = RESULTS_DIR / 'switching_all_experiments.json'
with open(combined_path, 'w') as f:
    json.dump(all_results, f, indent=2)

print('\n' + '='*70)
print(f'All results combined and saved to {combined_path}')
print('='*70)
print('Done!')
"

echo ""
echo "=========================================="
echo "All experiments complete!"
echo "=========================================="
