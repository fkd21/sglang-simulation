# Per-Iteration Inference Statistics Logging

## Overview

The iteration logging system records detailed statistics for **every batch execution** during simulation runs. This provides fine-grained visibility into:

- Batch formation decisions
- Inference timing
- Per-request processing details
- Prefix cache effectiveness

## Quick Start

### Enable Logging

```python
from simulation.core.engine import SimulationEngine
from simulation.config import SimConfig

config = SimConfig(
    trace_path="simulation/AzureLLMInferenceTrace_code.csv",
    num_prefill_instances=4,
    num_decode_instances=4
)

# Enable iteration logging
engine = SimulationEngine(config, enable_iteration_logging=True)
results = engine.run()
```

### Output Files

Results are saved to `simulation/result/` directory:

- **`{run_name}_prefill.jsonl`** - All prefill batch executions
- **`{run_name}_decode.jsonl`** - All decode batch executions

Default run name: `{num_prefill}P{num_decode}D` (e.g., `4P4D`)

## Record Format

Each line in the JSONL file contains:

```json
{
  "iteration": 0,
  "timestamp": 1.234,
  "instance_id": "prefill_0",
  "instance_type": "prefill",
  "forward_mode": "PREFILL",
  "inference_time": 0.0791,
  "batch_size": 5,
  "total_prefill_tokens": 1200,
  "decode_batch_size": 0,
  "decode_computed_token_sum": 0,
  "requests": [
    {
      "rid": "req_0",
      "input_length": 128,
      "output_length": 0,
      "extend_input_len": 128,
      "is_prefill": true,
      "prefix_matched": 0
    }
  ]
}
```

## Fields Description

### Batch-Level Fields

| Field | Type | Description |
|-------|------|-------------|
| `iteration` | int | Global iteration counter |
| `timestamp` | float | Batch completion time (simulation seconds) |
| `instance_id` | str | Instance that executed this batch |
| `instance_type` | str | "prefill" or "decode" |
| `forward_mode` | str | "PREFILL", "DECODE", "EXTEND", or "MIXED" |
| `inference_time` | float | Actual batch execution time (seconds) |
| `batch_size` | int | Number of requests in this batch |
| `total_prefill_tokens` | int | Total prefill tokens processed |
| `decode_batch_size` | int | Number of decode requests |
| `decode_computed_token_sum` | int | Sum of (context + output) for decode |

### Per-Request Fields

| Field | Type | Description |
|-------|------|-------------|
| `rid` | str | Request ID |
| `input_length` | int | Original context tokens |
| `output_length` | int | Output tokens generated so far |
| `extend_input_len` | int | Tokens to process (after prefix match) |
| `is_prefill` | bool | Whether request is in prefill phase |
| `prefix_matched` | int | Tokens matched in prefix cache |

## Example Usage

### Basic Analysis

```python
import json

# Count total batches
with open('simulation/result/4P4D_prefill.jsonl', 'r') as f:
    total_batches = sum(1 for _ in f)
print(f"Total prefill batches: {total_batches}")

# Calculate average batch size
total_reqs = 0
total_batches = 0
with open('simulation/result/4P4D_prefill.jsonl', 'r') as f:
    for line in f:
        record = json.loads(line)
        total_reqs += record['batch_size']
        total_batches += 1

avg_batch_size = total_reqs / total_batches
print(f"Average batch size: {avg_batch_size:.2f}")
```

### Pandas Analysis

```python
import pandas as pd
import json

# Load into DataFrame
records = []
with open('simulation/result/4P4D_prefill.jsonl', 'r') as f:
    records = [json.loads(line) for line in f]

df = pd.DataFrame(records)

# Statistics
print(df[['batch_size', 'inference_time', 'total_prefill_tokens']].describe())

# Plot batch size over time
import matplotlib.pyplot as plt
plt.plot(df['iteration'], df['batch_size'])
plt.xlabel('Iteration')
plt.ylabel('Batch Size')
plt.title('Prefill Batch Size Over Time')
plt.savefig('batch_size.png')
```

### Per-Request Analysis

```python
import json

# Track a specific request through iterations
target_rid = "req_1234"

with open('simulation/result/4P4D_prefill.jsonl', 'r') as f:
    for line in f:
        rec = json.loads(line)
        for req in rec['requests']:
            if req['rid'] == target_rid:
                print(f"Iteration {rec['iteration']}: "
                      f"extend={req['extend_input_len']}, "
                      f"matched={req['prefix_matched']}")
```

### Prefix Cache Hit Rate

```python
import json

total_matched = 0
total_extend = 0

with open('simulation/result/4P4D_prefill.jsonl', 'r') as f:
    for line in f:
        rec = json.loads(line)
        for req in rec['requests']:
            total_matched += req['prefix_matched']
            total_extend += req['extend_input_len']

hit_rate = total_matched / (total_matched + total_extend) * 100
print(f"Prefix cache hit rate: {hit_rate:.1f}%")
```

## Implementation Details

### Location
- **Logger class:** [simulation/results/iteration_logger.py](simulation/results/iteration_logger.py)
- **Integration:** [simulation/core/engine.py](simulation/core/engine.py)
- **Output directory:** `simulation/result/`

### Logging Points
Batches are logged in two places:

1. **`_handle_prefill_batch_complete()`** - After prefill batch completes
2. **`_handle_decode_batch_complete()`** - After decode batch completes

### Performance
- **Overhead:** Minimal (buffered file I/O)
- **File size:** ~1KB per iteration (depends on batch size)
- **Format:** JSONL (streaming-friendly, one record per line)

## Use Cases

### 1. Debug Scheduling
Identify small batches that indicate inefficient scheduling:

```python
for line in open('result/1P1D_prefill.jsonl'):
    rec = json.loads(line)
    if rec['batch_size'] < 5:
        print(f"Small batch at t={rec['timestamp']:.2f}s")
```

### 2. Validate Profiling Formula
Compare estimated vs actual timing (not currently logged, but can be added).

### 3. Analyze Cache Effectiveness
Track prefix matching effectiveness over time:

```python
for line in open('result/4P4D_prefill.jsonl'):
    rec = json.loads(line)
    total_matched = sum(req['prefix_matched'] for req in rec['requests'])
    total_extend = sum(req['extend_input_len'] for req in rec['requests'])
    if total_matched + total_extend > 0:
        hit_rate = total_matched / (total_matched + total_extend)
        print(f"Iter {rec['iteration']}: cache hit rate = {hit_rate:.1%}")
```

### 4. Identify Bottlenecks
Find iterations with unexpectedly long inference times:

```python
for line in open('result/4P4D_prefill.jsonl'):
    rec = json.loads(line)
    if rec['inference_time'] > 0.1:  # > 100ms
        print(f"Slow batch: {rec['batch_size']} reqs, "
              f"{rec['total_prefill_tokens']} tokens, "
              f"{rec['inference_time']:.4f}s")
```

## Example Output

```
IterationLogger initialized:
  Output dir: simulation/result
  Prefill log: 4P4D_prefill.jsonl
  Decode log: 4P4D_decode.jsonl

Starting simulation with 4 prefill and 4 decode instances
All 8819 requests completed!
Simulation completed at t=234.56s

IterationLogger Summary:
  Total iterations: 12345
  Prefill iterations: 6789
  Decode iterations: 5556
  Prefill log: simulation/result/4P4D_prefill.jsonl
  Decode log: simulation/result/4P4D_decode.jsonl
```

## Future Enhancements

Potential additions:
- Log estimated duration vs actual (to validate profiling formula)
- Log memory usage per iteration
- Log queue lengths at batch formation time
- Compress output to reduce file size
- Add CSV export option
- Real-time streaming to analysis dashboard
