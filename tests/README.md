# Simulation Framework Unit Tests

Comprehensive unit test suite for the P/D-disaggregated LLM inference simulator.

## Test Coverage

### Core Components (74 tests total)

#### 1. **Profiling Formulas** (`test_profiling_formulas.py`) - 10 tests
Tests the regression formulas for inference time and KV transfer latency:
- `TestInferenceTimeFormula`: Validates profiling formula for prefill, decode, and mixed batches
  - Formula: `t = 0.009076 + 7.063e-5 * prefill_tokens + 5.623e-5 * decode_bs + 6.926e-8 * decode_token_sum`
  - Edge cases: zero inputs, large batches, pure prefill/decode
- `TestKVTransferFormula`: Validates KV cache transfer formula
  - Formula: `t = 1.86683e-6 * tokens`
  - Tests small, large, and typical context sizes

#### 2. **Memory Pools** (`test_memory_pools.py`) - 12 tests
Tests memory allocation and management:
- `TestTokenToKVPool`: KV cache token allocation
  - Allocation/deallocation correctness
  - Memory pressure handling
  - Free slot tracking
- `TestReqToTokenPool`: Request slot management
  - Request pool index allocation
  - Slot reuse after freeing
  - Capacity limits

#### 3. **Radix Cache** (`test_radix_cache.py`) - 13 tests
Tests prefix caching for reducing redundant computation:
- `TestSimTreeNode`: Tree node data structure
- `TestRadixCache`: Prefix matching and eviction
  - Exact/partial prefix matching
  - Multi-branch caching
  - LRU eviction
  - Edge cases (empty sequences, single tokens)

#### 4. **Request Lifecycle** (`test_request.py`) - 10 tests
Tests request state management:
- `TestSimReq`: Request initialization, state transitions, and tracking
  - Auto-generation of unique token IDs per request
  - Stage transitions (WAITING → PREFILL → TRANSFER → DECODE → COMPLETED)
  - Latency timestamp tracking (TTFT, ITL, E2E)
  - Memory allocation tracking
  - Prefix matching integration
  - New mechanism tracking (offloading, continuation)

#### 5. **Batch Operations** (`test_batch.py`) - 10 tests
Tests batch creation and metrics computation:
- `TestSimBatch`: Batch construction and forward mode handling
  - Prefill, decode, and mixed batch metrics
  - Inference time calculation using profiling formulas
  - ForwardMode enum validation

#### 6. **Event System** (`test_event_ordering.py`) - 4 tests
Tests discrete-event simulation infrastructure:
- `TestEventOrdering`: Event queue and chronological ordering
  - Timestamp-based ordering
  - Priority handling for simultaneous events
  - Event data preservation

#### 7. **Metrics Collection** (`test_metrics_collector.py`) - 11 tests
Tests SLA tracking and performance metrics:
- `TestMetricsCollector`: TTFT/ITL SLA calculation
  - TTFT SLA: ≤ 1.0s (Time To First Token)
  - ITL SLA: ≤ 100ms (Inter-Token Latency)
  - Overall SLA: both must be met
  - Utilization tracking (prefill/decode instances)
  - Throughput calculation
  - Mechanism tracking (offloading, continuation, switching)

#### 8. **Admission Control** (`test_prefill_adder.py`) - 8 tests
Tests PrefillAdder for batch scheduling:
- `TestPrefillAdder`: Request admission and resource management
  - Token budget tracking
  - Memory availability checking
  - Automatic eviction when memory is full
  - Multiple request handling

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_profiling_formulas.py -v

# Run with coverage
python -m pytest tests/ --cov --cov-report=html

# Run tests matching a pattern
python -m pytest tests/ -k "test_sla" -v
```

## Test Results Summary

```
============== 74 passed in 0.64s ==============

✅ Profiling Formulas: 10/10 passed
✅ Memory Pools: 12/12 passed
✅ Radix Cache: 13/13 passed
✅ Request Lifecycle: 10/10 passed
✅ Batch Operations: 10/10 passed
✅ Event System: 4/4 passed
✅ Metrics Collection: 11/11 passed
✅ Admission Control: 8/8 passed
```

## Key Test Scenarios

### SLA Validation
Tests verify correct calculation of:
1. **TTFT (Time To First Token)**: `prefill_end_time - arrival_time ≤ 1.0s`
2. **ITL (Inter-Token Latency)**: `(decode_end_time - decode_start_time) / tokens_generated ≤ 100ms`
3. **Overall SLA**: Both TTFT and ITL must meet thresholds

### Memory Management
- Tests verify that eviction correctly frees memory when pool is full
- Tests ensure requests get unique KV cache indices
- Tests validate prefix cache reduces redundant computation

### Formula Accuracy
- Validates profiling formula matches expected behavior
- Tests edge cases (zero inputs, max batch sizes)
- Verifies KV transfer time scales linearly with tokens

## Debugging with Tests

The test suite is designed for debugging:

1. **Isolate components**: Each test file focuses on one component
2. **Clear test names**: Test names describe what is being validated
3. **Edge case coverage**: Tests include boundary conditions and error cases
4. **Fixtures**: `conftest.py` provides reusable test fixtures

## Future Test Additions

When implementing the three new mechanisms, add tests for:

1. **PartialPrefillOffloader** (β parameter)
   - Test β ∈ [0.0, 1.0] behavior
   - Verify work distribution between prefill/decode instances
   - Test memory allocation for split batches

2. **DecodeContinuation** (M parameter)
   - Test M token continuation on prefill instances
   - Verify decision logic (idle prefill, short remaining work)
   - Test interaction with normal decode flow

3. **RoleSwitcher**
   - Test role switching triggers
   - Verify queue migration
   - Test switching overhead

## Test Maintenance

- Update tests when modifying core implementations
- Add regression tests when fixing bugs
- Keep test execution time under 1 second for rapid iteration
