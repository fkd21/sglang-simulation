# Performance Analysis and Optimization Results

## Executive Summary

After implementing Phase 2 and Phase 3 optimizations, profiling reveals the **true bottleneck is not queue operations but repeated decode instance capacity calculations**.

## Profiling Results (azure_code_8000.csv, 2P2D, 77.8s wall-clock)

### Top Hotspots by Total Time

| Function | Calls | Total Time | Per Call | % Total |
|----------|-------|------------|----------|---------|
| `_decode_allocatable_tokens()` | **11,142,735** | **35.5s** | 0.000s | **45.6%** |
| `len()` (built-in) | 178,596,425 | 13.6s | 0.000s | 17.5% |
| `_try_admit_from_bootstrap()` | 33,527 | 5.7s | 0.000s | 7.3% |
| `_select_best_decode_instance()` | 3,714,245 | 3.4s | 0.000s | 4.4% |
| `max()` (built-in) | 14,906,150 | 3.3s | 0.000s | 4.2% |

### Key Findings

1. **`_decode_allocatable_tokens()` is called 11 MILLION times**
   - Called once for each decode instance, for each bootstrap admission attempt
   - With 4 decode instances and frequent bootstrap attempts: 4 × 3,714,245 ≈ 14.8M calls
   - Each call iterates over `prealloc_reserved` list (line 586-587)
   - This is **45.6% of total execution time**

2. **`_select_best_decode_instance()` is called 3.7 MILLION times**
   - Called once per bootstrap admission attempt
   - Iterates over all decode instances and calls `_decode_allocatable_tokens()` for each
   - Pattern: O(n_requests × n_decode_instances)

3. **Phase 2/3 optimizations are working**
   - `deque.append()`: 3,729,621 calls (O(1) operations)
   - `deque.popleft()`: 3,722,124 calls (O(1) operations)
   - No more O(n²) `list.remove()` operations visible in profile

## Root Cause Analysis

### Why is `_decode_allocatable_tokens()` so expensive?

```python
def _decode_allocatable_tokens(self, decode_instance: DecodeInstance) -> int:
    available = decode_instance.token_to_kv_pool.available_size()

    # O(n) iteration over prealloc_reserved for EVERY call
    for req in decode_instance.prealloc_reserved:
        available -= req.context_tokens + len(req.output_ids)

    # Count active requests
    num_running = len(decode_instance.running_batch.reqs)
    num_transfer = len(decode_instance.transfer_queue)
    num_active_inflight = num_running + num_transfer

    reserved_tokens = self.config.num_reserved_decode_tokens * num_active_inflight
    return max(0, available - reserved_tokens)
```

**Problem**: This function is called repeatedly during bootstrap admission, but decode instance state rarely changes between calls within a single event processing cycle.

### Why so many calls?

In `_try_admit_from_bootstrap()`:
```python
while prefill_instance.bootstrap_queue:
    req = prefill_instance.bootstrap_queue.popleft()

    # EVERY request calls this, which calls _decode_allocatable_tokens() 4 times!
    decode_instance = self._select_best_decode_instance()

    # ... admission logic ...
```

With bootstrap queue of 200 requests and 4 decode instances:
- 200 requests × 4 instances = **800 calls** per bootstrap processing round
- Bootstrap processing happens frequently (after arrivals, batch completions, etc.)

## Phase 4 Optimization: Cache Decode Instance Capacity

### Proposed Solution

Add **event-level caching** for decode instance allocatable tokens:

```python
class SimulationEngine:
    def __init__(self, config: SimConfig):
        # ... existing init ...
        self._decode_capacity_cache = {}  # {instance_id: (timestamp, tokens)}
        self._cache_valid_until = -1.0

    def _invalidate_decode_capacity_cache(self):
        """Invalidate cache when decode state changes."""
        self._cache_valid_until = -1.0
        self._decode_capacity_cache.clear()

    def _decode_allocatable_tokens_cached(self, decode_instance: DecodeInstance) -> int:
        """Cached version of _decode_allocatable_tokens()."""
        # Cache valid only within current event processing (same timestamp)
        if self.current_time > self._cache_valid_until:
            self._invalidate_decode_capacity_cache()
            self._cache_valid_until = self.current_time

        instance_id = decode_instance.instance_id
        if instance_id in self._decode_capacity_cache:
            return self._decode_capacity_cache[instance_id]

        # Compute and cache
        tokens = self._decode_allocatable_tokens(decode_instance)
        self._decode_capacity_cache[instance_id] = tokens
        return tokens

    def _select_best_decode_instance(self):
        """Select decode instance with most allocatable tokens (cached)."""
        best = None
        best_tokens = -float("inf")
        for inst in self.instance_manager.decode_instances:
            if not inst.accepting_requests:
                continue
            tokens = self._decode_allocatable_tokens_cached(inst)  # Use cached version
            if tokens > best_tokens:
                best_tokens = tokens
                best = inst
        return best
```

### Cache Invalidation Points

Invalidate cache when decode instance state changes:
1. After decode batch completes (`_handle_decode_batch_complete`)
2. After request completes (`_handle_request_complete`)
3. After role switch completes
4. After any event that modifies `prealloc_reserved` or running batch

### Expected Impact

With 4 decode instances and 200-request bootstrap queue:
- **Before**: 11.1M calls to `_decode_allocatable_tokens()`
- **After**: ~100K calls (only when cache invalidated)
- **Reduction**: 99% fewer calls
- **Time saved**: ~34s out of 35.5s = **95% reduction in this function's time**
- **Overall speedup**: Additional **2-3x** on top of Phase 2/3

### Implementation Complexity

- **Difficulty**: Medium (requires careful cache invalidation)
- **Risk**: Low (cache is conservative - invalidated on any state change)
- **Testing**: Verify cached values match uncached values

## Simulation Queue Buildup Analysis

Separate issue identified: Bootstrap queue grows from 0 to 206 requests during simulation (t=250-450s).

### Queue Growth Timeline

```
  t=   0s: bootstrap=   0, prealloc_reserved=   1
  t=  50s: bootstrap=   0, prealloc_reserved= 147
  t= 100s: bootstrap=   0, prealloc_reserved=  33
  t= 150s: bootstrap=   0, prealloc_reserved= 125
  t= 200s: bootstrap=   0, prealloc_reserved= 176
  t= 250s: bootstrap=  29, prealloc_reserved= 307  ← Growth starts
  t= 300s: bootstrap=  58, prealloc_reserved= 288
  t= 350s: bootstrap=  66, prealloc_reserved= 241
  t= 400s: bootstrap= 128, prealloc_reserved= 241
  t= 450s: bootstrap= 206, prealloc_reserved= 285
  t= 490s: bootstrap= 119, prealloc_reserved= 232
```

### Root Cause

This is **not a performance bug** but a **capacity issue**:
- Arrival rate spikes at t=250s
- Decode instances are saturated (prealloc_reserved ≈ 280-300)
- New arrivals cannot be admitted → accumulate in bootstrap queue
- This is the **intended behavior** when system is overloaded

### Why Simulation is "Slow"

The simulation takes 18+ minutes wall-clock time because:
1. **High event count**: With queues of 200+ requests, each admission attempt processes many requests
2. **Frequent cache misses**: With Phase 2/3 optimizations, the bottleneck shifted to capacity calculations
3. **Large trace**: azure_code_1h has thousands of requests over 3600s simulation time

**Phase 4 caching will address issue #2** by reducing redundant capacity calculations by 99%.

## Optimization Priority

1. ✅ **Phase 2**: Data structure optimizations (O(n²) → O(n)) - **COMPLETE**
2. ✅ **Phase 3**: Monitoring caching (O(n) → O(1)) - **COMPLETE**
3. 🚀 **Phase 4**: Decode capacity caching (11M calls → 100K calls) - **RECOMMENDED NEXT**
4. ⚠️ **Phase 5**: LP solver optimization (if still slow after Phase 4)

## Files to Modify for Phase 4

1. `core/engine.py`:
   - Add cache fields to `__init__`
   - Add `_invalidate_decode_capacity_cache()` method
   - Add `_decode_allocatable_tokens_cached()` method
   - Replace `_decode_allocatable_tokens()` with cached version in `_select_best_decode_instance()`
   - Invalidate cache in `_handle_decode_batch_complete()`, `_handle_request_complete()`, etc.

## Benchmark Comparison

### Current State (Phase 2 + Phase 3)
- Wall-clock time: 77.8s (azure_code_8000.csv)
- `_decode_allocatable_tokens()`: 35.5s (45.6%)
- `_try_admit_from_bootstrap()`: 5.7s (7.3%)

### Phase 4 Attempt Results (FAILED - REVERTED)

**Implementation**: Event-level caching of decode instance capacity

**Results** (azure_code_8000.csv):
- Wall-clock time: **120.4s** (1.55x **SLOWER** than Phase 2+3)
- `_decode_allocatable_tokens()`: 2.9s (99% call reduction ✅)
- `_decode_allocatable_tokens_cached()`: 13.6s (78M calls ❌)
- `_try_admit_from_bootstrap()`: **39.0s** (6.8x slower than baseline!)

**Why Phase 4 Failed**:

1. **Cache check overhead exceeded savings**
   - Each cached call requires: timestamp check + dict lookup
   - With 78M cache calls, even O(1) operations add up
   - Cache check overhead: ~13.6s
   - Savings from reduced computation: ~32.6s
   - Net benefit: ~19s, but...

2. **Bootstrap admission became bottleneck**
   - `_try_admit_from_bootstrap()` went from 5.7s → 39s
   - Cache invalidation called too frequently
   - Python dict operations at this scale are expensive

3. **Cache hit rate was TOO HIGH**
   - 99.82% hit rate means cache checking on every call
   - Would be better with ~90% hit rate and fewer total calls
   - The problem isn't cache misses, it's cache existence overhead

**Lessons Learned**:

- **Profiling is essential**: The original analysis was correct about the number of calls, but didn't account for cache overhead
- **Sometimes simple is better**: Dictionary lookups at 78M scale are not "free"
- **Cache granularity matters**: Event-level caching may be too frequent
- **Alternative approach needed**: Rather than caching, need to reduce the number of calls themselves

### Current State: Phase 2 + Phase 3 + Phase 4v2 (COMPLETE)

**Phase 4v2 Results** (Incremental Capacity Tracking - SUCCESSFUL):
- Wall-clock time: **15.0s** (azure_code_8000.csv, integration test)
- Profiling time: **43.1s** (with profiling overhead)
- `_decode_allocatable_tokens()`: 98.1s → 13.9s (7x faster)
- `_try_admit_from_bootstrap()`: 117.0s → 28.2s (4x faster)

**Implementation**:
- Added `_prealloc_reserved_tokens` running sum counter to SimInstance base class
- Replaced O(n) loop (150 iterations × 11M calls) with O(1) subtraction
- Updated all 8 prealloc_reserved modification points to maintain counter
- Counter tracked in base class to support role switching

**Cumulative Speedup (Phases 2+3+4v2)**:
- **Baseline**: ~400s (estimated with no optimizations)
- **After Phase 2+3**: 77.8s (5.1x faster)
- **After Phase 4v2**: 15.0s (**26.7x faster overall**)
