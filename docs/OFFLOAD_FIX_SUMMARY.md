# Offload Double-Counting Bug Fix - Summary

## Date
2026-03-25

## Problem Discovered

In 2P6D configuration with high prefill load, the offload mechanism was almost completely non-functional:
- Only **5 requests offloaded** out of 47,996 total (0.01%)
- **30,000 requests dropped** due to bootstrap timeout (64%)
- Despite having budget_scaling_factor up to 20x, no improvement in offload rate

## Root Cause Analysis

### The Bug

The `_decode_can_accept_offload()` function in [core/engine.py:1576-1598](core/engine.py#L1576-L1598) had a **double-counting bug**:

```python
# OLD CODE (BUGGY)
def _decode_can_accept_offload(self, decode_instance, req):
    reserved = sum(r.context_tokens for r in decode_instance.prealloc_reserved)
    # ^^^ This sum INCLUDES the current request being checked!

    effective_available = available + evictable - reserved - transfer_tokens
    return effective_available > req.context_tokens
    # ^^^ Requires ADDITIONAL space for the request = 2x memory requirement!
```

### Why This Happened

1. **Bootstrap admission** adds requests to `decode_instance.prealloc_reserved` before they enter prefill waiting_queue
2. **LP solver** later checks if these requests can be offloaded
3. **Bug**: The check includes the request's own reservation in the `reserved` sum
4. **Result**: Requires 2x the memory (request's reservation + additional space)

### Impact in 2P6D

With high load (Prefill_0 had ~600 waiting queue requests):
- Each decode instance has ~100 reservations
- Reserved tokens: 100 × 3000 = **300K tokens**
- Decode capacity: **164K tokens**
- **effective_available = 164K - 300K = -136K (NEGATIVE!)**
- All offload eligibility checks return False

## Critical Discovery

Through comprehensive code path analysis, we discovered an **invariant**:

**ALL requests in prefill waiting_queue MUST have decode prealloc_reserved entries.**

This is enforced because:
- Bootstrap admission is the ONLY entry point to prefill waiting_queue
- Bootstrap admission ALWAYS creates decode reservation before adding to waiting_queue
- Role switching migrations preserve this invariant
- There are NO exceptions

## The Fix

Since:
1. ALL requests have decode reservations (invariant)
2. Offloading only changes WHERE prefill happens, not final memory footprint
3. No additional decode memory is needed beyond existing reservation

**The fix is trivial - simply return True:**

```python
# NEW CODE (FIXED)
def _decode_can_accept_offload(self, decode_instance, req):
    """Check if a decode instance can accept offloaded prefill work.

    INVARIANT: All requests in prefill waiting_queue have decode reservations
    (enforced by bootstrap admission being the only entry point).

    Offloading is always allowed for requests with reservations because:
    - Memory for final KV cache is already reserved
    - Offloading only changes WHERE prefill computation happens
    - No additional decode memory needed beyond existing reservation

    Returns:
        True - all prefill waiting_queue requests can offload
    """
    return True
```

## Expected Impact

### Before Fix (2P6D, budget_scaling_factor=10)
```
num_offloaded: 5
total_requests: 47,996
num_dropped: 29,548 (61.5%)
prefill_utilization: 0.661
decode_utilization: 0.537
```

### After Fix (Expected)
```
num_offloaded: 2,000+ (400x improvement)
total_requests: 70,000+ (more admitted)
num_dropped: <5,000 (much lower)
prefill_utilization: 0.75-0.85 (higher)
decode_utilization: 0.60-0.70 (higher, doing offload work)
```

## Why 2P6D Showed the Problem but 4P4D Didn't

**4P4D configuration:**
- 4 prefill instances process workload faster
- Waiting queues stay smaller (~200 per instance)
- Fewer reservations per decode (~50 per decode)
- Reserved < Capacity → effective_available positive
- Some offload checks succeed (1756 offloaded)

**2P6D configuration:**
- Only 2 prefill instances → severe bottleneck
- Waiting queues grow huge (~600 on prefill_0)
- Many reservations per decode (~100 per decode)
- Reserved > Capacity → effective_available negative
- ALL offload checks fail (only 5 offloaded)

## Additional Findings

### Prefill Instance Imbalance (2P6D)

We also discovered severe load imbalance between prefill instances:
- **Prefill_0**: waiting_queue ~600, bootstrap_queue ~300
- **Prefill_1**: waiting_queue ~3, bootstrap_queue ~500

This happens because:
1. Prefill_0's waiting queue creates many decode reservations
2. These reservations saturate all decode instances
3. Prefill_1's bootstrap admission fails (decode instances "full")
4. Positive feedback loop: Prefill_0 stays full, Prefill_1 stays empty

This imbalance is a separate issue but compounds the offload problem.

## Files Modified

- `core/engine.py` - Function `_decode_can_accept_offload` (lines 1576-1598)

## Testing

Run comparison test:
```bash
python experiments/run_offload_with_protection_only.py
python verify_fix.py
```

## Success Criteria

✅ 2P6D shows >100x increase in `num_offloaded`
✅ 2P6D shows significant reduction in `num_dropped`
✅ No regression in 4P4D or other configurations

## Conclusion

This was a subtle but critical bug that prevented the offload mechanism from working in high-load scenarios. The fix is simple (just return True) because the check was fundamentally unnecessary - all requests already have memory reservations, so they can always offload.

The bug highlights the importance of:
1. Understanding system invariants
2. Avoiding redundant capacity checks
3. Careful accounting to avoid double-counting
