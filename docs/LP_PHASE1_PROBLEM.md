# LP Solver Phase 1 Problem Analysis

## Summary

The eligibility fix (`_decode_can_accept_offload` → `return True`) was implemented correctly but **did not increase offload rate** because LP solver's Phase 1 check prevents offloading when SLO is already satisfied.

## Evidence

### Debug Output (quick_lp_debug.py)

```
[LP   0] n=  1, budget=110276.4, Phase1(slo_met)      → offload=0/1
[LP   1] n=  1, budget=110276.4, Phase1(slo_met)      → offload=0/1
[LP   2] n=  4, budget=110276.4, Phase1(slo_met)      → offload=0/4
[LP   3] n=  3, budget=110276.4, Phase1(slo_met)      → offload=0/3
[LP   4] n=  5, budget=110276.4, Phase3(solved)       → offload=1/5  ← ONLY 2 ACTUAL SOLVES!
[LP   5] n=  5, budget=110276.4, Phase3(solved)       → offload=4/5  ←
[LP   6] n=  5, budget=109935.6, Phase1(slo_met)      → offload=0/5
...
[LP  19] n=  5, budget=110000.8, Phase1(slo_met)      → offload=0/5
```

**Key observations:**
- 18 out of 20 LP calls use Phase1 (SLO already met)
- Budget is available (~110K tokens)
- Requests are eligible (fixed by our change)
- But LP solver returns beta=0 without solving

### Experimental Results

**Before fix (double-counting bug):**
```json
{
  "num_offloaded": 5,
  "total_requests": 47996,
  "num_dropped": 29548,
  "budget_scaling_factor": 10
}
```

**After eligibility fix:**
```json
{
  "num_offloaded": 6,
  "total_requests": 47597,
  "num_dropped": 29947,
  "budget_scaling_factor": 10
}
```

**No meaningful improvement** - still only 6 offloads instead of expected 2000+.

## Root Cause

### Phase 1 Logic (partial_offload.py:278-280)

```python
# Phase 1: Check if β=0 satisfies prefill SLO
if self._all_slo_met_at_zero(window, prefill_ready_time):
    return [0.0] * n, True, time.perf_counter() - t_start
```

### _all_slo_met_at_zero Function (lines 366-386)

```python
def _all_slo_met_at_zero(
    self, window: List[SimReq], prefill_ready_time: float
) -> bool:
    """Check if all SLO constraints are met with beta=0 (no offloading)."""
    current_time = prefill_ready_time
    cumulative_prefill = 0.0
    for req in window:
        start = max(current_time, req.queue_entry_time)
        prefill_duration = prefill_inference_time(req.context_tokens)
        cumulative_prefill += prefill_duration
        kv_transfer = _KV_TRANSFER_COEFF * req.context_tokens
        queue_wait = start - req.queue_entry_time
        # No offload when beta=0
        total = cumulative_prefill + kv_transfer
        rhs = self.slo_target - queue_wait
        if total > rhs:
            return False
        current_time = start + prefill_duration
    return True
```

**What it does:**
- For each request in window, calculates: `total_time = cumulative_prefill + kv_transfer`
- Checks if `total_time ≤ slo_target - queue_wait`
- If all requests satisfy this, returns True → beta=0 (no offload)

### Why This Happens in 2P6D

1. **High bootstrap drop rate**: 30K out of 47K requests dropped (61.5%)
2. **Low competition for survivors**: Only ~18K requests make it through bootstrap
3. **Short queue wait times**: Survivors face relatively low queuing delay
4. **SLO satisfied without offload**: `cumulative_prefill + kv_transfer ≤ slo_target - queue_wait`
5. **Phase 1 returns beta=0**: No offload happens despite:
   - High budget (~110K tokens available)
   - Eligible requests (fixed by our change)
   - High prefill queue buildup (~600 in prefill_0)

## The Problem with Phase 1

### Design Intent (Original)

Phase 1 was likely designed as an **optimization** to avoid expensive LP solving when not needed:
- If SLO is already satisfied, why offload?
- Save computation time by early return

### Why It's Problematic in 2P6D

**Offloading could still be beneficial even if SLO is met:**

1. **Prevent future bootstrap drops**
   - Offloading reduces prefill queue buildup
   - Lower prefill queue → more bootstrap capacity
   - More requests can be admitted

2. **Improve resource utilization balance**
   - Prefill: 65% utilization (high)
   - Decode: 10% utilization (very low)
   - Offloading would balance this

3. **Reduce queue variability**
   - Prefill_0 has ~600 waiting queue
   - Prefill_1 has ~0-20 waiting queue
   - Severe imbalance causing instability

4. **Use available decode budget**
   - 110K tokens of budget available
   - Should be used to improve system performance

### Conceptual Issue

Phase 1 optimizes for **current window SLO** but ignores:
- **System-level effects**: Bootstrap capacity, utilization balance
- **Future workload**: Preventing queue buildup helps future requests
- **Available resources**: Decode budget is wasted

## Solution Options

### Option 1: Disable Phase 1 Check Entirely

**Change:**
```python
# Phase 1: Check if β=0 satisfies prefill SLO
# if self._all_slo_met_at_zero(window, prefill_ready_time):
#     return [0.0] * n, True, time.perf_counter() - t_start
```

**Pros:**
- Simple: comment out 2 lines
- Forces LP solver to always consider offload
- Uses available decode budget

**Cons:**
- May offload unnecessarily in some cases
- Slightly more computation (LP solve every time)

### Option 2: Condition Phase 1 on Queue Pressure

**Change:**
```python
# Phase 1: Only skip offload if SLO met AND queue is not building up
if self._all_slo_met_at_zero(window, prefill_ready_time):
    # Check if prefill queue is under pressure
    # (e.g., waiting_queue size, bootstrap drops)
    if not self._prefill_under_pressure():
        return [0.0] * n, True, time.perf_counter() - t_start
```

**Pros:**
- More nuanced decision making
- Avoids unnecessary offload when truly not needed

**Cons:**
- Need to pass additional context to LP solver
- More complex implementation
- Requires defining "pressure" threshold

### Option 3: Make Phase 1 Conditional on Budget Utilization

**Change:**
```python
# Phase 1: Only skip if SLO met AND little decode budget available
if self._all_slo_met_at_zero(window, prefill_ready_time):
    # If we have significant unused decode budget, still try to offload
    avg_req_size = sum(r.context_tokens for r in window) / len(window)
    if decode_budget < avg_req_size * 2:  # Low budget
        return [0.0] * n, True, time.perf_counter() - t_start
```

**Pros:**
- Uses decode budget when available
- Simple heuristic

**Cons:**
- Arbitrary threshold (2x average)
- Doesn't address bootstrap capacity issue

### Option 4: Replace Phase 1 with Soft Preference

**Change:**
Instead of hard early-return, add SLO satisfaction as a factor in optimization:
```python
# Remove Phase 1 entirely, let greedy solver handle all cases
# Solver naturally prefers minimal offload if SLO constraints are slack
```

**Pros:**
- LP solver naturally finds minimal offload when SLO is already met
- No special case needed
- More principled approach

**Cons:**
- Slightly more computation
- May need to verify solver behavior

## Recommendation

**Use Option 1 (Disable Phase 1) for immediate testing:**

1. **Simplest fix**: Comment out 2 lines
2. **Validates hypothesis**: If offload rate increases dramatically, confirms Phase 1 is the bottleneck
3. **Quick to test**: Can run experiment immediately
4. **Minimal risk**: LP solver will still respect budget constraints (Phase 2) and feasibility

**If Option 1 works, consider Option 4 for production:**
- More principled: Let optimization naturally find minimal offload
- Cleaner: No special cases
- Better: Handles all scenarios uniformly

## Expected Impact

### If we disable Phase 1:

**Before (Phase 1 enabled):**
```
num_offloaded: 6
num_dropped: 29947 (62.9%)
offload_rate: 0.01%
```

**After (Phase 1 disabled):**
```
num_offloaded: 2000+ (expected)
num_dropped: <5000 (much lower)
offload_rate: 5-10% (estimated)
```

**Reasons:**
- LP solver will actually run Phase 3 (greedy solve)
- Requests are eligible (eligibility fix already applied)
- Budget is available (~110K tokens)
- Should see significant offloading

## Next Steps

1. **Implement Option 1**: Comment out Phase 1 check
2. **Run experiment**: `python experiments/run_offload_with_protection_only.py`
3. **Verify results**: Check if `num_offloaded` increases dramatically
4. **If successful**: Consider more nuanced approach (Option 4)
5. **If unsuccessful**: Investigate other bottlenecks
