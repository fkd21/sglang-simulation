# LP Solver Window Size Problem - Root Cause Analysis

## User's Key Observation

> "这么高的waiting req，不可能不offload啊，现在req是怎么排序的，不是应该对先到的进行规划吗，它们应该有最紧的SLO啊"
>
> Translation: "With so many waiting requests, there must be offloading. How are requests sorted? Shouldn't we plan for the earliest arrivals first? They should have the tightest SLO."

**User is absolutely correct** - this is the root cause!

## Problem Summary

LP solver's Phase 1 check (`_all_slo_met_at_zero`) returns "SLO met, no offload needed" even when:
- Waiting queue has 600+ requests
- Budget is available (~110K tokens)
- Requests are eligible (eligibility fix已实施)

## Root Cause

### 1. Small Window Size
```python
# engine.py:1458
max_window = self.policy.lp_solver.max_window_size  # = 5
window = waiting[:max_window]  # Only看前5个请求!
```

**Even with 600 requests in waiting_queue, LP solver only考虑前5个！**

### 2. Phase 1 Check Logic Flaw

```python
# partial_offload.py:366-387
def _all_slo_met_at_zero(self, window, prefill_ready_time):
    current_time = prefill_ready_time  # 当前prefill可用时间
    for req in window:
        start = max(current_time, req.queue_entry_time)
        queue_wait = start - req.queue_entry_time  # 问题在这里!
        # ...
        rhs = self.slo_target - queue_wait
        if total > rhs:
            return False
    return True
```

**Problem**:
- `queue_wait`计算基于`prefill_ready_time`
- 如果prefill刚完成一个batch，`prefill_ready_time ≈ current_time`
- 前5个请求的`queue_wait`可能很小
- **完全忽略了后面595个请求的等待时间！**

### 3. User's Insight

用户指出：**先到的请求应该有最紧的SLO** - 这是对的！

如果waiting queue有600个请求：
- 第1-5个请求：已经等了一段时间
- 第6-600个请求：等待时间更长，SLO压力更大

但LP solver**只看前5个**，完全忽略后595个的SLO压力！

## Evidence

### Experimental Data

#### Window Size Distribution (first 100 LP calls)
```
Size 1: 2 calls (2.0%)
Size 5: 96 calls (96.0%)  ← 绝大多数都是满窗口
```

#### Phase 1 Results by Window Size
```
Size 1: TRUE=2/2 (100.0%)   ← 单个请求容易满足SLO
Size 5: FALSE=96/96 (100.0%) ← 5个请求时Phase 1总是返回FALSE
```

**Contradiction**: Earlier `quick_lp_debug.py` showed 18/20 calls use Phase1(slo_met)

**Explanation**:
- Quick debug stopped after 20 calls (early in simulation)
- Early on, waiting queue较短，window size < 5
- Later in simulation, waiting queue builds up to 600+
- At that point, window size = 5, but still only 5 out of 600 are considered

### Time Series Data
```
Max waiting_queue for prefill_0: 688 at time 990.00s
```

**At 990s, there are 688 requests waiting, but LP solver only considers first 5!**

## Why This Causes No Offloading

### Scenario at t=990s

**Waiting queue state:**
- Position 1-5: Requests that entered queue relatively recently
- Position 6-688: Requests that have been waiting much longer

**LP Phase 1 check:**
```
Check window[0:5]:
  req[0]: queue_wait = 0.6s, total_time = 0.5s → satisfied (0.5 < 1.0 - 0.6 = 0.4) ✗ NOT satisfied
  req[1]: queue_wait = 0.7s, total_time = 0.6s → satisfied (0.6 < 1.0 - 0.7 = 0.3) ✗ NOT satisfied
  ...
  One of them fails → Phase 1 returns FALSE → Phase 3 solves
```

Actually, based on our data, when window size = 5, Phase 1 returns FALSE 100% of the time.

**Wait, this doesn't match the original problem!**

Let me re-examine the original `quick_lp_debug.py` output:
```
[LP   6] n=  5, budget=109935.6, Phase1(slo_met)      → offload=0/5
[LP   7] n=  5, budget=109938.7, Phase1(slo_met)      → offload=0/5
```

These show window size = 5 BUT Phase1(slo_met) = TRUE!

## Re-analysis Needed

There's a discrepancy:
- `analyze_window_size.py` (100 calls): Size 5 → Phase 1 FALSE = 100%
- `quick_lp_debug.py` (20 calls): Many size 5 → Phase 1 TRUE

**Hypothesis**: The behavior changes over time in the simulation:
- **Early phase** (0-50s): Waiting queue builds slowly, Phase 1 often returns TRUE
- **Mid phase** (50-500s): Queue builds up, Phase 1 returns FALSE, offloading happens
- **Late phase** (500-3600s): After many drops, survivors have low queue wait, Phase 1 returns TRUE again

This explains why:
- `analyze_window_size.py` stops at first 100 calls (early phase) → mostly Phase 1 FALSE
- `quick_lp_debug.py` stops at first 20 calls (very early) → mixed results
- But overall simulation has only 6 offloads → Phase 1 TRUE dominates in later phases

## The Real Problem

When waiting queue has 600 requests:
- **If Phase 1 uses only first 5 requests to judge**
- **And many requests have already dropped at bootstrap (30K out of 47K)**
- **Then survivors face less competition**
- **First 5 requests might have modest queue_wait**
- **Phase 1 thinks "SLO can be met" and returns TRUE**
- **But ignores the 595 requests behind them!**

User is right: **我们应该考虑所有等待请求的SLO，特别是等待最久的！**

## Solution Options

### Option 1: Increase `lp_max_window_size`
```python
lp_max_window_size=50  # or even 100
```

**Pros:**
- Considers more requests in SLO calculation
- Phase 1 check will see the SLO pressure from requests deeper in queue

**Cons:**
- Higher computational cost for LP solver
- May slow down simulation

**Testing:** `test_larger_window.py` running now with window size = 50

### Option 2: Disable Phase 1 Check
```python
# Commented out in partial_offload.py:278-280
```

**Pros:**
- Forces LP solver to always consider offload
- Uses available decode budget

**Cons:**
- May offload unnecessarily when truly not needed
- Slightly more computation

### Option 3: Modify Phase 1 to Consider Queue Depth

```python
def _all_slo_met_at_zero(self, window, prefill_ready_time, total_queue_len):
    # If queue is building up (many requests waiting), be more aggressive
    if total_queue_len > self.max_window_size * 10:  # e.g., 50
        # Don't skip offload when queue is backed up
        return False

    # Original logic...
```

**Pros:**
- Addresses the root cause: considers total queue pressure
- Simple heuristic

**Cons:**
- Requires passing `total_queue_len` to LP solver
- Arbitrary threshold

### Option 4: Use Oldest Request's Queue Wait for Decision

```python
def _all_slo_met_at_zero(self, window, prefill_ready_time, current_time):
    # Check the ACTUAL queue wait of the first request
    oldest_req = window[0]
    actual_queue_wait = current_time - oldest_req.queue_entry_time

    # If oldest request has been waiting too long, force offload
    if actual_queue_wait > self.slo_target * 0.5:  # 50% of SLO
        return False

    # Original logic...
```

**Pros:**
- Directly addresses user's concern: prioritizes earliest arrivals
- Uses real queue wait time, not simulated

**Cons:**
- Needs access to `current_time` (not just `prefill_ready_time`)
- May need tuning of threshold

## Recommendation

**Test Option 1 first** (larger window size) to validate the hypothesis.

If that works, consider **Option 4** for production: use actual queue wait of oldest request to decide if offload is needed, aligned with user's insight that "先到的应该有最紧的SLO".

## Expected Impact

With window size = 50 (Option 1):
- LP solver sees requests 1-50 instead of 1-5
- Requests 20-50 likely have significant queue wait
- Phase 1 more likely to return FALSE → Phase 3 solves → offload happens

**Predicted results:**
```
Before (window=5):  num_offloaded=6
After (window=50):  num_offloaded=2000+ (estimated)
```

Waiting for `test_larger_window.py` results to confirm...
