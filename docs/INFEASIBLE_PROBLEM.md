# LP Solver Infeasibility Problem - Root Cause Found!

## Summary

**真正的问题找到了！**

LP solver进入Phase 3求解，但返回`feasible=False`（infeasible），导致返回全0的beta（无offload）。

## Evidence

```
[LP   4] n= 5, budget=110276.4, Phase3(feasible)          → offload=1/5
[LP   5] n= 5, budget=110276.4, Phase3(feasible)          → offload=4/5
[LP   6] n= 5, budget=109935.6, Phase3(INFEASIBLE)        → offload=0/5  ← 从这里开始全是INFEASIBLE!
[LP   7] n= 5, budget=109938.7, Phase3(INFEASIBLE)        → offload=0/5
...
[LP  19] n= 5, budget=110000.8, Phase3(INFEASIBLE)        → offload=0/5
```

**从LP call 6开始，所有Phase 3求解都返回infeasible！**

## Root Cause: Infeasibility Fallback Logic

### Code Location: `partial_offload.py:360-362`

```python
# Fallback: prioritize decode protection
if not feasible:
    return [0.0] * n, False, solve_time
```

**当LP solver无法找到可行解时，返回全0的beta（不offload）。**

### Why Infeasible?

Phase 3求解过程（lines 308-352）：

```python
for i in range(n):
    # Compute deficit for constraint i
    C_i = (i + 1) * b + a * cum_c_list[i] + k * c[i] + D
    R_i = self.slo_target - q[i]

    deficit = (C_i - R_i) - lhs_reduction

    if deficit <= 0.0:
        continue  # Constraint satisfied

    # Try to reduce deficit by increasing betas
    for j in range(i + 1):
        # ... increase betas with budget constraint ...

    if deficit > 1e-9:
        feasible = False  # Line 352: 无法满足constraint i
```

**Infeasible发生在：**
1. 某个请求的SLO constraint无法满足
2. 即使将所有beta设为1.0（全部offload）
3. 或者budget用完了，但deficit还是>0

### Why This Happens in 2P6D

从之前的Phase 1 debug可以看到：

```
Phase 1 Debug #4 - prefill_0
Req  entry    wait     tokens start    q_wait   cum_pf   kv       total    rhs      OK
-----------------------------------------------------------------------------------------------
0    0.573    1.098    3117   1.671    1.098    0.2311   0.0058   0.2369   -0.098   N
1    0.626    1.046    993    1.902    1.277    0.3147   0.0019   0.3166   -0.277   N
2    0.651    1.020    2373   1.986    1.335    0.4941   0.0044   0.4986   -0.335   N
3    0.757    0.914    1196   2.165    1.408    0.5918   0.0022   0.5941   -0.408   N
4    0.876    0.796    2924   2.263    1.388    0.8095   0.0055   0.8150   -0.388   N
```

**第1个请求的RHS已经是负数（-0.098）！**

这意味着：
- SLO target = 1.0s
- 请求已经等待了1.098s
- RHS = 1.0 - 1.098 = -0.098s

**请求已经违反SLO了！**

即使offload也无法让这个请求满足SLO，因为它已经等太久了。

### LP Solver's Logic

LP solver看到请求已经违反SLO，认为：
- 即使offload也无法满足SLO
- 返回`feasible=False`
- Fallback逻辑返回beta=0（不offload）

**这个设计的问题：**

"宁可不offload，也不违反SLO" —— 但实际上SLO已经违反了！

不offload的后果：
1. Prefill queue继续积累
2. 更多请求等待更久
3. 更多请求在bootstrap被drop
4. 雪崩效应

## The Real Problem

### Issue 1: SLO Already Violated

当请求等待时间超过SLO target时：
- `q[i] = current_time - req.queue_entry_time > slo_target`
- `R_i = slo_target - q[i] < 0`
- 任何offload都无法让`C_i < R_i`
- LP solver认为infeasible

### Issue 2: Fallback Logic Too Conservative

```python
if not feasible:
    return [0.0] * n, False, solve_time
```

**这个fallback太保守了！**

即使无法满足所有SLO，offload仍然是有益的：
1. 减少prefill queue压力
2. 使用空闲的decode资源
3. 改善后续请求的SLO
4. 避免bootstrap drops

### Issue 3: Doesn't Account for Already-Violated SLOs

LP solver应该区分：
1. **可以满足的SLO** → 优先满足
2. **已经违反的SLO** → 尽量减少违反程度，但不应该因为它而放弃所有offload

## Solution Options

### Option 1: Remove Infeasibility Fallback

```python
# Don't fallback to zero when infeasible
# if not feasible:
#     return [0.0] * n, False, solve_time

# Return whatever betas were computed, even if not all constraints satisfied
return betas, feasible, solve_time
```

**Pros:**
- Uses available decode budget
- Helps reduce queue pressure
- Pragmatic: some offload is better than no offload

**Cons:**
- May violate SLO constraints
- But SLOs are already violated anyway!

### Option 2: Relax Infeasible Constraints

```python
if deficit > 1e-9:
    # Don't mark as infeasible immediately
    # Try best effort with available budget
    pass  # Continue to next constraint
```

**Pros:**
- Best-effort approach
- Doesn't give up when one constraint fails

**Cons:**
- May need to adjust solver logic

### Option 3: Adjust SLO for Already-Violated Requests

```python
# If request has already waited > slo_target, use extended SLO
R_i = max(self.slo_target, q[i] + 0.5) - q[i]  # Allow 0.5s extension
```

**Pros:**
- Pragmatic: acknowledges reality
- Allows solver to make progress

**Cons:**
- Arbitrary extension amount
- Philosophically questionable

### Option 4: Skip Infeasible Constraints

```python
for i in range(n):
    R_i = self.slo_target - q[i]
    if R_i < 0:
        # SLO already violated, skip this constraint
        continue

    # ... solve for this constraint ...
```

**Pros:**
- Focuses on requests that can still meet SLO
- Doesn't let already-violated requests block all offloading

**Cons:**
- Ignores some requests entirely
- May be unfair

## Recommendation

**Use Option 1 (Remove Infeasibility Fallback) as immediate fix:**

The current logic "return [0.0] if infeasible" is causing all offloading to stop when:
- Queue builds up
- Requests wait > SLO target
- LP solver can't satisfy all constraints

**Better approach:** Return whatever betas were computed, even if not all constraints are satisfied.

**Rationale:**
- Some offload is better than no offload
- Helps prevent queue avalanche
- Uses available decode resources
- SLOs are already violated anyway, so returning beta=0 doesn't help

## Expected Impact

### Before (with fallback):
```
LP  6-19: Phase3(INFEASIBLE) → offload=0/5
Total offloads: ~6 out of 47K requests
```

### After (without fallback):
```
LP  6-19: Phase3(partial solution) → offload=1-5/5
Total offloads: 1000-5000+ (estimated)
```

## Next Step

Test Option 1: Comment out the infeasibility fallback and see if offload rate increases significantly.
