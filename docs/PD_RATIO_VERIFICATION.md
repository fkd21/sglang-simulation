# P/D Ratio Verification After Role Switching

## Question
Is the offload with protection mechanism using the correct P/D ratio after role switches?

## Answer: YES ✓

The P/D ratio is **always correct** because the code uses dynamic instance counts from the instance manager, which is updated immediately during role switches.

## Code Flow Analysis

### 1. Budget Calculation During Offload

When calculating the decode offload budget, the code uses:

```python
# core/engine.py:1502-1510
decode_budget = calculate_decode_offload_budget(
    decode_bs=decode_bs,
    decode_token_sum=decode_token_sum,
    tpot_sla=self.config.tpot_sla,
    prefill_max_tokens=self.config.max_prefill_tokens,
    num_decode_instances=len(self.instance_manager.decode_instances),  # ← Dynamic
    num_prefill_instances=len(self.instance_manager.prefill_instances),  # ← Dynamic
    budget_scaling_factor=self.config.budget_scaling_factor
)
```

**Key observation**: The code uses `len(self.instance_manager.decode_instances)` and `len(self.instance_manager.prefill_instances)`, which reflect the **current state** of the system.

### 2. Instance Manager Lists During Role Switch

During a role switch, the instance manager lists are **immediately updated**:

```python
# core/engine.py:2161-2170
# Move instance between instance manager lists
if old_role == InstanceType.PREFILL and new_role == InstanceType.DECODE:
    # Remove from prefill list
    self.instance_manager.prefill_instances.remove(instance)
    # Add to decode list
    self.instance_manager.decode_instances.append(instance)
elif old_role == InstanceType.DECODE and new_role == InstanceType.PREFILL:
    # Remove from decode list
    self.instance_manager.decode_instances.remove(instance)
    # Add to prefill list
    self.instance_manager.prefill_instances.append(instance)
```

**Key observation**: The lists are updated **before** any subsequent budget calculations.

### 3. Budget Formula

The budget calculation uses the D/P ratio:

```python
# mechanisms/partial_offload.py:110
budget = per_iter_budget * n_iter * (num_decode_instances/num_prefill_instances) * budget_scaling_factor
```

This means:
- **More decode instances** → **higher budget** (more capacity to accept offload)
- **More prefill instances** → **lower budget** (less pressure to offload)

## Event Processing Timeline

The simulation follows an event-driven architecture. Here's the order of operations:

```
Event Queue (priority queue ordered by timestamp)
│
├─ Event: ROLE_SWITCH @ t=100.0
│  └─ Handler: _handle_role_switch()
│     ├─ Updates instance_manager.prefill_instances  [4P → 3P]
│     ├─ Updates instance_manager.decode_instances   [4D → 5D]
│     └─ Completes switch atomically
│
├─ Event: PREFILL_BATCH_COMPLETE @ t=100.5
│  └─ Handler: _handle_prefill_batch_complete()
│     ├─ Calls _try_schedule_prefill()
│     │  └─ Calls _apply_dynamic_lp_betas()
│     │     ├─ Queries: len(instance_manager.decode_instances) → 5
│     │     ├─ Queries: len(instance_manager.prefill_instances) → 3
│     │     ├─ Calculates budget with D/P = 5/3 = 1.67
│     │     └─ Solves LP with updated budget
│     └─ Schedules next batch with new betas
│
└─ ... (more events)
```

**Key observations:**
1. Role switch completes **before** subsequent batch scheduling events
2. Budget calculation happens **inside** batch scheduling
3. Lists are already updated when budget is calculated
4. No race conditions (single-threaded event loop)

## Verification Test Results

```
Configuration    | P  | D  | D/P  | Budget
------------------------------------------------------------
Initial (4P,4D)  | 4  | 4  | 1.00 | 2544.48
After P→D (3P,5D)| 3  | 5  | 1.67 | 4240.79
After D→P (4P,4D)| 4  | 4  | 1.00 | 2544.48
```

The test demonstrates:
1. ✓ Lists are correctly updated during role switch
2. ✓ Budget calculation uses current list lengths
3. ✓ P/D ratio changes immediately affect budget
4. ✓ Reversing the switch restores the original ratio

## Why This Design Works

The design is robust because:

1. **Single source of truth**: `instance_manager.prefill_instances` and `decode_instances` are the only lists used everywhere
2. **Atomic updates**: List modifications happen in a single location during role switch
3. **No cached values**: Budget calculation always queries current list lengths via `len()`
4. **Temporal correctness**: Lists are updated **before** the instance accepts new work

## Potential Issues (None Found)

We checked for common bugs:
- ❌ Using config values (fixed P/D) instead of dynamic counts → **Not present**
- ❌ Caching old P/D ratios → **Not present**
- ❌ Race conditions in list updates → **Not applicable (single-threaded simulation)**
- ❌ Stale list lengths → **Not possible (Python list.len() is always current)**

## Conclusion

✅ **The offload with protection mechanism correctly uses the updated P/D ratio after role switches.**

The implementation is sound because:
- Role switches immediately update the instance manager lists
- Budget calculations dynamically query current list lengths
- There are no cached or stale values that could cause inconsistency

## Call Stack Analysis

Here's the complete call stack when budget is calculated:

```
1. Event loop processes PREFILL_BATCH_COMPLETE event
   ↓
2. _handle_prefill_batch_complete()
   ↓
3. _try_schedule_prefill(instance)
   ├─ Line 1401: if self.policy.dynamic_lp_enabled and instance.waiting_queue:
   ↓
4. _apply_dynamic_lp_betas(instance)
   ├─ Line 1489-1497: Query decode instance state
   ├─ Line 1500-1510: Calculate budget with protection
   │  ├─ Line 1507: num_decode_instances=len(self.instance_manager.decode_instances)
   │  └─ Line 1508: num_prefill_instances=len(self.instance_manager.prefill_instances)
   ↓
5. calculate_decode_offload_budget()
   ├─ Line 110: budget = per_iter_budget * n_iter * (D/P) * scaling
   └─ Returns budget value
   ↓
6. policy.solve_dynamic_betas_with_budget()
   └─ Returns optimal betas respecting budget
```

**Critical insight**: Every time budget is calculated, it **directly queries** the instance manager lists via `len()`, which always reflects the current state after any role switches.

## Files Checked

1. [core/engine.py:1402](core/engine.py#L1402) - LP betas called during batch scheduling
2. [core/engine.py:1507-1508](core/engine.py#L1507-L1508) - Budget calculation with dynamic counts
3. [core/engine.py:2163-2170](core/engine.py#L2163-L2170) - List updates during role switch
4. [mechanisms/partial_offload.py:110](mechanisms/partial_offload.py#L110) - Budget formula with D/P ratio
5. [instances/instance_manager.py:22-29](instances/instance_manager.py#L22-L29) - Instance list initialization

## Test Scripts

1. `verify_pd_ratio_after_switch.py` - Unit test of list updates and budget calculation
2. `test_offload_pd_ratio_integration.py` - Integration test with realistic scenario
