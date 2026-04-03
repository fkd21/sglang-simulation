# Fix: Double Counting in _decode_can_accept_offload

## Problem

The `_decode_can_accept_offload` function has a double-counting bug that prevents offloading for requests that already have decode memory reservations.

### Current Implementation (Buggy)

```python
def _decode_can_accept_offload(self, decode_instance: DecodeInstance, req: SimReq) -> bool:
    available = decode_instance.token_to_kv_pool.available_size()
    evictable = decode_instance.tree_cache.evictable_size

    # BUG: This includes the current request's reservation!
    reserved = sum(
        r.context_tokens for r in decode_instance.prealloc_reserved
    )
    transfer_tokens = sum(
        r.context_tokens for r in decode_instance.transfer_queue
    )

    effective_available = available + evictable - reserved - transfer_tokens

    # BUG: This checks if we have EXTRA space for req.context_tokens
    # But req is already counted in 'reserved'!
    return effective_available > req.context_tokens
```

### Why This Is Wrong

When a request is in `decode_instance.prealloc_reserved`:
1. Its memory is already accounted for in the decode instance
2. The `reserved` sum includes this request's `context_tokens`
3. Then we check if `effective_available > req.context_tokens`
4. This effectively requires **2x the space** for the request!

Example with numbers:
- Total capacity: 164K tokens
- Reserved (including current request): 252K tokens (100 requests × 2.5K each)
- Available: 164K - 252K = **-88K (negative!)**
- Check: -88K > 2.5K? **False**
- Result: Cannot offload, even though memory is already reserved!

### Conceptual Issue

Offloading a request that **already has a decode reservation** does NOT require additional decode memory because:
- Offload changes WHERE prefill computation happens (prefill vs decode instance)
- It does NOT change the final KV cache footprint
- Memory for `context_tokens + output_ids` is already reserved

## Solution

### Option 1: Exclude Self from Reservation Count (Conservative)

```python
def _decode_can_accept_offload(self, decode_instance: DecodeInstance, req: SimReq) -> bool:
    available = decode_instance.token_to_kv_pool.available_size()
    evictable = decode_instance.tree_cache.evictable_size

    # Exclude current request from reserved count to avoid double-counting
    reserved = sum(
        r.context_tokens for r in decode_instance.prealloc_reserved
        if r.rid != req.rid  # <-- KEY FIX
    )
    transfer_tokens = sum(
        r.context_tokens for r in decode_instance.transfer_queue
    )

    effective_available = available + evictable - reserved - transfer_tokens
    return effective_available > req.context_tokens
```

### Option 2: Always Allow Offload for Reserved Requests (Simpler, More Correct)

```python
def _decode_can_accept_offload(self, decode_instance: DecodeInstance, req: SimReq) -> bool:
    # If request already has a reservation, it can ALWAYS offload
    # because its decode memory is already allocated
    if req in decode_instance.prealloc_reserved:
        return True

    # For requests without reservations, check if decode has capacity
    available = decode_instance.token_to_kv_pool.available_size()
    evictable = decode_instance.tree_cache.evictable_size

    reserved = sum(
        r.context_tokens for r in decode_instance.prealloc_reserved
    )
    transfer_tokens = sum(
        r.context_tokens for r in decode_instance.transfer_queue
    )

    effective_available = available + evictable - reserved - transfer_tokens
    return effective_available > req.context_tokens
```

## Recommendation

**Use Option 2** because:
1. **Conceptually correct**: Requests with reservations should always be eligible
2. **Simpler**: Early return avoids complex calculation
3. **More efficient**: O(1) check instead of O(n) sum
4. **Clear intent**: Code explicitly states the logic

## Expected Impact

After fixing:
- Requests in prefill waiting queues will be eligible for offload
- 2P6D configuration should see significant offloading
- LP solver will be able to optimize for requests that already have decode reservations
- Offload mechanism will work as designed

## Additional Note: Reservation Accounting Inconsistency

There's also an inconsistency in how reservations are calculated:

Bootstrap admission reserves:
```python
_prealloc_reserved_tokens += req.context_tokens + len(req.output_ids)
```

But offload check only considers:
```python
reserved = sum(r.context_tokens for r in decode_instance.prealloc_reserved)
```

This should be fixed to:
```python
reserved = sum(
    r.context_tokens + len(r.output_ids) for r in decode_instance.prealloc_reserved
)
```

Or better, use the pre-computed counter:
```python
reserved = decode_instance._prealloc_reserved_tokens
```
