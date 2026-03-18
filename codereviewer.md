Simulation vs Real SGLang PD Disaggregation: Comparative Review
Context
This is a comprehensive code review comparing the simulation under simulation/ against the real SGLang PD disaggregation implementation. The goal is to verify fidelity and identify mismatches that could lead to incorrect simulation results.

ISSUE 1: decode_computed_token_sum Missing Context Tokens (HIGH IMPACT)
Simulation: batch.py:81-83


self.decode_computed_token_sum = sum(
    len(req.output_ids) for req in self.reqs
)
Real system: In ScheduleBatch.prepare_for_decode(), seq_lens tracks len(origin_input_ids) + len(output_ids) — the full KV cache footprint per request.

Difference: The simulation only sums output_ids count, but the real system's decode attention cost scales with the full sequence length (context + output). The profiling formula 6.926e-8 * decode_computed_token_sum should use context_tokens + len(output_ids) for each request.

Impact: Critical correctness problem. Decode iteration times are significantly underestimated, especially for long-context requests. A request with 4000 context tokens and 10 output tokens would contribute 10 instead of 4010 — a 400x error for that term.

Fix: sum(req.context_tokens + len(req.output_ids) for req in self.reqs)

ISSUE 2: No rem_total_tokens / Conservative Memory Budgeting (MEDIUM-HIGH IMPACT)
Simulation: prefill_adder.py:66-116

Only tracks rem_input_tokens and rem_chunk_tokens
Memory check is purely via _try_allocate_kv_cache(num_tokens) (checks pool + eviction)
Real system: schedule_policy.py:376-396


@property
def rem_total_tokens(self):
    available_and_evictable = (
        self.token_to_kv_pool_allocator.available_size()
        + self.tree_cache.evictable_size()
    )
    return available_and_evictable - self.rem_total_token_offset
rem_total_token_offset includes running batch's estimated future decode tokens: max_new_tokens * new_token_ratio for each running request
_update_prefill_budget adds extend_input_len + max_new_tokens to offset
total_tokens >= rem_total_tokens is checked BEFORE trying to add each request
Difference: The simulation doesn't reserve memory for future decode token generation. The real system conservatively accounts for how many tokens running decode requests might still produce, preventing OOM during decode.

Classification: Intentional simplification that becomes a correctness issue under memory pressure. The simulation may over-admit prefill requests, leading to scenarios that would cause retraction in the real system.

Impact: Under high load, simulation admits more prefill requests than the real system would, producing optimistic throughput and lower latency estimates.

ISSUE 3: No Decode Retraction Mechanism (MEDIUM IMPACT)
Simulation: engine.py:509-550 — decode batch completion simply generates tokens with no memory check.

Real system: scheduler.py:1857-1900


def update_running_batch(self, batch):
    if not batch.check_decode_mem(self.decode_mem_cache_buf_multiplier):
        retracted_reqs, new_token_ratio = batch.retract_decode(self.server_args)
        # Retracted requests go back to waiting queue
Difference: The real system checks decode memory before each decode iteration and retracts (evicts) low-priority requests if memory is insufficient. The simulation has no retraction — once a request enters decode, it runs to completion.

Classification: Missing mechanism. Important under memory pressure with many concurrent decode requests.

Impact: The simulation never models decode-side memory pressure, giving optimistic results when the decode instance's KV cache is near capacity.

ISSUE 4: No Page Alignment (LOW-MEDIUM IMPACT)
Simulation: No concept of page-aligned allocation.

Real system: schedule_policy.py:420-421


def ceil_paged_tokens(self, tokens: int) -> int:
    return -(-tokens // self.page_size) * self.page_size
All budget checks use page-aligned sizes.

Difference: The real system rounds up allocations to page boundaries (typically 1 token, but configurable). This wastes some memory per allocation.

Classification: Intentional simplification. Low impact since the default page_size is 1 in SGLang for non-paged setups.

Impact: Minimal unless page_size > 1 is used in the target configuration.

ISSUE 5: No Decode-Side KV Cache Pre-allocation (MEDIUM IMPACT)
Simulation: engine.py:473-483 — KV_TRANSFER_COMPLETE directly adds to decode waiting_queue, memory allocated implicitly via allocate_memory() during batch formation.

Real system: decode.py:171-400

DecodePreallocQueue: Pre-allocates KV cache memory on decode side before transfer begins
DecodeTransferQueue: Polls for actual KV data arrival
Memory is reserved early to unblock the prefill instance from waiting
Difference: The simulation skips the pre-allocation pipeline. In the real system, memory contention on the decode side can delay pre-allocation, which in turn delays the prefill side (since KV transfer can't start until memory is allocated). The simulation doesn't model this backpressure.

Classification: Missing mechanism. Medium impact on scheduling fidelity under high decode-side memory pressure.

Impact: The simulation may show smoother KV transfer flow than the real system when decode memory is constrained.

ISSUE 6: No Bootstrap Handshake Latency (LOW IMPACT)
Simulation: Requests go directly from arrival to waiting queue to scheduling.

Real system: prefill.py:67-200 — Requests must go through PrefillBootstrapQueue where they complete a handshake (sender initialization, HTTP registration) before being moved to the waiting queue.

Difference: The real system has a non-trivial bootstrap phase per request involving network communication between prefill and decode instances. This adds latency before a request can be scheduled for prefill.

Classification: Intentional simplification. The bootstrap overhead is typically small relative to prefill/decode times but can become significant under high request rates.

Impact: Low in most scenarios. Could cause the simulation to be slightly optimistic for TTFT under burst arrivals.

ISSUE 7: Prefill Side — Missing cache_unfinished_req for Chunked Requests (LOW-MEDIUM IMPACT)
Simulation: engine.py:284-288 — For chunked requests, just frees memory and continues.

Real system: scheduler.py:1608


self.tree_cache.cache_unfinished_req(self.chunked_req, chunked=True)
Before freeing req_pool_idx, the real system caches the partially-computed KV in the radix tree so it can be reused as a prefix in the next round.

Difference: In the simulation, the KV from a chunked round is freed entirely and the next round must recompute from scratch (or rely on prefill_tokens_done to skip). In the real system, the partial KV is cached as a prefix, so the next round benefits from cache hits on the already-computed portion.

Classification: Potential correctness issue for chunked prefill behavior. The simulation accounts for this via prefill_tokens_done in _init_next_round_input, which achieves the same effect of not recomputing already-done tokens. But the radix tree state differs.

Impact: Low. The prefill_tokens_done mechanism in simulation achieves the correct effective behavior even without caching in the tree.

ISSUE 8: is_chunked Counter Semantics (LOW IMPACT)
Simulation: prefill_adder.py:95 — req.is_chunked += 1 inside adder.

Real system: scheduler.py:1803 — self.chunked_req.is_chunked += 1 in scheduler after adder returns. In process_batch_result_disagg_prefill, req.is_chunked -= 1 after processing.

Difference: The counter semantics differ slightly. In the real system, is_chunked > 0 means the request is still being chunked (used as a flag in batch result processing). The simulation also uses it as a counter but increments it in a different place.

Classification: Minor semantic difference. No functional impact since the simulation uses req is chunked_req identity check rather than is_chunked counter.

Impact: Negligible.

ISSUE 9: Missing batch_is_full Flag and Capacity Checks (MEDIUM IMPACT)
Simulation: No concept of batch_is_full — simply stops adding when budget is exhausted.

Real system: scheduler.py:1687-1703


if (self.running_batch.batch_is_full or len(self.waiting_queue) == 0) and self.chunked_req is None:
    return None
if self.get_num_allocatable_reqs(running_bs) <= 0 and not self.chunked_req:
    self.running_batch.batch_is_full = True
    return None
batch_is_full persists across scheduling rounds, preventing unnecessary scheduling attempts
get_num_allocatable_reqs checks pp_max_micro_batch_size - running_bs
In disagg prefill mode: additional req_to_token_pool.available_size() check
Difference: The simulation checks only token budgets, not request-slot limits. The real system limits the total number of concurrent requests via max_running_requests / pp_max_micro_batch_size.

Classification: Missing mechanism that affects behavior when the running batch is large.

Impact: Medium. Without request-count limits, the simulation can schedule more requests than the real system would allow.

ISSUE 10: Missing DFS-Weight and LOF Scheduling Policies (LOW IMPACT)
Simulation: schedule_policy.py — Supports FCFS, LPM, RANDOM.

Real system: schedule_policy.py:95-308 — Supports LPM, DFS-Weight, FCFS, LOF, RANDOM, plus priority scheduling with preemption.

Difference: Missing DFS-weight (depth-first tree weighting for better cache locality), LOF (longest output first), and priority-based preemption.

Classification: Intentional simplification. The simulation_rule.md doesn't specify which policies need to be supported.

Impact: Low if using FCFS (default). Could matter for workloads that benefit from DFS-weight or priority scheduling.

ISSUE 11: Decode Scheduling Does Not Allocate KV for Decode Tokens (MEDIUM IMPACT)
Simulation: engine.py:600-623 — Decode requests are moved from waiting_queue to running_batch without allocating additional KV for each new decode token.

Real system: prepare_for_decode() allocates one new KV cache slot per request per decode iteration via alloc_for_decode().

Difference: The simulation doesn't model per-token KV allocation during decode. After a request enters decode, its KV footprint never grows in the simulation's memory accounting.

Classification: Missing mechanism. Combined with Issue 3 (no retraction), this means the simulation never models decode-side memory exhaustion.

Impact: Medium. The simulation will not detect memory pressure during decode that the real system would, leading to overly optimistic results under high decode concurrency.

ISSUE 12: No new_token_ratio Adaptation (LOW IMPACT)
Simulation: prefill_scheduler.py:53-60 — Has decay_token_ratio() method but new_token_ratio defaults to 0.0 per config.schedule_conservativeness.

Real system: scheduler.py:1870-1894 — new_token_ratio starts at schedule_conservativeness and adaptively decays when decode succeeds, resets when retraction happens.

Difference: The simulation's schedule_conservativeness defaults to 0 (as stated in simulation_rule.md: --schedule-conservativeness 0), making this difference moot for the baseline config. The simulation has the decay mechanism but it's never triggered.

Classification: Intentional match to the specified config.

Impact: None for baseline config. Low impact if schedule_conservativeness > 0 is tested.

ISSUE 13: Prefill Memory Not Freed After Request Completes (LOW IMPACT)
Simulation: engine.py:290-298 — After prefill completes, the KV is inserted into the radix tree cache but not freed from the KV pool. Memory is only freed on the decode side upon request completion.

Real system: In disagg prefill mode, after prefill completes and KV is transferred, the prefill instance's memory for that request should eventually be reclaimable via tree cache eviction.

Difference: The simulation inserts into radix cache after prefill completion, which makes the memory evictable but not immediately free. This is actually consistent with the real system's behavior — the tree cache holds the KV until eviction. So this is likely correct.

Classification: Likely correct behavior.

Impact: None.

ISSUE 14: KV Transfer is Per-Request, Not Batched (LOW IMPACT)
Simulation: engine.py:440-455 — Each request gets its own KV_TRANSFER_START/COMPLETE event.

Real system: KV transfer can be batched/pipelined via thread pool (SGLANG_DISAGGREGATION_THREAD_POOL_SIZE). Multiple transfers can happen concurrently.

Difference: The simulation models KV transfer as independent per-request events that happen concurrently (since they don't block the prefill/decode instance). The transfer time formula 1.86683e-6 * total_prefill_tokens is applied per-request. This is reasonable for RDMA-based transfers.

Classification: Acceptable simplification. Concurrent transfers are modeled by non-blocking events.

Impact: Low. Could matter if transfers share bandwidth (not modeled).

ISSUE 15: Load Balancing is Only Round-Robin (LOW IMPACT)
Simulation: instance_manager.py — Round-robin for both prefill and decode instance selection.

Real system: Router supports multiple policies (round-robin, shortest-queue, etc.). Load info includes num_reqs, num_tokens, num_waiting_reqs.

Difference: The simulation only supports round-robin. The real system's router can use more sophisticated policies.

Classification: Intentional simplification. Round-robin is a common baseline.

Impact: Low for balanced workloads. Could matter for heterogeneous request sizes.

ISSUE 16: Missing mixed_with_decode_tokens Budget Adjustment (LOW IMPACT for Disagg)
Simulation: prefill_adder.py:63 — rem_input_tokens = max_prefill_tokens with no decode token deduction.

Real system: schedule_policy.py:334 — self.rem_input_tokens = rem_input_tokens - mixed_with_decode_tokens

Difference: In mixed-chunk mode, the real system subtracts running decode tokens from the prefill budget. In disaggregated P/D mode, prefill instances don't have running decode batches, so mixed_with_decode_tokens = 0. The simulation passes running_decode_bs=0 to the adder.

Classification: Correct for disagg mode. Would need fixing if modeling unified (non-disagg) mode.

Impact: None in disagg mode.

ISSUE 17: AddReqResult Missing OTHER Variant (LOW IMPACT)
Simulation: Has CONTINUE, NO_TOKEN, NO_MEM.

Real system: Has CONTINUE, NO_TOKEN, OTHER (used for "can't add more to batch but not out of tokens").

Difference: The simulation uses NO_MEM instead of OTHER for some cases. The real system distinguishes between "out of total memory" and "out of per-batch budget" — NO_TOKEN breaks the loop while OTHER also breaks but for non-memory reasons.

Classification: Cosmetic difference. Functional behavior is equivalent.

Impact: None.

Summary
Consistent with real implementation:
Discrete event simulation architecture — faithfully models the async event-driven nature of SGLang
Chunked prefill mechanism — correctly implements multi-round chunking with prefill_tokens_done tracking
Radix cache with LRU eviction — prefix matching and cache management works correctly
PrefillAdder admission control — core logic of token budget checking and chunking decision matches
KV transfer as non-blocking events — correctly allows prefill/transfer/decode overlap
Request lifecycle stages — correct state machine for P/D disaggregation
Profiling formulas — correctly applied (except decode_computed_token_sum, see below)
Scheduling policies — FCFS and LPM correctly implemented
Inconsistent or incomplete:
decode_computed_token_sum excludes context tokens — CRITICAL BUG causing major underestimation of decode times
No conservative memory budgeting (rem_total_tokens / new_token_ratio) — leads to over-admission
No decode retraction — no handling of decode-side memory pressure
No decode-side KV allocation tracking — decode KV footprint never grows after initial allocation
No decode-side pre-allocation pipeline — missing PreallocQueue/TransferQueue
No request-count limits (batch_is_full / max_running_requests)
Missing DFS-weight and priority scheduling policies
Most important fixes (priority order):
FIX decode_computed_token_sum in batch.py:81-83 — change to sum(req.context_tokens + len(req.output_ids) for req in self.reqs). This is a one-line bug fix with the highest impact on simulation fidelity.

Add rem_total_tokens conservative budgeting in prefill_adder.py — track running decode batch's estimated future tokens in the memory budget. This prevents over-admission of prefill requests.

Add decode-side KV allocation per token — allocate 1 KV slot per decode iteration per request, and check memory availability before each decode batch.

Add decode retraction — when decode KV pool is full, retract low-priority requests back to the waiting queue with new_token_ratio reset.

Add request-count limits — enforce max_running_requests to prevent unlimited request concurrency.

Verification Plan
After applying fixes:

Run existing test suite: cd simulation && python -m pytest tests/ -v
Compare simulation output for small workloads (10-50 requests) against known SGLang benchmark results
Verify decode iteration times match the profiling formula with context tokens included
Test memory pressure scenarios: workloads where total KV exceeds capacity, verify retraction triggers