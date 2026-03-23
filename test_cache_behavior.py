"""Test cache behavior to understand Phase 4 performance."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from config import SimConfig
from core.engine import SimulationEngine

# Small config for quick testing
config = SimConfig(
    trace_path="azure_code_500.csv",  # Smaller trace
    num_prefill_instances=2,
    num_decode_instances=2,
    enable_switching=False,
    enable_dynamic_lp=False,
    enable_monitoring=False,
    enable_periodic_plots=False,
    enable_iteration_logging=False,
    enable_request_trace_logging=False,
)

print("Running simulation with cache instrumentation...")

# Patch the cache functions to track hits/misses
original_cached_fn = None
cache_hits = 0
cache_misses = 0
total_calls = 0

def instrumented_cached(self, decode_instance):
    global cache_hits, cache_misses, total_calls
    total_calls += 1

    # Check if cache is valid
    if self.current_time > self._cache_valid_until:
        cache_misses += 1
        self._invalidate_decode_capacity_cache()
        self._cache_valid_until = self.current_time

    instance_id = decode_instance.instance_id
    if instance_id in self._decode_capacity_cache:
        cache_hits += 1
    else:
        cache_misses += 1

    # Call original
    return original_cached_fn(self, decode_instance)

engine = SimulationEngine(config)

# Monkey-patch
original_cached_fn = engine._decode_allocatable_tokens_cached.__func__
engine._decode_allocatable_tokens_cached = lambda inst: instrumented_cached(engine, inst)

results = engine.run()

print(f"\n{'='*80}")
print("CACHE STATISTICS")
print("="*80)
print(f"Total _decode_allocatable_tokens_cached() calls: {total_calls}")
print(f"Cache hits: {cache_hits}")
print(f"Cache misses: {cache_misses}")
print(f"Hit rate: {cache_hits/total_calls*100:.2f}%" if total_calls > 0 else "N/A")
print(f"\nCache invalidations (timestamp changes): {cache_misses}")
print(f"Average calls per timestamp: {total_calls/cache_misses:.1f}" if cache_misses > 0 else "N/A")
