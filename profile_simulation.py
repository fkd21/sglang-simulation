"""Profile a short simulation to find hotspots."""

import cProfile
import pstats
from io import StringIO
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from config import SimConfig
from core.engine import SimulationEngine

# Small config for quick profiling
config = SimConfig(
    trace_path="azure_code_8000.csv",
    num_prefill_instances=2,
    num_decode_instances=2,
    enable_switching=False,
    enable_dynamic_lp=True,
    lp_max_window_size=3,
    enable_monitoring=True,
    monitoring_sample_interval=10.0,
    enable_periodic_plots=False,
    enable_iteration_logging=False,
    enable_request_trace_logging=False,
)

print("Running profiled simulation...")

profiler = cProfile.Profile()
profiler.enable()

engine = SimulationEngine(config)
results = engine.run()

profiler.disable()

# Print top 30 time-consuming functions
s = StringIO()
ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
ps.print_stats(30)
print(s.getvalue())

print("\n" + "="*80)
print("Top functions by total time:")
s = StringIO()
ps = pstats.Stats(profiler, stream=s).sort_stats('tottime')
ps.print_stats(30)
print(s.getvalue())
