"""Test baseline simulation with azure_code_500.csv"""

from config import SimConfig
from core.engine import SimulationEngine

# Run baseline 1P7D
config = SimConfig(
    trace_path="azure_code_500.csv",
    num_prefill_instances=1,
    num_decode_instances=7,
    M=0,
    enable_dynamic_lp=False,
    enable_continuation=False,
    enable_switching=False,
)

print("Running baseline 1P7D simulation...")
engine = SimulationEngine(config)
results = engine.run()

print(f"\nCompleted: {results.total_requests}")
print(f"Simulation time: {results.total_simulation_time:.2f}s")
print(f"Throughput: {results.throughput:.2f} req/s")
