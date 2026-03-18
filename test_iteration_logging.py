"""Test script for iteration logging."""

from pathlib import Path
from main import run_baseline_simulation
from config import SimConfig
from core.engine import SimulationEngine

def test_iteration_logging():
    """Test iteration logging with a small trace."""
    trace_path = Path("AzureLLMInferenceTrace_code.csv")

    # Create config for 1P1D
    config = SimConfig(
        trace_path=str(trace_path),
        num_prefill_instances=1,
        num_decode_instances=1,
        M=0,
        enable_dynamic_lp=False,
        enable_continuation=False,
        enable_switching=False
    )

    # Run with iteration logging enabled
    print("Running simulation with iteration logging enabled...")
    engine = SimulationEngine(config, enable_iteration_logging=True)
    results = engine.run()

    print("\n" + "="*80)
    print("SIMULATION COMPLETED")
    print("="*80)
    print(f"Total requests: {results.total_requests}")
    print(f"SLA attainment: {results.sla_attainment_rate:.1f}%")
    print(f"Check result files in: result/")
    print("="*80)

if __name__ == "__main__":
    test_iteration_logging()
