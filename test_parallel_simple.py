"""Simple test to diagnose parallel execution issues."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

print("[DEBUG] Starting import test...")

try:
    print("[DEBUG] Importing config...")
    from config import SimConfig
    print("[DEBUG] ✓ Config imported")

    print("[DEBUG] Importing engine...")
    from core.engine import SimulationEngine
    print("[DEBUG] ✓ Engine imported")

    print("[DEBUG] Creating simple config...")
    config = SimConfig(
        trace_path="azure_code_1h.csv",
        num_prefill_instances=1,
        num_decode_instances=1,
        enable_streaming_loading=False,
    )
    print("[DEBUG] ✓ Config created")

    print("[DEBUG] Creating engine...")
    engine = SimulationEngine(config)
    print("[DEBUG] ✓ Engine created successfully")
    print(f"[DEBUG] iteration_logger is: {engine.iteration_logger}")
    print(f"[DEBUG] request_trace_logger is: {engine.request_trace_logger}")

except Exception as e:
    print(f"[ERROR] Failed during import/initialization: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[SUCCESS] All imports and initialization successful!")
