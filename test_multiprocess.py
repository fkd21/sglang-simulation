"""Test multiprocessing with engine initialization."""
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import SimConfig
from core.engine import SimulationEngine


def worker_test(task_id):
    """Simple worker that creates an engine and returns."""
    print(f"[Worker {task_id}] Starting...")
    try:
        config = SimConfig(
            trace_path="azure_code_1h.csv",
            num_prefill_instances=1,
            num_decode_instances=1,
            enable_streaming_loading=False,
        )
        print(f"[Worker {task_id}] Config created")

        engine = SimulationEngine(config)
        print(f"[Worker {task_id}] Engine created, iteration_logger={engine.iteration_logger}")

        time.sleep(0.5)  # Simulate some work
        print(f"[Worker {task_id}] Done")
        return {"task_id": task_id, "status": "success"}
    except Exception as e:
        print(f"[Worker {task_id}] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {"task_id": task_id, "status": "failed", "error": str(e)}


def main():
    print("[MAIN] Starting multiprocess test with 3 workers...")

    try:
        with ProcessPoolExecutor(max_workers=3) as executor:
            print("[MAIN] Submitting 3 tasks...")
            futures = [executor.submit(worker_test, i) for i in range(3)]
            print("[MAIN] All tasks submitted, waiting for results...")

            for i, future in enumerate(futures):
                print(f"\n[MAIN] Waiting for task {i}...")
                result = future.result(timeout=10)
                print(f"[MAIN] Task {i} result: {result}")

        print("\n[SUCCESS] All tasks completed!")

    except Exception as e:
        print(f"\n[FATAL] Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
