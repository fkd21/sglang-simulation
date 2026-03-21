"""Manual test for EMA smoothing functionality."""

import argparse
import sys

from mechanisms.policy_throughput_sim import PolicyThroughputSim
from mechanisms.worker_state import WorkerState


def _make_worker_state(port: int, role: str, running: int = 0, queue: int = 0) -> WorkerState:
    """Create a WorkerState for testing."""
    metrics = {
        "ok": True,
        "num_running_reqs": running,
        "num_queue_reqs": queue,
    }
    if role == "prefill":
        metrics["num_prefill_inflight_queue_reqs"] = 0
        metrics["max_alpha"] = 0.0
    else:
        metrics["num_decode_transfer_queue_reqs"] = 0
        metrics["num_decode_prealloc_queue_reqs"] = 0
    return WorkerState(port=port, role=role, last_parsed=metrics)


def _make_states(np: int, nd: int, running: int = 0) -> dict:
    """Create worker states for np prefill + nd decode instances."""
    states = {}
    for i in range(np):
        port = 1000 + i
        states[port] = _make_worker_state(port, "prefill", running=running)
    for i in range(nd):
        port = 2000 + i
        states[port] = _make_worker_state(port, "decode", running=running)
    return states


def test_ema_initialization():
    """Test EMA state is initialized correctly on first evaluation."""
    print("\n=== Test 1: EMA Initialization ===")

    args = argparse.Namespace(
        global_cooldown_s=10.0,
        min_prefill=1,
        min_decode=1,
        idle_scrapes=2,
        max_prefill_tokens=8192,
        max_running_requests=1000,
    )

    policy = PolicyThroughputSim(args)
    states = _make_states(np=2, nd=2)

    interval_metrics = {
        "input_throughput": 1000.0,
        "output_throughput": 500.0,
        "avg_decode_computed_token_sum": 5000.0,
        "avg_decode_batch_size": 10.0,
    }

    result = policy.evaluate(now_unix=10.0, states=states, interval_metrics=interval_metrics)

    # First time: EMA should equal raw values
    assert policy._ema_input_throughput == 1000.0, f"Expected 1000.0, got {policy._ema_input_throughput}"
    assert policy._ema_output_throughput == 500.0, f"Expected 500.0, got {policy._ema_output_throughput}"
    assert result["context"]["smoothed_input_throughput"] == 1000.0
    assert result["context"]["smoothed_output_throughput"] == 500.0

    print("✓ EMA initialized correctly")
    print(f"  - EMA input: {policy._ema_input_throughput}")
    print(f"  - EMA output: {policy._ema_output_throughput}")


def test_ema_smoothing_stable_workload():
    """Test EMA smooths small fluctuations in stable workload."""
    print("\n=== Test 2: EMA Smoothing (Stable Workload) ===")

    args = argparse.Namespace(
        global_cooldown_s=10.0,
        min_prefill=1,
        min_decode=1,
        idle_scrapes=2,
        max_prefill_tokens=8192,
        max_running_requests=1000,
    )

    policy = PolicyThroughputSim(args)
    states = _make_states(np=2, nd=2)

    # First evaluation: initialize EMA
    interval_metrics_1 = {
        "input_throughput": 1000.0,
        "output_throughput": 500.0,
        "avg_decode_computed_token_sum": 5000.0,
        "avg_decode_batch_size": 10.0,
    }
    policy.evaluate(now_unix=5.0, states=states, interval_metrics=interval_metrics_1)

    # Second evaluation: small fluctuation
    interval_metrics_2 = {
        "input_throughput": 1050.0,  # +5%
        "output_throughput": 525.0,   # +5%
        "avg_decode_computed_token_sum": 5000.0,
        "avg_decode_batch_size": 10.0,
    }
    result = policy.evaluate(now_unix=40.0, states=states, interval_metrics=interval_metrics_2)

    # With base_alpha=0.3, EMA should be: 0.3 * new + 0.7 * old
    expected_input = 0.3 * 1050.0 + 0.7 * 1000.0  # = 1015.0
    expected_output = 0.3 * 525.0 + 0.7 * 500.0   # = 507.5

    actual_input = result["context"]["smoothed_input_throughput"]
    actual_output = result["context"]["smoothed_output_throughput"]

    print(f"✓ EMA smoothing applied (base_alpha=0.3)")
    print(f"  - Expected smoothed input: {expected_input:.1f}, got {actual_input:.1f}")
    print(f"  - Expected smoothed output: {expected_output:.1f}, got {actual_output:.1f}")

    assert abs(actual_input - expected_input) < 0.1, f"Expected {expected_input}, got {actual_input}"
    assert abs(actual_output - expected_output) < 0.1, f"Expected {expected_output}, got {actual_output}"


def test_ema_high_sensitivity_on_sharp_change():
    """Test EMA uses higher alpha when detecting sharp workload changes."""
    print("\n=== Test 3: EMA High Sensitivity (Sharp Change) ===")

    args = argparse.Namespace(
        global_cooldown_s=10.0,
        min_prefill=1,
        min_decode=1,
        idle_scrapes=2,
        max_prefill_tokens=8192,
        max_running_requests=1000,
    )

    policy = PolicyThroughputSim(args)
    states = _make_states(np=2, nd=2)

    # First evaluation: initialize EMA
    interval_metrics_1 = {
        "input_throughput": 1000.0,
        "output_throughput": 500.0,  # ratio = 2.0
        "avg_decode_computed_token_sum": 5000.0,
        "avg_decode_batch_size": 10.0,
    }
    policy.evaluate(now_unix=5.0, states=states, interval_metrics=interval_metrics_1)

    # Second evaluation: sharp change (ratio changes from 2.0 to 3.0, 50% change > 20% threshold)
    interval_metrics_2 = {
        "input_throughput": 1500.0,  # +50%
        "output_throughput": 500.0,  # same -> ratio = 3.0
        "avg_decode_computed_token_sum": 5000.0,
        "avg_decode_batch_size": 10.0,
    }
    result = policy.evaluate(now_unix=40.0, states=states, interval_metrics=interval_metrics_2)

    # Sharp change detected: should use max_alpha=0.7 instead of base_alpha=0.3
    expected_input = 0.7 * 1500.0 + 0.3 * 1000.0  # = 1350.0
    expected_output = 0.7 * 500.0 + 0.3 * 500.0   # = 500.0

    actual_input = result["context"]["smoothed_input_throughput"]
    actual_output = result["context"]["smoothed_output_throughput"]

    print(f"✓ Sharp change detected, using max_alpha=0.7")
    print(f"  - Ratio change: 2.0 → 3.0 (50% > 20% threshold)")
    print(f"  - Expected smoothed input: {expected_input:.1f}, got {actual_input:.1f}")
    print(f"  - Expected smoothed output: {expected_output:.1f}, got {actual_output:.1f}")

    assert abs(actual_input - expected_input) < 0.1, f"Expected {expected_input}, got {actual_input}"
    assert abs(actual_output - expected_output) < 0.1, f"Expected {expected_output}, got {actual_output}"


def test_ema_custom_parameters():
    """Test that custom EMA parameters are properly used."""
    print("\n=== Test 4: Custom EMA Parameters ===")

    args = argparse.Namespace(
        global_cooldown_s=10.0,
        min_prefill=1,
        min_decode=1,
        idle_scrapes=2,
        max_prefill_tokens=8192,
        max_running_requests=1000,
        ema_base_alpha=0.1,  # Very smooth
        ema_max_alpha=0.9,   # Very responsive
        ema_sensitivity_threshold=0.5,  # High threshold
    )
    policy = PolicyThroughputSim(args)

    assert policy._base_alpha == 0.1, f"Expected 0.1, got {policy._base_alpha}"
    assert policy._max_alpha == 0.9, f"Expected 0.9, got {policy._max_alpha}"
    assert policy._sensitivity_threshold == 0.5, f"Expected 0.5, got {policy._sensitivity_threshold}"

    print("✓ Custom parameters applied correctly")
    print(f"  - base_alpha: {policy._base_alpha}")
    print(f"  - max_alpha: {policy._max_alpha}")
    print(f"  - sensitivity_threshold: {policy._sensitivity_threshold}")


if __name__ == "__main__":
    try:
        test_ema_initialization()
        test_ema_smoothing_stable_workload()
        test_ema_high_sensitivity_on_sharp_change()
        test_ema_custom_parameters()
        print("\n" + "="*50)
        print("✅ All tests passed!")
        print("="*50)
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
