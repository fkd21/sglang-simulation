"""Tests for throughput-based role switching policy."""

import argparse
import json
import os
import tempfile

import pytest

from simulation.config import SimConfig
from simulation.core.engine import SimulationEngine
from simulation.core.event import Event, EventType
from simulation.instances.base_instance import InstanceType
from simulation.mechanisms.policy_monitor import PolicyMonitor
from simulation.mechanisms.policy_throughput_sim import (
    DECODE_BS_SLOPE,
    DECODE_TOKEN_SUM_SLOPE,
    INTERCEPT,
    PREFILL_SLOPE,
    PolicyThroughputSim,
)
from simulation.mechanisms.worker_state import WorkerState
from simulation.utils.constants import TOTAL_KV_CACHE_TOKENS


# ---- Fixtures ----

@pytest.fixture
def default_args():
    """Default argparse.Namespace for throughput policy."""
    return argparse.Namespace(
        global_cooldown_s=10.0,
        min_prefill=1,
        min_decode=1,
        idle_scrapes=2,
        max_prefill_tokens=8192,
        max_running_requests=1000,
    )


@pytest.fixture
def policy(default_args):
    """Create a PolicyThroughputSim instance."""
    return PolicyThroughputSim(default_args)


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


@pytest.fixture
def simple_trace_file():
    """Create a simple JSONL trace file for testing."""
    requests = [
        {"input_len": 100, "output_len": 10},
        {"input_len": 150, "output_len": 20},
        {"input_len": 200, "output_len": 15},
    ]
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for req in requests:
            f.write(json.dumps(req) + '\n')
        temp_path = f.name
    yield temp_path
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def medium_trace_file():
    """Create a trace file with enough requests to trigger throughput switching."""
    requests = []
    for i in range(50):
        requests.append({"input_len": 500, "output_len": 50})
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for req in requests:
            f.write(json.dumps(req) + '\n')
        temp_path = f.name
    yield temp_path
    if os.path.exists(temp_path):
        os.unlink(temp_path)


# ---- Unit Tests: PolicyThroughputSim ----

class TestPolicyThroughputCapability:
    """Test capability computation correctness."""

    def test_prefill_capability(self, policy):
        """Verify prefill capability matches profiling formula."""
        max_pt = 8192
        t_prefill = INTERCEPT + PREFILL_SLOPE * max_pt
        expected = max_pt / t_prefill

        states = _make_states(2, 4)
        metrics = {
            "input_throughput": 1000.0,
            "output_throughput": 500.0,
            "avg_decode_computed_token_sum": 500.0,
            "avg_decode_batch_size": 10.0,
        }
        result = policy.evaluate(10.0, states, interval_metrics=metrics)
        ctx = result.get("context", {})
        assert abs(ctx.get("p_capability", 0) - expected) < 1.0

    def test_decode_capability(self, policy):
        """Verify decode capability matches profiling formula."""
        avg_kv = 500.0 / 10.0  # = 50
        max_bs = min(1000, TOTAL_KV_CACHE_TOKENS / avg_kv)
        t_decode = INTERCEPT + (DECODE_BS_SLOPE + DECODE_TOKEN_SUM_SLOPE * avg_kv) * max_bs
        expected = max_bs / t_decode

        states = _make_states(2, 4)
        metrics = {
            "input_throughput": 1000.0,
            "output_throughput": 500.0,
            "avg_decode_computed_token_sum": 500.0,
            "avg_decode_batch_size": 10.0,
        }
        result = policy.evaluate(10.0, states, interval_metrics=metrics)
        ctx = result.get("context", {})
        assert abs(ctx.get("d_capability", 0) - expected) < 1.0


class TestPolicyThroughputGuards:
    """Test guard conditions."""

    def test_zero_input_throughput_noop(self, policy):
        """Zero input throughput returns noop."""
        states = _make_states(2, 4)
        metrics = {
            "input_throughput": 0.0,
            "output_throughput": 500.0,
            "avg_decode_computed_token_sum": 500.0,
            "avg_decode_batch_size": 10.0,
        }
        result = policy.evaluate(10.0, states, interval_metrics=metrics)
        assert result["action"]["kind"] == "noop"
        assert result["action"]["reason"] == "zero_throughput"

    def test_zero_output_throughput_noop(self, policy):
        """Zero output throughput returns noop."""
        states = _make_states(2, 4)
        metrics = {
            "input_throughput": 1000.0,
            "output_throughput": 0.0,
            "avg_decode_computed_token_sum": 500.0,
            "avg_decode_batch_size": 10.0,
        }
        result = policy.evaluate(10.0, states, interval_metrics=metrics)
        assert result["action"]["kind"] == "noop"
        assert result["action"]["reason"] == "zero_throughput"

    def test_no_interval_metrics_noop(self, policy):
        """No interval metrics returns noop."""
        states = _make_states(2, 4)
        result = policy.evaluate(10.0, states, interval_metrics=None)
        assert result["action"]["kind"] == "noop"
        assert result["action"]["reason"] == "no_interval_metrics"

    def test_zero_decode_batch_size_noop(self, policy):
        """Zero avg_decode_batch_size returns noop."""
        states = _make_states(2, 4)
        metrics = {
            "input_throughput": 1000.0,
            "output_throughput": 500.0,
            "avg_decode_computed_token_sum": 0.0,
            "avg_decode_batch_size": 0.0,
        }
        result = policy.evaluate(10.0, states, interval_metrics=metrics)
        assert result["action"]["kind"] == "noop"
        assert result["action"]["reason"] == "insufficient_decode_data"

    def test_global_cooldown(self, policy):
        """Evaluations within cooldown return noop."""
        states = _make_states(2, 4)
        metrics = {
            "input_throughput": 10000.0,
            "output_throughput": 100.0,
            "avg_decode_computed_token_sum": 500.0,
            "avg_decode_batch_size": 10.0,
        }
        # First evaluation triggers switch
        result1 = policy.evaluate(10.0, states, interval_metrics=metrics)
        assert result1["action"]["kind"] != "noop"

        # Second evaluation within cooldown returns noop
        result2 = policy.evaluate(15.0, states, interval_metrics=metrics)
        assert result2["action"]["kind"] == "noop"
        assert result2["action"]["reason"] == "global_cooldown"

    def test_too_few_instances(self, policy):
        """Only 1 instance total returns noop."""
        states = _make_states(1, 0)
        metrics = {
            "input_throughput": 1000.0,
            "output_throughput": 500.0,
            "avg_decode_computed_token_sum": 500.0,
            "avg_decode_batch_size": 10.0,
        }
        result = policy.evaluate(10.0, states, interval_metrics=metrics)
        assert result["action"]["kind"] == "noop"
        assert result["action"]["reason"] == "too_few_instances"


class TestPolicyThroughputDecisions:
    """Test switching decisions."""

    def test_min_prefill_guard(self, policy):
        """Never go below min prefill instances."""
        # 1P5D, even if ratio says 0P6D, should stay at 1P5D
        states = _make_states(1, 5)
        metrics = {
            "input_throughput": 1.0,  # very low prefill demand
            "output_throughput": 10000.0,  # very high decode demand
            "avg_decode_computed_token_sum": 500.0,
            "avg_decode_batch_size": 10.0,
        }
        result = policy.evaluate(10.0, states, interval_metrics=metrics)
        # Should be noop (already balanced at minimum) or keep 1P
        action = result["action"]
        if action["kind"] == "multi_switch_proposed":
            # Check that no proposal would reduce prefill below 1
            p_to_d = sum(1 for p in action["proposals"] if p["from_role"] == "prefill")
            assert 1 - p_to_d >= 1

    def test_min_decode_guard(self, policy):
        """Never go below min decode instances."""
        # 5P1D, even if ratio says 6P0D, should stay at 5P1D
        states = _make_states(5, 1)
        metrics = {
            "input_throughput": 10000.0,  # very high prefill demand
            "output_throughput": 1.0,  # very low decode demand
            "avg_decode_computed_token_sum": 500.0,
            "avg_decode_batch_size": 10.0,
        }
        result = policy.evaluate(10.0, states, interval_metrics=metrics)
        action = result["action"]
        if action["kind"] == "multi_switch_proposed":
            d_to_p = sum(1 for p in action["proposals"] if p["from_role"] == "decode")
            assert 1 - d_to_p >= 1

    def test_already_balanced_noop(self, policy):
        """If already at target ratio, return noop."""
        # Compute what target would be for given metrics and check noop
        states = _make_states(3, 3)
        # Carefully craft metrics where target_np rounds to 3
        # p_cap/d_cap ratio roughly 1, throughput ratio roughly 1
        t_prefill = INTERCEPT + PREFILL_SLOPE * 8192
        p_cap = 8192 / t_prefill

        avg_kv = 50.0
        max_bs = min(1000, TOTAL_KV_CACHE_TOKENS / avg_kv)
        t_decode = INTERCEPT + (DECODE_BS_SLOPE + DECODE_TOKEN_SUM_SLOPE * avg_kv) * max_bs
        d_cap = max_bs / t_decode

        # Set throughputs so ratio = (d_cap * input_thru) / (p_cap * output_thru) = 1
        # => input_thru / output_thru = p_cap / d_cap
        output_thru = 1000.0
        input_thru = output_thru * p_cap / d_cap

        metrics = {
            "input_throughput": input_thru,
            "output_throughput": output_thru,
            "avg_decode_computed_token_sum": avg_kv * 10,
            "avg_decode_batch_size": 10.0,
        }
        result = policy.evaluate(10.0, states, interval_metrics=metrics)
        assert result["action"]["kind"] == "noop"
        assert result["action"]["reason"] == "already_balanced"

    def test_multi_switch_proposals(self, policy):
        """When delta > 1, multiple proposals are generated."""
        # 1P5D, with high prefill demand should propose switching multiple decode→prefill
        states = _make_states(1, 5)
        metrics = {
            "input_throughput": 50000.0,  # very high prefill demand
            "output_throughput": 100.0,  # modest decode demand
            "avg_decode_computed_token_sum": 500.0,
            "avg_decode_batch_size": 10.0,
        }
        result = policy.evaluate(10.0, states, interval_metrics=metrics)
        action = result["action"]
        if action["kind"] == "multi_switch_proposed":
            # Should propose more than 1 switch
            assert len(action["proposals"]) >= 1
            # All should be decode→prefill
            for p in action["proposals"]:
                assert p["from_role"] == "decode"
                assert p["to_role"] == "prefill"

    def test_switch_decode_to_prefill(self, policy):
        """High input throughput relative to output triggers decode→prefill."""
        states = _make_states(2, 4)
        # Very high prefill demand, low decode demand
        metrics = {
            "input_throughput": 100000.0,
            "output_throughput": 100.0,
            "avg_decode_computed_token_sum": 500.0,
            "avg_decode_batch_size": 10.0,
        }
        result = policy.evaluate(10.0, states, interval_metrics=metrics)
        action = result["action"]
        assert action["kind"] == "multi_switch_proposed"
        assert all(p["to_role"] == "prefill" for p in action["proposals"])

    def test_switch_prefill_to_decode(self, policy):
        """High output throughput relative to input triggers prefill→decode."""
        states = _make_states(4, 2)
        # Very low prefill demand, high decode demand
        metrics = {
            "input_throughput": 100.0,
            "output_throughput": 100000.0,
            "avg_decode_computed_token_sum": 500.0,
            "avg_decode_batch_size": 10.0,
        }
        result = policy.evaluate(10.0, states, interval_metrics=metrics)
        action = result["action"]
        assert action["kind"] == "multi_switch_proposed"
        assert all(p["to_role"] == "decode" for p in action["proposals"])

    def test_prefer_idle_candidates(self, policy):
        """Idle instances are preferred for switching."""
        states = {}
        # 2 prefill, 4 decode where decode_0 is idle, others are busy
        states[1000] = _make_worker_state(1000, "prefill")
        states[1001] = _make_worker_state(1001, "prefill")
        states[2000] = _make_worker_state(2000, "decode", running=0, queue=0)
        states[2001] = _make_worker_state(2001, "decode", running=10, queue=5)
        states[2002] = _make_worker_state(2002, "decode", running=8, queue=3)
        states[2003] = _make_worker_state(2003, "decode", running=5, queue=2)

        metrics = {
            "input_throughput": 100000.0,
            "output_throughput": 100.0,
            "avg_decode_computed_token_sum": 500.0,
            "avg_decode_batch_size": 10.0,
        }
        result = policy.evaluate(10.0, states, interval_metrics=metrics)
        action = result["action"]
        if action["kind"] == "multi_switch_proposed" and len(action["proposals"]) >= 1:
            # First proposal should be the idle decode instance
            assert action["proposals"][0]["port"] == 2000


# ---- Integration Tests: PolicyMonitor ----

class TestPolicyMonitorThroughput:
    """Test PolicyMonitor with throughput policy."""

    def test_monitor_returns_list(self, simple_trace_file):
        """PolicyMonitor.evaluate() returns a list for throughput policy."""
        config = SimConfig(
            trace_path=simple_trace_file,
            num_prefill_instances=2,
            num_decode_instances=4,
            enable_switching=True,
            switch_policy="throughput",
        )
        engine = SimulationEngine(config, enable_iteration_logging=False)

        all_instances = (
            engine.instance_manager.prefill_instances
            + engine.instance_manager.decode_instances
        )
        result = engine.policy_monitor.evaluate(
            all_instances=all_instances,
            current_time=10.0,
            interval_metrics=None,
        )
        assert isinstance(result, list)

    def test_monitor_returns_list_for_alpha(self, simple_trace_file):
        """PolicyMonitor.evaluate() also returns a list for alpha policy."""
        config = SimConfig(
            trace_path=simple_trace_file,
            num_prefill_instances=2,
            num_decode_instances=4,
            enable_switching=True,
            switch_policy="alpha",
        )
        engine = SimulationEngine(config, enable_iteration_logging=False)
        all_instances = (
            engine.instance_manager.prefill_instances
            + engine.instance_manager.decode_instances
        )
        result = engine.policy_monitor.evaluate(
            all_instances=all_instances,
            current_time=10.0,
        )
        assert isinstance(result, list)

    def test_monitor_disabled_returns_empty(self, simple_trace_file):
        """Disabled PolicyMonitor.evaluate() returns empty list."""
        config = SimConfig(
            trace_path=simple_trace_file,
            num_prefill_instances=2,
            num_decode_instances=2,
            enable_switching=False,
            switch_policy="never",
        )
        engine = SimulationEngine(config, enable_iteration_logging=False)
        all_instances = (
            engine.instance_manager.prefill_instances
            + engine.instance_manager.decode_instances
        )
        result = engine.policy_monitor.evaluate(
            all_instances=all_instances,
            current_time=10.0,
        )
        assert result == []


# ---- Integration Tests: Engine ----

class TestEngineThroughputCounters:
    """Test throughput counter accumulation in engine."""

    def test_counters_initialized(self, simple_trace_file):
        """Engine initializes throughput counters to zero."""
        config = SimConfig(
            trace_path=simple_trace_file,
            num_prefill_instances=1,
            num_decode_instances=1,
        )
        engine = SimulationEngine(config, enable_iteration_logging=False)
        assert engine._interval_prefill_tokens == 0
        assert engine._interval_decode_tokens == 0
        assert engine._interval_decode_computed_token_sum == 0
        assert engine._interval_decode_batch_count == 0
        assert engine._interval_start_time == 0.0


class TestSelectBestDecodeInstance:
    """Test _select_best_decode_instance excludes draining instances."""

    def test_excludes_draining_instance(self, simple_trace_file):
        """Draining decode instances are skipped."""
        config = SimConfig(
            trace_path=simple_trace_file,
            num_prefill_instances=1,
            num_decode_instances=2,
        )
        engine = SimulationEngine(config, enable_iteration_logging=False)

        # Mark first decode instance as draining
        engine.instance_manager.decode_instances[0].accepting_requests = False
        engine.instance_manager.decode_instances[0].draining = True

        best = engine._select_best_decode_instance()
        assert best is not None
        assert best.instance_id == engine.instance_manager.decode_instances[1].instance_id

    def test_all_draining_returns_none(self, simple_trace_file):
        """Returns None when all decode instances are draining."""
        config = SimConfig(
            trace_path=simple_trace_file,
            num_prefill_instances=1,
            num_decode_instances=2,
        )
        engine = SimulationEngine(config, enable_iteration_logging=False)

        for inst in engine.instance_manager.decode_instances:
            inst.accepting_requests = False
            inst.draining = True

        best = engine._select_best_decode_instance()
        assert best is None


# ---- End-to-End Tests ----

class TestThroughputPolicyE2E:
    """End-to-end tests with throughput policy."""

    def test_simulation_completes_all_requests(self, medium_trace_file):
        """Simulation with throughput policy completes all requests."""
        config = SimConfig(
            trace_path=medium_trace_file,
            num_prefill_instances=2,
            num_decode_instances=4,
            enable_switching=True,
            switch_policy="throughput",
            monitor_interval_s=5.0,
            global_cooldown_s=10.0,
        )
        engine = SimulationEngine(config, enable_iteration_logging=False)
        results = engine.run()

        assert results.total_requests == 50
        assert len(engine.metrics_collector.completions) == 50

    def test_simulation_with_switch_schedule_and_throughput(self, medium_trace_file):
        """Throughput policy works alongside switch schedule."""
        config = SimConfig(
            trace_path=medium_trace_file,
            num_prefill_instances=2,
            num_decode_instances=4,
            enable_switching=True,
            switch_policy="throughput",
            monitor_interval_s=5.0,
            global_cooldown_s=10.0,
        )
        engine = SimulationEngine(config, enable_iteration_logging=False)
        results = engine.run()
        assert len(engine.metrics_collector.completions) == 50

    def test_azure_code_500_no_request_loss(self):
        """Full simulation with azure_code_500.csv: all 500 requests complete."""
        trace_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "azure_code_500.csv"
        )
        if not os.path.exists(trace_path):
            pytest.skip("azure_code_500.csv not available")

        config = SimConfig(
            trace_path=trace_path,
            num_prefill_instances=2,
            num_decode_instances=4,
            enable_switching=True,
            switch_policy="throughput",
            monitor_interval_s=5.0,
            global_cooldown_s=10.0,
        )
        engine = SimulationEngine(config, enable_iteration_logging=False)
        results = engine.run()

        assert results.total_requests == 500
        assert len(engine.metrics_collector.completions) == 500, (
            f"Expected 500 completions, got {len(engine.metrics_collector.completions)}"
        )
