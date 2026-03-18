"""Tests for the optimization mechanisms and PolicyController."""

import pytest
from simulation.mechanisms.partial_offload import DynamicBetaSolver
from simulation.mechanisms.decode_continuation import DecodeContinuation
from simulation.mechanisms.role_switching import RoleSwitcher
from simulation.mechanisms.policy_monitor import PolicyMonitor
from simulation.policy.policy_controller import PolicyController
from simulation.config import SimConfig
from simulation.instances.base_instance import InstanceType, SimInstance
from simulation.instances.prefill_instance import PrefillInstance
from simulation.instances.decode_instance import DecodeInstance
from simulation.request.request import SimReq
from simulation.request.batch import SimBatch
from simulation.core.event import EventType


def make_req(rid, context_tokens, generated_tokens=50):
    """Create a SimReq for testing."""
    return SimReq(rid, 0.0, context_tokens, generated_tokens)


# ---- DynamicBetaSolver ----


class TestDynamicBetaSolver:
    """Tests for dynamic LP-based beta solver."""

    def test_empty_window(self):
        solver = DynamicBetaSolver(slo_target=1.0, max_window_size=5)
        betas, feasible, solve_time = solver.solve([], 0.0, 0, 0)
        assert betas == []
        assert feasible
        assert solve_time == 0.0

    def test_single_request_generous_slo(self):
        """Request that easily meets SLO should get beta near 0."""
        req = make_req("r1", context_tokens=100, generated_tokens=10)
        req.queue_entry_time = 0.0
        solver = DynamicBetaSolver(slo_target=10.0, max_window_size=5)
        betas, feasible, solve_time = solver.solve([req], 0.0, 0, 0)
        assert feasible
        assert len(betas) == 1
        assert betas[0] < 0.01  # Should be near zero with generous SLO
        assert solve_time > 0  # Wall-clock time measured

    def test_tight_slo_may_require_offloading(self):
        """With tight SLO and queue delay, solver may assign positive beta."""
        req = make_req("r1", context_tokens=5000, generated_tokens=10)
        req.queue_entry_time = 0.0
        solver = DynamicBetaSolver(slo_target=0.05, max_window_size=5)
        betas, feasible, solve_time = solver.solve([req], 0.0, 0, 0)
        assert len(betas) == 1
        # With very tight SLO, solver may need to offload
        # (exact result depends on profiling formula)

    def test_window_respects_max_size(self):
        """Only max_window_size requests are processed."""
        reqs = [make_req(f"r{i}", 100, 10) for i in range(10)]
        for r in reqs:
            r.queue_entry_time = 0.0
        solver = DynamicBetaSolver(slo_target=1.0, max_window_size=3)
        betas, feasible, solve_time = solver.solve(reqs, 0.0, 0, 0)
        assert len(betas) == 3  # Only first 3

    def test_betas_bounded_zero_one(self):
        """All betas should be in [0, 1]."""
        reqs = [make_req(f"r{i}", 500, 10) for i in range(5)]
        for r in reqs:
            r.queue_entry_time = 0.0
        solver = DynamicBetaSolver(slo_target=0.5, max_window_size=5)
        betas, feasible, solve_time = solver.solve(reqs, 0.0, 0, 0)
        for beta in betas:
            assert 0.0 <= beta <= 1.0

    def test_solve_time_is_positive(self):
        """Wall-clock solve time should be positive for non-empty input."""
        req = make_req("r1", context_tokens=200, generated_tokens=10)
        req.queue_entry_time = 0.0
        solver = DynamicBetaSolver(slo_target=1.0, max_window_size=5)
        _, _, solve_time = solver.solve([req], 0.0, 0, 0)
        assert solve_time > 0

    def test_decode_state_affects_solver(self):
        """Different decode states should not crash the solver."""
        req = make_req("r1", context_tokens=2000, generated_tokens=10)
        req.queue_entry_time = 0.0
        solver = DynamicBetaSolver(slo_target=0.5, max_window_size=5)

        betas_idle, f1, _ = solver.solve([req], 0.0, decode_bs=0, decode_token_sum=0)
        betas_busy, f2, _ = solver.solve([req], 0.0, decode_bs=50, decode_token_sum=100000)
        assert len(betas_idle) == 1
        assert len(betas_busy) == 1

    def test_multiple_requests_in_window(self):
        """Solver handles multiple requests correctly."""
        reqs = [make_req(f"r{i}", 300 + i * 100, 10) for i in range(5)]
        for r in reqs:
            r.queue_entry_time = 0.0
        solver = DynamicBetaSolver(slo_target=2.0, max_window_size=5)
        betas, feasible, solve_time = solver.solve(reqs, 0.0, 0, 0)
        assert len(betas) == 5
        assert solve_time > 0


# ---- DecodeContinuation ----


class TestDecodeContinuation:
    """Tests for decode continuation (M parameter)."""

    def test_disabled_when_M_zero(self):
        cont = DecodeContinuation(M=0)
        assert not cont.enabled

    def test_enabled_when_M_positive(self):
        cont = DecodeContinuation(M=10)
        assert cont.enabled

    def test_invalid_M_raises(self):
        with pytest.raises(AssertionError):
            DecodeContinuation(M=-1)

    def test_no_continuation_when_disabled(self):
        cont = DecodeContinuation(M=0)
        req = make_req("r1", 100, generated_tokens=50)
        tokens = cont.tokens_to_continue(req)
        assert tokens == 0

    def test_continuation_returns_M(self):
        cont = DecodeContinuation(M=10)
        req = make_req("r1", 100, generated_tokens=50)
        tokens = cont.tokens_to_continue(req)
        assert tokens == 10

    def test_continuation_capped_by_remaining(self):
        """If M > tokens_remaining, return tokens_remaining."""
        cont = DecodeContinuation(M=100)
        req = make_req("r1", 100, generated_tokens=5)
        # 5 tokens to generate, M=100
        tokens = cont.tokens_to_continue(req)
        assert tokens == 5

    def test_continuation_with_partially_generated(self):
        """Accounts for already-generated tokens."""
        cont = DecodeContinuation(M=10)
        req = make_req("r1", 100, generated_tokens=20)
        req.decode_tokens_generated = 15  # 5 remaining
        tokens = cont.tokens_to_continue(req)
        assert tokens == 5  # min(10, 5)

    def test_continuation_exactly_M(self):
        """When remaining > M, returns exactly M."""
        cont = DecodeContinuation(M=25)
        req = make_req("r1", 100, generated_tokens=100)
        tokens = cont.tokens_to_continue(req)
        assert tokens == 25

    def test_continuation_with_zero_remaining(self):
        """No continuation when request is already done."""
        cont = DecodeContinuation(M=10)
        req = make_req("r1", 100, generated_tokens=50)
        req.decode_tokens_generated = 50
        tokens = cont.tokens_to_continue(req)
        assert tokens == 0


# ---- RoleSwitcher ----


class TestRoleSwitcher:
    """Tests for role switching mechanism (simplified - policy logic moved to PolicyMonitor)."""

    def test_disabled_when_never(self):
        switcher = RoleSwitcher(policy="never")
        assert not switcher.enabled

    def test_always_returns_none(self):
        """RoleSwitcher.should_switch always returns None (policies use PolicyMonitor)."""
        switcher = RoleSwitcher(policy="never")
        result = switcher.should_switch(
            InstanceType.PREFILL, 100, 200, 10.0, True
        )
        assert result is None

    def test_record_switch(self):
        """record_switch updates last switch time."""
        switcher = RoleSwitcher(policy="never", switch_cooldown=5.0)
        switcher.record_switch(current_time=20.0)
        assert switcher._last_switch_time == 20.0

    def test_switch_cost_property(self):
        """switch_cost is accessible."""
        switcher = RoleSwitcher(policy="never", switch_cost=0.5)
        assert switcher.switch_cost == 0.5


# ---- PolicyMonitor ----


class TestPolicyMonitor:
    """Tests for PolicyMonitor (policy-driven role switching)."""

    def test_monitor_never_disabled(self):
        """Monitor with 'never' policy is disabled."""
        monitor = PolicyMonitor(policy="never")
        assert not monitor.enabled

    def test_monitor_alpha_enabled(self):
        """Monitor with alpha policy is enabled."""
        monitor = PolicyMonitor(policy="alpha")
        assert monitor.enabled

    def test_monitor_v1_enabled(self):
        """Monitor with v1 policy is enabled."""
        monitor = PolicyMonitor(policy="v1")
        assert monitor.enabled

    def test_monitor_disabled_returns_none(self):
        """Disabled monitor returns None."""
        monitor = PolicyMonitor(policy="never")
        # Create mock instances
        instances = []
        result = monitor.evaluate(instances, current_time=10.0)
        assert result is None

    def test_monitor_with_alpha_high(self):
        """Monitor with alpha policy creates switch event when alpha high."""
        monitor = PolicyMonitor(
            policy="alpha",
            alpha_threshold=0.5,
            stable_evals=1,  # Trigger on first eval
            min_decode=0,  # Allow switching last decode
        )

        # Create one prefill instance with high alpha and one decode instance
        prefill_inst = PrefillInstance(instance_id="prefill_0")
        decode_inst = DecodeInstance(instance_id="decode_0")

        # Add a request with high alpha (prefill_tokens_done / input_len)
        req = SimReq("r1", 0.0, 100, 50)
        req.prefill_tokens_done = 80  # Alpha = 80/100 = 0.8 > 0.5
        prefill_inst.waiting_queue.append(req)

        instances = [prefill_inst, decode_inst]
        result = monitor.evaluate(instances, current_time=10.0)

        # Should create ROLE_SWITCH event for decode→prefill
        assert result is not None
        assert result.event_type == EventType.ROLE_SWITCH
        assert result.data["instance_id"] == "decode_0"
        assert result.data["target_role"] == InstanceType.PREFILL


# ---- PolicyController ----


class TestPolicyController:
    """Tests for PolicyController coordinating all mechanisms."""

    def _make_config(self, **kwargs):
        defaults = dict(
            trace_path="dummy.csv",
            M=0,
            enable_dynamic_lp=False,
            enable_continuation=False,
            enable_switching=False,
            switch_policy="never",
        )
        defaults.update(kwargs)
        return SimConfig(**defaults)

    def test_all_disabled_by_default(self):
        config = self._make_config()
        pc = PolicyController(config)
        assert not pc.offloading_enabled
        assert not pc.continuation_enabled
        assert not pc.switching_enabled

    def test_offloading_enabled_with_dynamic_lp(self):
        config = self._make_config(enable_dynamic_lp=True, slo_target=1.0)
        pc = PolicyController(config)
        assert pc.offloading_enabled
        assert pc.dynamic_lp_enabled

    def test_offloading_disabled_by_default(self):
        config = self._make_config(enable_dynamic_lp=False)
        pc = PolicyController(config)
        assert not pc.offloading_enabled

    def test_continuation_enabled(self):
        config = self._make_config(enable_continuation=True, M=10)
        pc = PolicyController(config)
        assert pc.continuation_enabled

    def test_continuation_disabled_when_flag_false(self):
        config = self._make_config(enable_continuation=False, M=10)
        pc = PolicyController(config)
        assert not pc.continuation_enabled

    def test_switching_enabled(self):
        config = self._make_config(enable_switching=True, switch_policy="load_based")
        pc = PolicyController(config)
        assert pc.switching_enabled

    def test_switching_disabled_when_flag_false(self):
        config = self._make_config(enable_switching=False, switch_policy="load_based")
        pc = PolicyController(config)
        assert not pc.switching_enabled

    def test_solve_dynamic_betas_when_disabled(self):
        """When LP is disabled, returns zero betas."""
        config = self._make_config(enable_dynamic_lp=False)
        pc = PolicyController(config)
        req = make_req("r1", 1000)
        req.queue_entry_time = 0.0
        betas, feasible, solve_time = pc.solve_dynamic_betas([req], 0.0, 0, 0)
        assert betas == [0.0]
        assert feasible
        assert solve_time == 0.0

    def test_solve_dynamic_betas_when_enabled(self):
        """When LP is enabled, solver runs and returns betas."""
        config = self._make_config(enable_dynamic_lp=True, slo_target=10.0)
        pc = PolicyController(config)
        req = make_req("r1", 1000)
        req.queue_entry_time = 0.0
        betas, feasible, solve_time = pc.solve_dynamic_betas([req], 0.0, 0, 0)
        assert len(betas) == 1
        assert 0.0 <= betas[0] <= 1.0
        assert solve_time > 0

    def test_continuation_delegates(self):
        config = self._make_config(enable_continuation=True, M=15)
        pc = PolicyController(config)
        req = make_req("r1", 100, generated_tokens=50)
        tokens = pc.tokens_to_continue_on_prefill(req)
        assert tokens == 15

    def test_should_switch_role_delegates(self):
        """Policy-driven switching moved to PolicyMonitor, so should_switch always returns None."""
        config = self._make_config(
            enable_switching=True, switch_policy="never"
        )
        pc = PolicyController(config)
        result = pc.should_switch_role(
            InstanceType.PREFILL,
            prefill_queue_len=0,
            decode_queue_len=200,
            current_time=100.0,
            instance_is_idle=True,
        )
        # Should always return None (policy logic moved to PolicyMonitor)
        assert result is None

    def test_record_switch_delegates(self):
        config = self._make_config(enable_switching=True, switch_policy="never")
        pc = PolicyController(config)
        pc.record_switch(current_time=5.0)
        assert pc.switcher._last_switch_time == 5.0

    def test_switch_cost_property(self):
        config = self._make_config(enable_switching=True, switch_policy="never")
        pc = PolicyController(config)
        assert pc.switch_cost > 0

    def test_repr(self):
        config = self._make_config(
            enable_dynamic_lp=True, slo_target=1.0,
            enable_continuation=True, M=10,
            enable_switching=True, switch_policy="load_based",
        )
        pc = PolicyController(config)
        r = repr(pc)
        assert "dynamic_lp=True" in r
        assert "M=10" in r
        assert "load_based" in r
