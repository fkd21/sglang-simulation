"""Unit tests for decode TPOT protection mechanism."""

from mechanisms.partial_offload import (
    DynamicBetaSolver,
    calculate_decode_offload_budget,
    _PREFILL_INTERCEPT,
    _PREFILL_COEFF,
    _DECODE_BS_COEFF,
    _DECODE_DTS_COEFF,
)
from request.request import SimReq


class TestBudgetCalculation:
    """Test the calculate_decode_offload_budget function."""

    def test_budget_calculation_idle(self):
        """Test budget when decode is idle."""
        budget = calculate_decode_offload_budget(
            decode_bs=0,
            decode_token_sum=0,
            tpot_sla=0.1,
            prefill_max_tokens=8192,
            num_decode_instances=2,
            num_prefill_instances=2
        )
        assert budget > 0, "Idle decode should have positive budget"

    def test_budget_calculation_overloaded(self):
        """Test budget when decode is overloaded."""
        # Setup: very large decode_bs to exceed TPOT
        budget = calculate_decode_offload_budget(
            decode_bs=1000,
            decode_token_sum=50000,
            tpot_sla=0.1,
            prefill_max_tokens=8192,
            num_decode_instances=2,
            num_prefill_instances=2
        )
        assert budget == 0.0, "Overloaded decode should have zero budget"

    def test_budget_calculation_dp_ratio_effect(self):
        """Test D/P ratio effect on budget."""
        base_budget = calculate_decode_offload_budget(
            decode_bs=10,
            decode_token_sum=5000,
            tpot_sla=0.1,
            prefill_max_tokens=8192,
            num_decode_instances=2,
            num_prefill_instances=2
        )

        # More decode instances → higher budget
        high_budget = calculate_decode_offload_budget(
            decode_bs=10,
            decode_token_sum=5000,
            tpot_sla=0.1,
            prefill_max_tokens=8192,
            num_decode_instances=4,  # 2x more decode
            num_prefill_instances=2
        )

        # Lower decode instances → lower budget
        low_budget = calculate_decode_offload_budget(
            decode_bs=10,
            decode_token_sum=5000,
            tpot_sla=0.1,
            prefill_max_tokens=8192,
            num_decode_instances=1,  # 0.5x decode
            num_prefill_instances=2
        )

        assert high_budget > base_budget, "More decode instances should increase budget"
        assert low_budget < base_budget, "Fewer decode instances should decrease budget"
        assert abs(high_budget - 2 * base_budget) < 1.0, "Budget should scale with D/P ratio"

    def test_budget_calculation_edge_cases(self):
        """Test edge cases in budget calculation."""
        # Zero instances
        budget = calculate_decode_offload_budget(
            decode_bs=0,
            decode_token_sum=0,
            tpot_sla=0.1,
            prefill_max_tokens=8192,
            num_decode_instances=0,
            num_prefill_instances=2
        )
        assert budget == 0.0, "Zero decode instances should return 0 budget"

        budget = calculate_decode_offload_budget(
            decode_bs=0,
            decode_token_sum=0,
            tpot_sla=0.1,
            prefill_max_tokens=8192,
            num_decode_instances=2,
            num_prefill_instances=0
        )
        assert budget == 0.0, "Zero prefill instances should return 0 budget"

    def test_budget_formula_correctness(self):
        """Test budget formula matches expected calculation."""
        decode_bs = 10
        decode_token_sum = 5000
        tpot_sla = 0.05
        prefill_max_tokens = 8192
        num_decode = 2
        num_prefill = 2

        # Calculate expected budget manually
        current_tpot = _PREFILL_INTERCEPT + _DECODE_BS_COEFF * decode_bs + _DECODE_DTS_COEFF * decode_token_sum
        headroom = tpot_sla - current_tpot
        per_iter_budget = headroom / _PREFILL_COEFF
        t_prefill = _PREFILL_INTERCEPT + _PREFILL_COEFF * (0.5 * prefill_max_tokens)
        n_iter = t_prefill / tpot_sla
        expected_budget = per_iter_budget * n_iter * (num_decode / num_prefill)

        actual_budget = calculate_decode_offload_budget(
            decode_bs, decode_token_sum, tpot_sla,
            prefill_max_tokens, num_decode, num_prefill
        )

        assert abs(actual_budget - expected_budget) < 1e-6, "Budget formula should match manual calculation"


class TestLPSolverWithBudget:
    """Test the DynamicBetaSolver with budget constraints."""

    def test_lp_respects_budget(self):
        """Test LP solver respects budget constraint."""
        solver = DynamicBetaSolver(slo_target=1.0, max_window_size=5)

        # Create test requests using SimReq constructor
        requests = [
            SimReq(
                rid=f"req_{i}",
                arrival_time=0.0,
                context_tokens=500,
                generated_tokens=100
            )
            for i in range(5)
        ]

        # Set queue entry times
        for i, req in enumerate(requests):
            req.queue_entry_time = 0.0 + i * 0.1

        # Solve with limited budget
        budget = 1000  # Can only offload 2 full requests (500 tokens each)

        betas, feasible, solve_time = solver.solve_with_decode_budget(
            requests, 0.0, 10, 5000, decode_budget=budget
        )

        total_offload = sum(beta * req.context_tokens
                           for beta, req in zip(betas, requests))

        assert total_offload <= budget + 1e-6, f"Total offload {total_offload} exceeds budget {budget}"

    def test_fallback_to_zero(self):
        """Test fallback when infeasible."""
        solver = DynamicBetaSolver(slo_target=0.1, max_window_size=5)  # Very tight SLO

        requests = [
            SimReq(
                rid=f"req_{i}",
                arrival_time=0.0,
                context_tokens=1000,
                generated_tokens=100
            )
            for i in range(5)
        ]

        for i, req in enumerate(requests):
            req.queue_entry_time = 0.0 + i * 0.1

        # Very small budget makes it infeasible
        betas, feasible, solve_time = solver.solve_with_decode_budget(
            requests, 0.0, 10, 5000, decode_budget=10
        )

        if not feasible:
            assert all(beta == 0.0 for beta in betas), "Infeasible solution should return all zeros"

    def test_zero_budget(self):
        """Test behavior with zero budget."""
        solver = DynamicBetaSolver(slo_target=1.0, max_window_size=5)

        requests = [
            SimReq(
                rid=f"req_{i}",
                arrival_time=0.0,
                context_tokens=500,
                generated_tokens=100
            )
            for i in range(3)
        ]

        for i, req in enumerate(requests):
            req.queue_entry_time = 0.0

        betas, feasible, solve_time = solver.solve_with_decode_budget(
            requests, 0.0, 10, 5000, decode_budget=0.0
        )

        assert all(beta == 0.0 for beta in betas), "Zero budget should result in no offload"
        # Note: feasible may still be True if beta=0 satisfies SLO, which is correct behavior

    def test_sufficient_budget(self):
        """Test behavior with abundant budget."""
        solver = DynamicBetaSolver(slo_target=1.0, max_window_size=5)

        requests = [
            SimReq(
                rid=f"req_{i}",
                arrival_time=0.0,
                context_tokens=500,
                generated_tokens=100
            )
            for i in range(3)
        ]

        for i, req in enumerate(requests):
            req.queue_entry_time = 0.0

        # Very large budget
        betas, feasible, solve_time = solver.solve_with_decode_budget(
            requests, 0.0, 10, 5000, decode_budget=100000
        )

        # With sufficient budget, solution should match unconstrained LP
        betas_unconstrained, _, _ = solver.solve(requests, 0.0, 10, 5000)

        for b1, b2 in zip(betas, betas_unconstrained):
            assert abs(b1 - b2) < 1e-3, "Large budget should not constrain solution"


if __name__ == "__main__":
    # Run all tests
    test_budget = TestBudgetCalculation()
    test_lp = TestLPSolverWithBudget()

    print("Running Budget Calculation Tests...")
    print("- test_budget_calculation_idle...", end=" ")
    test_budget.test_budget_calculation_idle()
    print("PASSED")

    print("- test_budget_calculation_overloaded...", end=" ")
    test_budget.test_budget_calculation_overloaded()
    print("PASSED")

    print("- test_budget_calculation_dp_ratio_effect...", end=" ")
    test_budget.test_budget_calculation_dp_ratio_effect()
    print("PASSED")

    print("- test_budget_calculation_edge_cases...", end=" ")
    test_budget.test_budget_calculation_edge_cases()
    print("PASSED")

    print("- test_budget_formula_correctness...", end=" ")
    test_budget.test_budget_formula_correctness()
    print("PASSED")

    print("\nRunning LP Solver with Budget Tests...")
    print("- test_lp_respects_budget...", end=" ")
    test_lp.test_lp_respects_budget()
    print("PASSED")

    print("- test_fallback_to_zero...", end=" ")
    test_lp.test_fallback_to_zero()
    print("PASSED")

    print("- test_zero_budget...", end=" ")
    test_lp.test_zero_budget()
    print("PASSED")

    print("- test_sufficient_budget...", end=" ")
    test_lp.test_sufficient_budget()
    print("PASSED")

    print("\n✅ All tests passed!")
