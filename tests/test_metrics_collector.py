"""Test metrics collection and SLA calculation."""

import pytest
from metrics.metrics_collector import MetricsCollector
from request.request import SimReq


class TestMetricsCollector:
    """Test MetricsCollector class."""

    def test_initialization(self):
        """Test collector initialization."""
        collector = MetricsCollector(ttft_sla=1.0, itl_sla=0.05)

        assert collector.ttft_sla == 1.0
        assert collector.itl_sla == 0.05
        assert len(collector.arrivals) == 0
        assert len(collector.completions) == 0

    def test_record_arrival(self):
        """Test recording arrivals."""
        collector = MetricsCollector()
        req = SimReq("req_1", 1.0, 100, 50)

        collector.record_arrival(req, 1.0)

        assert len(collector.arrivals) == 1
        assert collector.arrivals[0] == (1.0, "req_1")

    def test_record_completion(self):
        """Test recording completions."""
        collector = MetricsCollector()
        req = SimReq("req_1", 1.0, 100, 50)

        req.prefill_end_time = 2.0
        req.decode_start_time = 3.0
        req.decode_end_time = 5.0
        req.decode_tokens_generated = 50

        collector.record_completion(req, 5.0)

        assert len(collector.completions) == 1
        # E2E latency = 5.0 - 1.0 = 4.0
        assert collector.completions[0] == (5.0, "req_1", 4.0)

    def test_ttft_calculation(self):
        """Test TTFT calculation and SLA tracking."""
        collector = MetricsCollector(ttft_sla=1.0)

        # Request that meets TTFT SLA
        req1 = SimReq("req_1", 1.0, 100, 50)
        req1.prefill_end_time = 1.5  # TTFT = 0.5s
        req1.decode_start_time = 2.0
        req1.decode_end_time = 3.0
        req1.decode_tokens_generated = 50

        collector.record_completion(req1, 3.0)

        assert len(collector.ttft_values) == 1
        assert collector.ttft_values[0] == 0.5
        assert collector.ttft_sla_met[0] is True

        # Request that violates TTFT SLA
        req2 = SimReq("req_2", 2.0, 100, 50)
        req2.prefill_end_time = 4.0  # TTFT = 2.0s > 1.0s
        req2.decode_start_time = 5.0
        req2.decode_end_time = 6.0
        req2.decode_tokens_generated = 50

        collector.record_completion(req2, 6.0)

        assert len(collector.ttft_values) == 2
        assert collector.ttft_values[1] == 2.0
        assert collector.ttft_sla_met[1] is False

    def test_itl_calculation(self):
        """Test ITL calculation and SLA tracking."""
        collector = MetricsCollector(itl_sla=0.05)

        # Request that meets ITL SLA
        req1 = SimReq("req_1", 1.0, 100, 50)
        req1.prefill_end_time = 2.0
        req1.decode_start_time = 3.0
        req1.decode_end_time = 4.0
        req1.decode_tokens_generated = 50

        collector.record_completion(req1, 4.0)

        # ITL = (4.0 - 3.0) / 50 = 0.02s
        assert len(collector.itl_values) == 1
        assert abs(collector.itl_values[0] - 0.02) < 1e-6
        assert collector.itl_sla_met[0] is True

        # Request that violates ITL SLA
        req2 = SimReq("req_2", 2.0, 100, 10)
        req2.prefill_end_time = 3.0
        req2.decode_start_time = 4.0
        req2.decode_end_time = 5.0
        req2.decode_tokens_generated = 10

        collector.record_completion(req2, 5.0)

        # ITL = (5.0 - 4.0) / 10 = 0.1s > 0.05s
        assert len(collector.itl_values) == 2
        assert abs(collector.itl_values[1] - 0.1) < 1e-6
        assert collector.itl_sla_met[1] is False

    def test_overall_sla_calculation(self):
        """Test overall SLA attainment (both TTFT and ITL must pass)."""
        collector = MetricsCollector(ttft_sla=1.0, itl_sla=0.05)

        # Request 1: meets both SLAs
        req1 = SimReq("req_1", 1.0, 100, 50)
        req1.prefill_end_time = 1.5  # TTFT = 0.5s ✓
        req1.decode_start_time = 2.0
        req1.decode_end_time = 3.0  # ITL = 1.0/50 = 0.02s ✓
        req1.decode_tokens_generated = 50
        collector.record_completion(req1, 3.0)

        # Request 2: meets TTFT but violates ITL
        req2 = SimReq("req_2", 2.0, 100, 10)
        req2.prefill_end_time = 2.8  # TTFT = 0.8s ✓
        req2.decode_start_time = 3.0
        req2.decode_end_time = 4.0  # ITL = 1.0/10 = 0.1s ✗
        req2.decode_tokens_generated = 10
        collector.record_completion(req2, 4.0)

        # Request 3: violates TTFT but meets ITL
        req3 = SimReq("req_3", 3.0, 100, 50)
        req3.prefill_end_time = 5.5  # TTFT = 2.5s ✗
        req3.decode_start_time = 6.0
        req3.decode_end_time = 7.0  # ITL = 1.0/50 = 0.02s ✓
        req3.decode_tokens_generated = 50
        collector.record_completion(req3, 7.0)

        results = collector.finalize(total_time=10.0, num_prefill=1, num_decode=1)

        # Only req1 meets both SLAs
        assert abs(results.sla_attainment_rate - 33.33) < 0.1  # 1/3 = 33.33%
        assert abs(results.ttft_sla_attainment - 66.67) < 0.1  # 2/3 = 66.67%
        assert abs(results.itl_sla_attainment - 66.67) < 0.1   # 2/3 = 66.67%

    def test_throughput_calculation(self):
        """Test throughput calculation."""
        collector = MetricsCollector()

        # Complete 10 requests over 5 seconds
        for i in range(10):
            req = SimReq(f"req_{i}", float(i), 100, 50)
            req.prefill_end_time = float(i) + 1.0
            req.decode_start_time = float(i) + 2.0
            req.decode_end_time = float(i) + 3.0
            req.decode_tokens_generated = 50
            collector.record_completion(req, float(i) + 3.0)

        results = collector.finalize(total_time=5.0, num_prefill=1, num_decode=1)

        # Throughput = 10 / 5.0 = 2.0 req/s
        assert abs(results.throughput - 2.0) < 1e-6

    def test_empty_results(self):
        """Test finalization with no completions."""
        collector = MetricsCollector()

        results = collector.finalize(total_time=10.0, num_prefill=1, num_decode=1)

        assert results.total_requests == 0
        assert results.throughput == 0.0
        assert results.sla_attainment_rate == 0.0

    def test_mechanism_tracking(self):
        """Test tracking of new mechanisms."""
        collector = MetricsCollector()

        # Request with partial offload
        req1 = SimReq("req_1", 1.0, 100, 50)
        req1.partial_offload_amount = 20
        req1.prefill_end_time = 2.0
        req1.decode_start_time = 3.0
        req1.decode_end_time = 4.0
        req1.decode_tokens_generated = 50
        collector.record_completion(req1, 4.0)

        # Request with continuation
        req2 = SimReq("req_2", 2.0, 100, 50)
        req2.continued_on_prefill = 10
        req2.prefill_end_time = 3.0
        req2.decode_start_time = 4.0
        req2.decode_end_time = 5.0
        req2.decode_tokens_generated = 50
        collector.record_completion(req2, 5.0)

        # Record switch
        collector.record_switch(
            time=3.0,
            instance_id="prefill_0",
            from_role="PREFILL",
            to_role="DECODE",
            drain_time=1.5
        )

        results = collector.finalize(total_time=10.0, num_prefill=1, num_decode=1)

        assert results.num_offloaded == 1
        assert results.num_continued == 1
        assert results.num_switches == 1
        assert len(results.role_switches) == 1
