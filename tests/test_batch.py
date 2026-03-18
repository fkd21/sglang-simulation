"""Test batch operations and metrics."""

import pytest
from simulation.request.request import SimReq
from simulation.request.batch import SimBatch, ForwardMode


class TestSimBatch:
    """Test SimBatch class."""

    def test_initialization(self):
        """Test batch initialization."""
        req1 = SimReq("req_1", 0.0, 100, 50)
        req2 = SimReq("req_2", 0.1, 200, 30)

        batch = SimBatch(
            reqs=[req1, req2],
            forward_mode=ForwardMode.PREFILL
        )

        assert len(batch.reqs) == 2
        assert batch.forward_mode == ForwardMode.PREFILL

    def test_batch_size(self):
        """Test batch size calculation."""
        req1 = SimReq("req_1", 0.0, 100, 50)
        req2 = SimReq("req_2", 0.1, 200, 30)

        batch = SimBatch(
            reqs=[req1, req2],
            forward_mode=ForwardMode.PREFILL
        )

        assert batch.batch_size() == 2

    def test_is_empty(self):
        """Test empty batch check."""
        batch = SimBatch(reqs=[], forward_mode=ForwardMode.PREFILL)
        assert batch.is_empty() is True

        req = SimReq("req_1", 0.0, 100, 50)
        batch.reqs.append(req)
        assert batch.is_empty() is False

    def test_update_metrics_prefill(self):
        """Test metrics update for prefill batch."""
        req1 = SimReq("req_1", 0.0, 100, 50)
        req1.extend_input_len = 80  # After prefix match

        req2 = SimReq("req_2", 0.1, 200, 30)
        req2.extend_input_len = 150

        batch = SimBatch(
            reqs=[req1, req2],
            forward_mode=ForwardMode.PREFILL
        )
        batch.update_metrics()

        assert batch.total_prefill_tokens == 230  # 80 + 150
        assert batch.decode_batch_size == 0
        assert batch.decode_computed_token_sum == 0

    def test_update_metrics_decode(self):
        """Test metrics update for decode batch."""
        req1 = SimReq("req_1", 0.0, 100, 50)
        req1.output_ids = [1, 2, 3, 4, 5]  # 5 tokens generated

        req2 = SimReq("req_2", 0.1, 200, 30)
        req2.output_ids = [1, 2, 3]  # 3 tokens generated

        batch = SimBatch(
            reqs=[req1, req2],
            forward_mode=ForwardMode.DECODE
        )
        batch.update_metrics()

        assert batch.total_prefill_tokens == 0
        assert batch.decode_batch_size == 2
        # decode_computed_token_sum = sum of (context_tokens + output_ids) per req
        assert batch.decode_computed_token_sum == 308  # (100+5) + (200+3)

    # def test_compute_inference_time_prefill(self):
    #     """Test inference time calculation for prefill."""
    #     req1 = SimReq("req_1", 0.0, 1000, 50)
    #     req1.extend_input_len = 1000

    #     batch = SimBatch(
    #         reqs=[req1],
    #         forward_mode=ForwardMode.PREFILL
    #     )
    #     batch.update_metrics()

    #     time = batch.compute_inference_time()

    #     # t = 0.009076 + 7.063e-5 * 1000 = 0.079706
    #     assert abs(time - 0.079706) < 1e-5

    # def test_compute_inference_time_decode(self):
    #     """Test inference time calculation for decode."""
    #     req1 = SimReq("req_1", 0.0, 1000, 50)
    #     req1.output_ids = [1] * 10  # 10 tokens

    #     req2 = SimReq("req_2", 0.1, 2000, 30)
    #     req2.output_ids = [1] * 5  # 5 tokens

    #     batch = SimBatch(
    #         reqs=[req1, req2],
    #         forward_mode=ForwardMode.DECODE
    #     )
    #     batch.update_metrics()

    #     time = batch.compute_inference_time()

    #     # Decode batch size = 2
    #     # decode_computed_token_sum = (1000+10) + (2000+5) = 3015
    #     # t = 0.009076 + 5.623e-5 * 2 + 6.926e-8 * 3015
    #     expected = 0.009076 + 0.00011246 + 6.926e-8 * 3015
    #     assert abs(time - expected) < 1e-5

    def test_compute_inference_time_mixed(self):
        """Test inference time for mixed batch."""
        req1 = SimReq("req_1", 0.0, 1000, 50)
        req1.extend_input_len = 500  # Prefill portion

        req2 = SimReq("req_2", 0.1, 2000, 30)
        req2.output_ids = [1] * 10  # Decode portion

        batch = SimBatch(
            reqs=[req1, req2],
            forward_mode=ForwardMode.MIXED
        )
        batch.update_metrics()

        time = batch.compute_inference_time()

        # Should include both prefill and decode components
        assert time > 0.014654  # More than base latency

    def test_empty_batch_inference_time(self):
        """Test inference time for empty batch."""
        batch = SimBatch(reqs=[], forward_mode=ForwardMode.PREFILL)
        batch.update_metrics()

        time = batch.compute_inference_time()

        # Empty batch should return base latency
        assert abs(time - 0.014654) < 1e-6

    def test_forward_mode_enum(self):
        """Test ForwardMode enum."""
        # Enum values are auto-generated integers, not strings
        assert isinstance(ForwardMode.PREFILL.value, int)
        assert isinstance(ForwardMode.DECODE.value, int)
        assert isinstance(ForwardMode.MIXED.value, int)
        assert isinstance(ForwardMode.EXTEND.value, int)
