"""Test dropped request tracking."""

from metrics.metrics_collector import MetricsCollector
from request.request import SimReq

def test_dropped_request_tracking():
    """Test that dropped requests are tracked correctly."""
    collector = MetricsCollector(enable_streaming=True)

    # Create some mock requests
    req1 = SimReq(
        rid="req1",
        arrival_time=0.0,
        context_tokens=100000,  # Oversized
        generated_tokens=100,
    )

    req2 = SimReq(
        rid="req2",
        arrival_time=1.0,
        context_tokens=5000,
        generated_tokens=50,
    )

    # Record drops
    collector.record_dropped_request(req1, "oversized")
    collector.record_dropped_request(req2, "bootstrap_timeout")

    # Debug: check dropped_requests list
    print(f"Dropped requests list length: {len(collector.dropped_requests)}")
    print(f"Dropped requests: {collector.dropped_requests}")

    # Finalize
    results = collector.finalize(total_time=100.0, num_prefill=1, num_decode=1)

    # Check results
    print(f"Total dropped: {results.num_dropped}")
    print(f"Breakdown: {results.dropped_breakdown}")

    assert results.num_dropped == 2
    assert results.dropped_breakdown["oversized"] == 1
    assert results.dropped_breakdown["bootstrap_timeout"] == 1

    print("✅ All tests passed!")

if __name__ == "__main__":
    test_dropped_request_tracking()
