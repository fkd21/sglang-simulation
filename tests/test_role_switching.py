"""Tests for role switching mechanism."""

import pytest
from config import SimConfig
from core.engine import SimulationEngine
from core.event import Event, EventType
from instances.base_instance import InstanceType
from instances.prefill_instance import PrefillInstance
from instances.decode_instance import DecodeInstance
from request.request import SimReq
import tempfile
import os


@pytest.fixture
def simple_trace_file():
    """Create a simple JSONL trace file for testing."""
    import json

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

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def long_trace_file():
    """Create a trace file with requests arriving over time to extend simulation."""
    import json

    # Create many requests with larger output to make simulation run longer
    requests = []
    for i in range(20):
        requests.append({"input_len": 100, "output_len": 100})  # Each takes ~10s to decode

    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for req in requests:
            f.write(json.dumps(req) + '\n')
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


def test_switch_schedule_loading(simple_trace_file):
    """Test that switch schedule is correctly loaded and creates events."""
    config = SimConfig(
        trace_path=simple_trace_file,
        num_prefill_instances=2,
        num_decode_instances=2,
        switch_schedule=[
            {"time": 1.0, "instance_id": "prefill_0", "target_role": "DECODE"},
            {"time": 2.0, "instance_id": "decode_1", "target_role": "PREFILL"},
        ]
    )

    engine = SimulationEngine(config, enable_iteration_logging=False)

    # Check that ROLE_SWITCH events were created
    switch_events = [e for e in engine.event_queue if e.event_type == EventType.ROLE_SWITCH]
    assert len(switch_events) == 2
    assert switch_events[0].timestamp == 1.0
    assert switch_events[1].timestamp == 2.0


def test_instance_state_fields():
    """Test that instance has correct role switching state fields."""
    instance = PrefillInstance(instance_id="test_prefill")

    assert hasattr(instance, 'draining')
    assert hasattr(instance, 'switch_target_role')
    assert hasattr(instance, 'switch_initiated_time')
    assert hasattr(instance, 'blocked_until')
    assert hasattr(instance, 'accepting_requests')

    assert instance.draining == False
    assert instance.switch_target_role is None
    assert instance.accepting_requests == True


def test_request_migration_fields():
    """Test that request has migration tracking fields."""
    req = SimReq(
        rid="test_req",
        arrival_time=0.0,
        context_tokens=100,
        generated_tokens=10
    )

    assert hasattr(req, 'migrated_from_switch')
    assert hasattr(req, 'migration_timestamp')

    assert req.migrated_from_switch == False
    assert req.migration_timestamp == 0.0


def test_initiate_switch_idle_instance(simple_trace_file):
    """Test initiating switch on an idle instance."""
    config = SimConfig(
        trace_path=simple_trace_file,
        num_prefill_instances=2,
        num_decode_instances=2,
        switch_min_blocking_time=5.0,
    )

    engine = SimulationEngine(config, enable_iteration_logging=False)

    # Get a prefill instance
    prefill_instance = engine.instance_manager.prefill_instances[0]

    # Ensure it's idle
    assert not prefill_instance.has_running_requests()
    assert prefill_instance.accepting_requests

    # Initiate switch
    events = engine._initiate_role_switch(prefill_instance, InstanceType.DECODE)

    # Check instance state
    assert prefill_instance.draining == True
    assert prefill_instance.switch_target_role == InstanceType.DECODE
    assert prefill_instance.accepting_requests == False

    # Should have SWITCH_UNBLOCK event since instance is idle
    assert len(events) == 1
    assert events[0].event_type == EventType.SWITCH_UNBLOCK
    assert events[0].timestamp == engine.current_time + config.switch_min_blocking_time


def test_no_accept_during_drain(simple_trace_file):
    """Test that draining instance doesn't accept new requests."""
    config = SimConfig(
        trace_path=simple_trace_file,
        num_prefill_instances=2,
        num_decode_instances=2,
    )

    engine = SimulationEngine(config, enable_iteration_logging=False)

    # Get a prefill instance and mark it as draining
    prefill_instance = engine.instance_manager.prefill_instances[0]
    prefill_instance.draining = True
    prefill_instance.accepting_requests = False

    # Try to admit from bootstrap
    result = engine._try_admit_from_bootstrap(prefill_instance)

    # Should return empty list
    assert result == []


def test_select_instance_for_migrated():
    """Test selecting instance for migrated request."""
    from instances.instance_manager import InstanceManager

    manager = InstanceManager(num_prefill=3, num_decode=3)

    # Mark one instance as draining
    manager.prefill_instances[1].draining = True
    manager.prefill_instances[1].accepting_requests = False

    # Select instance for migrated request
    target = manager.select_instance_for_migrated(
        InstanceType.PREFILL,
        exclude_instance_id="prefill_0"
    )

    # Should not select prefill_0 (excluded) or prefill_1 (draining)
    assert target is not None
    assert target.instance_id == "prefill_2"


def test_all_instances_draining_defers_requests(simple_trace_file):
    """Test that requests are deferred when all instances are draining."""
    config = SimConfig(
        trace_path=simple_trace_file,
        num_prefill_instances=2,
        num_decode_instances=2,
    )

    engine = SimulationEngine(config, enable_iteration_logging=False)

    # Mark all prefill instances as draining
    for pi in engine.instance_manager.prefill_instances:
        pi.draining = True
        pi.accepting_requests = False

    # Create a request arrival event
    req = SimReq(
        rid="test_req",
        arrival_time=0.0,
        context_tokens=100,
        generated_tokens=10
    )

    event = Event(
        timestamp=0.0,
        event_type=EventType.REQUEST_ARRIVAL,
        data={"request": req}
    )

    # Handle request arrival
    events = engine._handle_request_arrival(event)

    # Should defer the request
    assert len(events) == 1
    assert events[0].event_type == EventType.REQUEST_ARRIVAL
    assert events[0].timestamp == engine.current_time + 0.1


def test_migration_priority_insertion(simple_trace_file):
    """Test that migrated requests are inserted at queue head."""
    config = SimConfig(
        trace_path=simple_trace_file,
        num_prefill_instances=2,
        num_decode_instances=2,
    )

    engine = SimulationEngine(config, enable_iteration_logging=False)

    source = engine.instance_manager.prefill_instances[0]
    target = engine.instance_manager.prefill_instances[1]

    # Mark target as busy to prevent scheduling
    target.busy = True

    # Add some requests to target's waiting queue
    for i in range(3):
        req = SimReq(
            rid=f"existing_{i}",
            arrival_time=0.0,
            context_tokens=100,
            generated_tokens=10
        )
        target.waiting_queue.append(req)

    # Add a request to source's waiting queue
    migrated_req = SimReq(
        rid="migrated",
        arrival_time=0.0,
        context_tokens=100,
        generated_tokens=10
    )
    source.waiting_queue.append(migrated_req)

    # Migrate from source
    count = engine._migrate_prefill_waiting_queue(source)

    # Check migration happened
    assert count == 1
    assert migrated_req not in source.waiting_queue
    assert migrated_req in target.waiting_queue

    # Check priority: migrated request should be at head (first in queue)
    assert target.waiting_queue[0] == migrated_req
    assert migrated_req.migrated_from_switch == True


def test_metrics_record_switch():
    """Test that metrics correctly record switch events."""
    from metrics.metrics_collector import MetricsCollector

    collector = MetricsCollector()

    collector.record_switch(
        time=10.5,
        instance_id="prefill_0",
        from_role="PREFILL",
        to_role="DECODE",
        drain_time=2.3
    )

    assert collector.num_switches == 1
    assert len(collector.role_switches) == 1

    switch = collector.role_switches[0]
    assert switch["time"] == 10.5
    assert switch["instance_id"] == "prefill_0"
    assert switch["from_role"] == "PREFILL"
    assert switch["to_role"] == "DECODE"
    assert switch["drain_time"] == 2.3


def test_switch_unblock_changes_role(simple_trace_file):
    """Test that unblock event correctly changes instance role."""
    config = SimConfig(
        trace_path=simple_trace_file,
        num_prefill_instances=2,
        num_decode_instances=2,
    )

    engine = SimulationEngine(config, enable_iteration_logging=False)

    # Get a prefill instance
    instance = engine.instance_manager.prefill_instances[0]
    old_id = instance.instance_id

    # Set up for unblock
    instance.draining = True
    instance.switch_target_role = InstanceType.DECODE
    instance.switch_initiated_time = 0.0
    instance.accepting_requests = False

    # Create unblock event
    event = Event(
        timestamp=5.0,
        event_type=EventType.SWITCH_UNBLOCK,
        data={"instance": instance}
    )

    # Handle unblock
    engine.current_time = 5.0
    events = engine._handle_switch_unblock(event)

    # Check instance state changed
    assert instance.instance_type == InstanceType.DECODE
    assert instance.draining == False
    assert instance.accepting_requests == True
    assert instance.switch_target_role is None

    # Check scheduler was recreated
    assert old_id in engine.decode_schedulers
    assert old_id not in engine.prefill_schedulers


def test_drain_completion_detection(simple_trace_file):
    """Test that drain completion is detected when running_batch empties."""
    config = SimConfig(
        trace_path=simple_trace_file,
        num_prefill_instances=2,
        num_decode_instances=2,
        switch_min_blocking_time=5.0,
    )

    engine = SimulationEngine(config, enable_iteration_logging=False)

    # Get a prefill instance
    instance = engine.instance_manager.prefill_instances[0]

    # Mark as draining with empty running batch
    instance.draining = True
    instance.switch_target_role = InstanceType.DECODE
    instance.switch_initiated_time = 0.0

    # Ensure running batch is empty
    assert not instance.has_running_requests()

    # Call _complete_drain
    events = engine._complete_drain(instance)

    # Should create SWITCH_UNBLOCK event
    assert len(events) == 1
    assert events[0].event_type == EventType.SWITCH_UNBLOCK
    assert events[0].timestamp == engine.current_time + config.switch_min_blocking_time


def test_blocking_period_minimum_5s(simple_trace_file):
    """Test that blocking period is at least 5s."""
    config = SimConfig(
        trace_path=simple_trace_file,
        num_prefill_instances=2,
        num_decode_instances=2,
        switch_min_blocking_time=5.0,
    )

    engine = SimulationEngine(config, enable_iteration_logging=False)
    instance = engine.instance_manager.prefill_instances[0]

    # Initiate switch on idle instance (drain_time = 0)
    events = engine._initiate_role_switch(instance, InstanceType.DECODE)

    # Check blocking time is at least 5s
    assert len(events) == 1
    assert events[0].timestamp == engine.current_time + 5.0


def test_blocking_period_uses_drain_time(simple_trace_file):
    """Test that blocking period = max(drain_duration, switch_min_blocking_time).

    Drain and blocking run concurrently. The instance unblocks at:
      max(drain_complete_time, switch_initiated_time + min_blocking_time)
    """
    config = SimConfig(
        trace_path=simple_trace_file,
        num_prefill_instances=2,
        num_decode_instances=2,
        switch_min_blocking_time=5.0,
    )

    engine = SimulationEngine(config, enable_iteration_logging=False)
    instance = engine.instance_manager.prefill_instances[0]

    # Case 1: drain (10s) > min_blocking (5s) → unblock at drain_complete_time
    instance.draining = True
    instance.switch_initiated_time = 0.0
    engine.current_time = 10.0  # drain took 10 seconds

    events = engine._complete_drain(instance)

    assert len(events) == 1
    # max(10.0, 0.0 + 5.0) = 10.0 — drain already exceeded min blocking
    assert events[0].timestamp == 10.0
    assert instance.drain_duration == 10.0

    # Case 2: drain (2s) < min_blocking (5s) → unblock at initiated + min_blocking
    instance.draining = True
    instance.switch_initiated_time = 100.0
    instance.blocked_until = 0.0
    engine.current_time = 102.0  # drain took 2 seconds

    events = engine._complete_drain(instance)

    assert len(events) == 1
    # max(102.0, 100.0 + 5.0) = 105.0 — must wait for min blocking
    assert events[0].timestamp == 105.0
    assert instance.drain_duration == 2.0


def test_full_switch_cycle(long_trace_file):
    """Test a complete switch cycle from trigger to completion."""
    config = SimConfig(
        trace_path=long_trace_file,
        num_prefill_instances=2,
        num_decode_instances=2,
        switch_min_blocking_time=0.1,  # Short blocking time so unblock happens quickly
        switch_schedule=[
            # Switch early in simulation
            {"time": 0.05, "instance_id": "prefill_0", "target_role": "DECODE"},
        ]
    )

    engine = SimulationEngine(config, enable_iteration_logging=False)

    # Run simulation
    results = engine.run()

    # Check that switch was recorded
    assert results.num_switches >= 1
    assert len(results.role_switches) >= 1

    # Check switch details
    switch = results.role_switches[0]
    assert switch["instance_id"] == "prefill_0"
    assert switch["from_role"] == "PREFILL"
    assert switch["to_role"] == "DECODE"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
