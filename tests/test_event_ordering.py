"""Test event ordering and simulation engine basics."""

import pytest
import heapq
from core.event import Event, EventType


class TestEventOrdering:
    """Test event ordering in priority queue."""

    def test_events_ordered_by_timestamp(self):
        """Test that events are ordered chronologically."""
        events = []

        heapq.heappush(events, Event(5.0, EventType.REQUEST_ARRIVAL, data={}))
        heapq.heappush(events, Event(2.0, EventType.PREFILL_BATCH_START, data={}))
        heapq.heappush(events, Event(8.0, EventType.REQUEST_COMPLETE, data={}))
        heapq.heappush(events, Event(1.0, EventType.REQUEST_ARRIVAL, data={}))

        # Pop in order
        e1 = heapq.heappop(events)
        e2 = heapq.heappop(events)
        e3 = heapq.heappop(events)
        e4 = heapq.heappop(events)

        assert e1.timestamp == 1.0
        assert e2.timestamp == 2.0
        assert e3.timestamp == 5.0
        assert e4.timestamp == 8.0

    def test_events_with_same_timestamp(self):
        """Test events at same timestamp use priority."""
        events = []

        # Both at t=5.0, but different priorities
        heapq.heappush(events, Event(5.0, EventType.REQUEST_ARRIVAL, priority=1, data={}))
        heapq.heappush(events, Event(5.0, EventType.PREFILL_BATCH_START, priority=0, data={}))

        e1 = heapq.heappop(events)
        e2 = heapq.heappop(events)

        # Priority 0 should come first
        assert e1.priority == 0
        assert e2.priority == 1

    def test_event_type_enum(self):
        """Test EventType enum values."""
        # Enum values are auto-generated integers, not strings
        assert isinstance(EventType.REQUEST_ARRIVAL.value, int)
        assert isinstance(EventType.PREFILL_BATCH_START.value, int)
        assert isinstance(EventType.PREFILL_BATCH_COMPLETE.value, int)
        assert isinstance(EventType.KV_TRANSFER_START.value, int)
        assert isinstance(EventType.KV_TRANSFER_COMPLETE.value, int)
        assert isinstance(EventType.DECODE_BATCH_START.value, int)
        assert isinstance(EventType.DECODE_BATCH_COMPLETE.value, int)
        assert isinstance(EventType.REQUEST_COMPLETE.value, int)

    def test_event_data_preservation(self):
        """Test that event data is preserved."""
        data = {'request': 'req_1', 'value': 42}
        event = Event(1.0, EventType.REQUEST_ARRIVAL, data=data)

        assert event.data == data
        assert event.data['request'] == 'req_1'
        assert event.data['value'] == 42
