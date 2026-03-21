"""Simulation engine for discrete-event simulation."""

from __future__ import annotations

import heapq
import logging
import random
from typing import List, Tuple

from config import SimConfig
from core.event import Event, EventType
from instances.base_instance import InstanceType
from instances.decode_instance import DecodeInstance
from instances.instance_manager import InstanceManager
from instances.prefill_instance import PrefillInstance
from metrics.metrics_collector import MetricsCollector, SimulationResults
from metrics.time_series_monitor import TimeSeriesMonitor
from policy.policy_controller import PolicyController
from mechanisms.policy_monitor import PolicyMonitor
from request.batch import ForwardMode, SimBatch
from request.request import RequestStage, SimReq
from results.iteration_logger import IterationLogger
from results.request_trace_logger import RequestTraceLogger
from scheduling.prefill_scheduler import PrefillScheduler
from scheduling.simple_scheduler import SimpleScheduler
from utils.profiling_formulas import (
    compute_inference_time,
    compute_kv_transfer_time,
)
from workload.trace_loader import WorkloadDriver, StreamingWorkloadDriver
from utils.memory_profiler import MemoryProfiler


class SimulationEngine:
    """Main discrete-event simulation engine."""

    def __init__(self, config: SimConfig, enable_iteration_logging: bool = True):
        """Initialize simulation engine.

        Args:
            config: Simulation configuration
            enable_iteration_logging: Whether to log per-iteration statistics
        """
        self.config = config
        self.current_time = 0.0
        self.event_queue: List[Event] = []
        random.seed(42)  # Deterministic batch duration noise

        # Initialize logger
        self.logger = logging.getLogger(__name__)

        # Memory profiler (disabled by default, enable via config)
        self.memory_profiler = MemoryProfiler(
            enabled=getattr(config, 'enable_memory_profiling', False),
            interval=getattr(config, 'memory_profiling_interval', 60.0),
        )
        self.memory_profiler.start()

        # Components - use streaming loader for large traces
        if config.enable_streaming_loading:
            self.workload_driver = StreamingWorkloadDriver(
                config.trace_path,
                window_size=config.streaming_window_size,
                lookback=config.streaming_lookback,
            )
            self.streaming_enabled = True
        else:
            self.workload_driver = WorkloadDriver(config.trace_path)
            self.streaming_enabled = False

        self.instance_manager = InstanceManager(
            num_prefill=config.num_prefill_instances,
            num_decode=config.num_decode_instances,
        )
        self.metrics_collector = MetricsCollector()
        self.policy = PolicyController(config)

        # Policy monitor for role switching
        self.policy_monitor = PolicyMonitor(
            policy=config.switch_policy if config.enable_switching else "never",
            monitor_interval_s=config.monitor_interval_s,
            # Alpha params
            alpha_threshold=config.alpha_threshold,
            alpha_threshold_down=config.alpha_threshold_down,
            # V1 params
            prefill_high=config.prefill_pressure_high,
            prefill_low=config.prefill_pressure_low,
            decode_high=config.decode_pressure_high,
            decode_low=config.decode_pressure_low,
            prefill_wait_weight=config.prefill_wait_weight,
            prefill_active_weight=config.prefill_active_weight,
            prefill_inflight_weight=config.prefill_inflight_weight,
            decode_prealloc_weight=config.decode_prealloc_weight,
            decode_transfer_weight=config.decode_transfer_weight,
            decode_active_weight=config.decode_active_weight,
            decode_prefill_prealloc_weight=config.decode_prefill_prealloc_weight,
            # Common params
            stable_evals=config.stable_evals,
            global_cooldown_s=config.global_cooldown_s,
            per_worker_cooldown_s=config.per_worker_cooldown_s,
            min_prefill=config.min_prefill_instances,
            min_decode=config.min_decode_instances,
            idle_scrapes=config.idle_scrapes,
            slo_target=config.slo_target,
            # Throughput policy params
            max_prefill_tokens=config.max_prefill_tokens,
            max_running_requests=config.max_running_requests,
        )

        # Throughput tracking for throughput-based policy
        self._interval_prefill_tokens = 0
        self._interval_decode_tokens = 0
        self._interval_decode_computed_token_sum = 0
        self._interval_decode_batch_count = 0
        self._interval_start_time = 0.0

        # Iteration logger (optional)
        self.iteration_logger = None
        self.request_trace_logger = None
        self.time_series_monitor = None
        self.iteration_count = 0
        if enable_iteration_logging:
            self.iteration_logger = IterationLogger(
                config=config,
                num_prefill=config.num_prefill_instances,
                num_decode=config.num_decode_instances,
            )
            self.request_trace_logger = RequestTraceLogger(
                self.iteration_logger.output_dir
            )
            # Initialize time-series monitor with same output directory
            if getattr(config, 'enable_monitoring', True):
                self.time_series_monitor = TimeSeriesMonitor(
                    sample_interval=getattr(config, 'monitoring_sample_interval', 1.0),
                    max_samples=getattr(config, 'monitoring_max_samples', 10000),
                    sla_window_size=getattr(config, 'monitoring_sla_window', 1000),
                    ttft_sla=getattr(config, 'ttft_sla', 1.0),
                    itl_sla=getattr(config, 'itl_sla', 0.1),
                    output_dir="result",
                    run_name=self.iteration_logger.run_name,
                    enable_periodic_plots=getattr(config, 'enable_periodic_plots', True),
                    plot_interval_minutes=getattr(config, 'monitoring_plot_interval_minutes', 15.0),
                )

        # Resolve chunked prefill size
        chunked_prefill_size = None
        if config.chunked_prefill_size > 0:
            chunked_prefill_size = config.chunked_prefill_size

        # Schedulers (one per instance)
        self.prefill_schedulers = {
            inst.instance_id: PrefillScheduler(
                inst,
                max_prefill_tokens=config.max_prefill_tokens,
                chunked_prefill_size=chunked_prefill_size,
                new_token_ratio=config.schedule_conservativeness,
                schedule_policy=config.schedule_policy,
                enable_mixed_chunk=config.enable_mixed_chunk,
                max_running_requests=config.max_running_requests,
            )
            for inst in self.instance_manager.prefill_instances
        }
        self.decode_schedulers = {
            inst.instance_id: SimpleScheduler(inst, config.max_prefill_tokens)
            for inst in self.instance_manager.decode_instances
        }

        # Counter for round-robin request routing
        self._arrival_counter = 0

        # Load switch schedule if configured
        if config.switch_schedule:
            self._load_switch_schedule(config.switch_schedule)

        # Schedule first monitor evaluation if policy-driven switching enabled
        if self.policy_monitor.enabled:
            event = Event(
                timestamp=config.monitor_interval_s,
                event_type=EventType.MONITOR_EVAL,
                data={},
            )
            heapq.heappush(self.event_queue, event)

    def run(self) -> SimulationResults:
        """Run simulation until ALL requests complete.

        Returns:
            Simulation results
        """
        info_parts = []
        if self.config.chunked_prefill_size > 0:
            info_parts.append(f"chunked={self.config.chunked_prefill_size}")
        if self.policy.offloading_enabled:
            info_parts.append(f"dynamic_lp(slo={self.config.slo_target})")
        if self.policy.continuation_enabled:
            info_parts.append(f"M={self.config.M}")
        if self.policy.switching_enabled:
            info_parts.append(f"switch={self.config.switch_policy}")
        extra = f" ({', '.join(info_parts)})" if info_parts else ""
        print(
            f"Starting simulation with {self.config.num_prefill_instances}P"
            f"{self.config.num_decode_instances}D{extra}"
        )

        # Load trace and initialize events
        requests = self._initialize_workload()
        if self.streaming_enabled:
            total_requests = None  # Unknown in streaming mode
            trace_exhausted = False
        else:
            total_requests = len(requests)
            trace_exhausted = True  # All loaded upfront

        # Start periodic plot generation (wall-clock timer thread)
        if self.time_series_monitor:
            self.time_series_monitor.start_periodic_plotting()

        # Main event loop
        while self.event_queue:
            event = heapq.heappop(self.event_queue)
            self.current_time = event.timestamp

            # Streaming mode: load next window if needed
            if self.streaming_enabled and not trace_exhausted:
                if self.workload_driver.should_load_next_window(self.current_time):
                    print(f"[Streaming @ t={self.current_time:.1f}s] Loading next window...")
                    next_requests = self.workload_driver.load_next_window()

                    # Schedule new arrival events
                    for req in next_requests:
                        new_event = Event(
                            timestamp=req.arrival_time,
                            event_type=EventType.REQUEST_ARRIVAL,
                            data={"request": req},
                        )
                        heapq.heappush(self.event_queue, new_event)

                    # Check if trace is exhausted
                    if self.workload_driver.trace_exhausted:
                        trace_exhausted = True
                        total_requests = self.workload_driver.total_requests_loaded
                        print(f"[Streaming] Trace exhausted. Total requests: {total_requests}")

            new_events = self._dispatch_event(event)

            for new_event in new_events:
                heapq.heappush(self.event_queue, new_event)

            # Memory profiling (periodic snapshots)
            self.memory_profiler.snapshot(
                self.current_time,
                event_queue=self.event_queue,
                metrics=self.metrics_collector,
            )

            # Time-series monitoring (periodic sampling)
            if self.time_series_monitor:
                engine_state = {
                    'all_instances': (self.instance_manager.prefill_instances +
                                     self.instance_manager.decode_instances),
                    'metrics_collector': self.metrics_collector,
                }
                self.time_series_monitor.maybe_sample(self.current_time, engine_state)

            # Check bootstrap queue timeouts (SGLang-style)
            for prefill_inst in self.instance_manager.prefill_instances:
                self._abort_bootstrap_on_timeout(prefill_inst)

            # Check completion condition
            completed = (self.metrics_collector.num_completions
                        if self.metrics_collector.enable_streaming
                        else len(self.metrics_collector.completions))

            # In streaming mode, only break when trace is exhausted AND all completed
            if total_requests is not None and completed >= total_requests:
                print(f"All {total_requests} requests completed!")
                break

        # Get final completion count
        completed = (self.metrics_collector.num_completions
                    if self.metrics_collector.enable_streaming
                    else len(self.metrics_collector.completions))

        # Close streaming loader if used
        if self.streaming_enabled:
            self.workload_driver.close()
            if total_requests is None:
                total_requests = self.workload_driver.total_requests_loaded
            print(f"[Streaming] Closed trace loader. Total loaded: {total_requests}")

        completion_rate = (
            (completed / total_requests * 100) if total_requests and total_requests > 0 else 0.0
        )

        print(f"Simulation completed at t={self.current_time:.2f}s")
        print(
            f"Completed {completed}/{total_requests} requests ({completion_rate:.1f}%)"
        )

        if total_requests and completed < total_requests:
            stuck = total_requests - completed
            print(
                f"WARNING: {stuck} requests did not complete (insufficient resources)"
            )

        if self.iteration_logger:
            self.iteration_logger.finalize()
        if self.request_trace_logger:
            self.request_trace_logger.finalize()

        # Generate time-series plots
        if self.time_series_monitor:
            self.time_series_monitor.finalize()
            self.time_series_monitor.generate_plots()

        # Generate memory profiling report
        if self.memory_profiler.enabled:
            self.memory_profiler.report(output_path="memory_profile.csv")
            self.memory_profiler.stop()

        return self.metrics_collector.finalize(
            total_time=self.current_time,
            num_prefill=self.config.num_prefill_instances,
            num_decode=self.config.num_decode_instances,
        )

    def _initialize_workload(self):
        """Load workload trace and schedule arrival events."""
        if self.streaming_enabled:
            # Streaming mode: load only first window
            self.workload_driver.open()
            requests = self.workload_driver.load_initial_window()
            total_requests = -1  # Unknown total, will be determined at end

            # Schedule initial window events
            for req in requests:
                event = Event(
                    timestamp=req.arrival_time,
                    event_type=EventType.REQUEST_ARRIVAL,
                    data={"request": req},
                )
                heapq.heappush(self.event_queue, event)

            print(f"[Streaming] Loaded initial window: {len(requests)} requests")
            # Return empty list to signal unknown total
            return []
        else:
            # Traditional mode: load all requests at once
            requests = self.workload_driver.load_trace()

            for req in requests:
                event = Event(
                    timestamp=req.arrival_time,
                    event_type=EventType.REQUEST_ARRIVAL,
                    data={"request": req},
                )
                heapq.heappush(self.event_queue, event)

            return requests

    def _load_switch_schedule(self, schedule: List[dict]):
        """Load role switch schedule and create ROLE_SWITCH events.

        Args:
            schedule: List of switch specifications, each with:
                - time: When to trigger the switch
                - instance_id: Which instance to switch
                - target_role: Target role ("PREFILL" or "DECODE")
        """
        for switch_spec in schedule:
            event = Event(
                timestamp=switch_spec["time"],
                event_type=EventType.ROLE_SWITCH,
                data={
                    "instance_id": switch_spec["instance_id"],
                    "target_role": InstanceType[switch_spec["target_role"]]
                }
            )
            heapq.heappush(self.event_queue, event)

    def _dispatch_event(self, event: Event) -> List[Event]:
        """Dispatch event to appropriate handler."""
        handlers = {
            EventType.REQUEST_ARRIVAL: self._handle_request_arrival,
            EventType.PREFILL_BATCH_START: self._handle_prefill_batch_start,
            EventType.PREFILL_BATCH_COMPLETE: self._handle_prefill_batch_complete,
            EventType.KV_TRANSFER_START: self._handle_kv_transfer_start,
            EventType.KV_TRANSFER_COMPLETE: self._handle_kv_transfer_complete,
            EventType.DECODE_BATCH_START: self._handle_decode_batch_start,
            EventType.DECODE_BATCH_COMPLETE: self._handle_decode_batch_complete,
            EventType.REQUEST_COMPLETE: self._handle_request_complete,
            EventType.PREFILL_DECODE_START: self._handle_prefill_decode_start,
            EventType.PREFILL_DECODE_COMPLETE: self._handle_prefill_decode_complete,
            EventType.OFFLOAD_PREFILL_START: self._handle_offload_prefill_start,
            EventType.OFFLOAD_PREFILL_COMPLETE: self._handle_offload_prefill_complete,
            EventType.ROLE_SWITCH: self._handle_role_switch,
            EventType.SWITCH_UNBLOCK: self._handle_switch_unblock,
            EventType.MONITOR_EVAL: self._handle_monitor_eval,
        }

        handler = handlers.get(event.event_type)
        if handler:
            return handler(event)
        return []

    # ---- BOOTSTRAP ADMISSION ----

    def _decode_allocatable_tokens(self, decode_instance: DecodeInstance) -> int:
        """Compute allocatable tokens on decode instance for bootstrap admission.

        Mirrors real SGLang's _allocatable_tokens() in decode.py.
        Accounts for physical available tokens, virtual reservations for
        bootstrapped requests, and per-request decode token reserves.

        Improved (SGLang-style): Only reserve for ACTIVE requests (running/transfer),
        not for requests already waiting for memory (waiting/prealloc queues).
        """
        available = decode_instance.token_to_kv_pool.available_size()

        # Subtract virtual reservations (KV tokens for prealloc_reserved requests)
        for req in decode_instance.prealloc_reserved:
            available -= req.context_tokens + len(req.output_ids)

        # Count ONLY active in-flight requests (SGLang pattern)
        # Don't count waiting/prealloc - they're already blocked waiting for memory
        num_running = len(decode_instance.running_batch.reqs)
        num_transfer = len(decode_instance.transfer_queue)

        num_active_inflight = num_running + num_transfer

        # Reserve decode tokens only for active requests
        # This prevents over-reservation and allows more bootstrap admissions
        reserved_tokens = self.config.num_reserved_decode_tokens * num_active_inflight

        return max(0, available - reserved_tokens)

    def _select_best_decode_instance(self):
        """Select decode instance with most allocatable tokens (least-loaded).

        Skips instances that are not accepting requests (e.g. draining during
        role switch) to prevent routing new work to switching instances.

        Returns:
            Best decode instance, or None if all are draining.
        """
        best = None
        best_tokens = -float("inf")
        for inst in self.instance_manager.decode_instances:
            if not inst.accepting_requests:
                continue
            tokens = self._decode_allocatable_tokens(inst)
            if tokens > best_tokens:
                best_tokens = tokens
                best = inst
        return best

    def _try_admit_from_bootstrap(self, prefill_instance: PrefillInstance) -> List[Event]:
        """Try to admit requests from bootstrap_queue to waiting_queue.

        SGLang-style skip-and-continue pattern: if a request doesn't fit,
        skip it and try the next one. This prevents head-of-line blocking.
        Skipped requests are moved to the end of the queue for fair retry.
        """
        # Don't admit if draining
        if not prefill_instance.accepting_requests:
            return []

        admitted_any = False
        requests_to_requeue = []  # Requests that don't fit now, retry later

        # Process all requests in current bootstrap queue
        for req in list(prefill_instance.bootstrap_queue):
            prefill_instance.bootstrap_queue.remove(req)

            # Select decode instance (least-loaded by allocatable tokens)
            if req.assigned_decode_instance is not None:
                decode_instance = next(
                    (inst for inst in self.instance_manager.decode_instances
                     if inst.instance_id == req.assigned_decode_instance
                     and inst.accepting_requests),
                    None,
                )
                if decode_instance is None:
                    # Pinned instance switched roles or draining; fall back
                    decode_instance = self._select_best_decode_instance()
            else:
                decode_instance = self._select_best_decode_instance()

            # No available decode instance (all draining) - requeue all remaining
            if decode_instance is None:
                requests_to_requeue.append(req)
                continue

            # Check if request is oversized (permanent failure)
            max_possible = max(
                decode_inst.token_to_kv_pool.total_kv_tokens
                for decode_inst in self.instance_manager.decode_instances
            )
            if req.context_tokens > max_possible:
                # Drop permanently - request can NEVER fit
                self.metrics_collector.record_dropped_request(req, "oversized")
                self.logger.warning(
                    f"Dropping oversized request {req.rid}: "
                    f"{req.context_tokens} > {max_possible} tokens"
                )
                continue

            # Check decode capacity
            allocatable = self._decode_allocatable_tokens(decode_instance)
            tokens_needed = req.context_tokens + self.config.num_reserved_decode_tokens

            if tokens_needed <= allocatable:
                # Admit: move to waiting queue, add virtual reservation
                decode_instance.prealloc_reserved.append(req)
                req.assigned_decode_instance = decode_instance.instance_id
                prefill_instance.add_request(req)  # adds to waiting_queue
                req.bootstrap_exit_time = self.current_time
                admitted_any = True
            else:
                # Can't fit now - SKIP to next request (SGLang pattern)
                # Will be requeued to end for fair retry
                requests_to_requeue.append(req)

        # Requeue skipped requests to END of queue (fairness)
        prefill_instance.bootstrap_queue.extend(requests_to_requeue)

        # If we admitted anything, try to schedule prefill
        if admitted_any:
            return self._try_schedule_prefill(prefill_instance)
        return []

    def _abort_bootstrap_on_timeout(self, prefill_instance: PrefillInstance) -> None:
        """Abort requests that have been stuck in bootstrap queue too long.

        SGLang-style timeout check: similar to _abort_on_waiting_timeout().
        Uses lazy initialization of _bootstrap_entry_time attribute.
        """
        timeout_seconds = self.config.bootstrap_timeout_seconds
        if timeout_seconds <= 0:
            return  # Timeout disabled

        for req in list(prefill_instance.bootstrap_queue):
            # Lazy init: track first time in bootstrap queue
            if not hasattr(req, '_bootstrap_entry_time'):
                req._bootstrap_entry_time = self.current_time

            elapsed = self.current_time - req._bootstrap_entry_time
            if elapsed > timeout_seconds:
                # Timeout exceeded - abort request silently
                prefill_instance.bootstrap_queue.remove(req)
                self.metrics_collector.record_dropped_request(req, "bootstrap_timeout")

    # ---- REQUEST ARRIVAL ----

    def _handle_request_arrival(self, event: Event) -> List[Event]:
        """Handle REQUEST_ARRIVAL event."""
        req: SimReq = event.data["request"]

        # Select prefill instance that is accepting requests
        candidates = [
            pi for pi in self.instance_manager.prefill_instances
            if pi.accepting_requests
        ]

        if not candidates:
            # All instances draining - defer request
            print(f"  WARNING: All prefill instances draining, deferring req {req.rid}")
            return [Event(
                timestamp=self.current_time + 0.1,  # Retry after 0.1s
                event_type=EventType.REQUEST_ARRIVAL,
                data={"request": req}
            )]

        # Round-robin among accepting instances
        prefill_instance = candidates[self._arrival_counter % len(candidates)]
        self._arrival_counter += 1

        req.queue_entry_time = self.current_time
        req.assigned_prefill_instance = prefill_instance.instance_id

        self.metrics_collector.record_arrival(req, self.current_time)

        events = []

        # Note: partial offloading betas are now determined dynamically by the
        # LP solver at scheduling time (_try_schedule_prefill), not at arrival.

        # Add to bootstrap queue (gated by decode capacity)
        prefill_instance.bootstrap_queue.append(req)
        events.extend(self._try_admit_from_bootstrap(prefill_instance))

        return events

    # ---- PREFILL ----

    def _handle_prefill_batch_start(self, event: Event) -> List[Event]:
        """Handle PREFILL_BATCH_START event."""
        batch = event.data["batch"]
        instance: PrefillInstance = event.data["instance"]

        instance.busy = True
        instance.busy_until = self.current_time + batch.estimated_duration
        instance.current_batch = batch

        for req in batch.reqs:
            if req.prefill_start_time == 0.0:
                req.prefill_start_time = self.current_time

        return [
            Event(
                timestamp=self.current_time + batch.estimated_duration,
                event_type=EventType.PREFILL_BATCH_COMPLETE,
                data={"batch": batch, "instance": instance},
            )
        ]

    def _handle_prefill_batch_complete(self, event: Event) -> List[Event]:
        """Handle PREFILL_BATCH_COMPLETE event."""
        batch = event.data["batch"]
        instance: PrefillInstance = event.data["instance"]

        if self.iteration_logger:
            batch_start_time = self.current_time - batch.estimated_duration
            self.iteration_logger.log_batch_execution(
                batch=batch,
                instance_id=instance.instance_id,
                instance_type="prefill",
                start_time=batch_start_time,
                end_time=self.current_time,
                iteration=self.iteration_count,
            )
            self.iteration_count += 1

        instance.busy = False
        instance.current_batch = None

        self.metrics_collector.record_prefill_busy(batch.estimated_duration)

        # Accumulate prefill tokens for throughput tracking
        self._interval_prefill_tokens += batch.total_prefill_tokens
        if self.time_series_monitor:
            self.time_series_monitor.record_throughput(
                prefill_tokens=batch.total_prefill_tokens,
                decode_tokens=0
            )

        # Check which requests are still being chunked
        scheduler = self.prefill_schedulers[instance.instance_id]
        chunked_req = scheduler.chunked_req

        events = []
        for req in batch.reqs:
            # If still being chunked, cache partial progress and free memory
            if req is chunked_req:
                req.prefill_tokens_done += req.extend_input_len

                # Cache the partially-computed KV in radix tree before freeing
                # Mirrors tree_cache.cache_unfinished_req(chunked_req, chunked=True)
                full_token_ids = req.origin_input_ids + req.output_ids
                if full_token_ids and req.kv_cache_indices:
                    instance.tree_cache.cache_unfinished_req(
                        token_ids=full_token_ids,
                        kv_indices=req.kv_cache_indices,
                        num_tokens_done=req.prefill_tokens_done,
                    )

                instance.free_memory(req)
                continue

            # Prefill is complete for this request
            req.prefill_end_time = self.current_time
            req.prefill_tokens_done = 0

            # Cache prefilled tokens in radix tree
            full_token_ids = req.origin_input_ids + req.output_ids
            if full_token_ids and req.kv_cache_indices:
                instance.tree_cache.insert(
                    token_ids=full_token_ids, kv_indices=req.kv_cache_indices
                )

            # Check decode continuation: generate M tokens on prefill instance
            continuation_tokens = self.policy.tokens_to_continue_on_prefill(req)

            if continuation_tokens > 0:
                req.continued_on_prefill = continuation_tokens
                events.extend(
                    self._schedule_prefill_continuation(
                        req, instance, continuation_tokens
                    )
                )
            else:
                # Normal path: free prefill memory, then transfer to decode
                # In P/D disaggregation, prefill instance frees KV after
                # sending it to decode — the KV lives in radix cache for
                # future prefix matching but the pool tokens are reclaimed.
                instance.free_memory(req)
                req.stage = RequestStage.PREFILL_TRANSFER_KV_CACHE
                events.extend(self._schedule_kv_transfer(req))

        # Reset batch_is_full flag (mirrors disagg prefill event loop reset)
        scheduler.batch_is_full = False

        # Try admitting from bootstrap queue (may have pending requests)
        if instance.bootstrap_queue:
            events.extend(self._try_admit_from_bootstrap(instance))

        # Try to schedule next prefill batch
        events.extend(self._try_schedule_prefill(instance))

        # Check if draining and now empty (and haven't already scheduled unblock)
        if instance.draining and not instance.has_running_requests() and instance.blocked_until == 0.0:
            events.extend(self._complete_drain(instance))

        return events

    # ---- DECODE CONTINUATION ON PREFILL ----

    def _schedule_prefill_continuation(
        self, req: SimReq, instance: PrefillInstance, num_tokens: int
    ) -> List[Event]:
        """Schedule decode continuation on prefill instance."""
        decode_time = compute_inference_time(
            total_prefill_tokens=0,
            decode_batch_size=1,
            decode_computed_token_sum=req.context_tokens + len(req.output_ids),
        )

        return [
            Event(
                timestamp=self.current_time,
                event_type=EventType.PREFILL_DECODE_START,
                data={
                    "request": req,
                    "instance": instance,
                    "tokens_remaining": num_tokens,
                    "decode_time": decode_time,
                },
            )
        ]

    def _handle_prefill_decode_start(self, event: Event) -> List[Event]:
        """Handle decode iteration starting on prefill instance."""
        req: SimReq = event.data["request"]
        instance: PrefillInstance = event.data["instance"]
        tokens_remaining: int = event.data["tokens_remaining"]
        decode_time: float = event.data["decode_time"]

        if req.decode_start_time == 0.0:
            req.decode_start_time = self.current_time
            req.continuation_start_time = self.current_time

        return [
            Event(
                timestamp=self.current_time + decode_time,
                event_type=EventType.PREFILL_DECODE_COMPLETE,
                data={
                    "request": req,
                    "instance": instance,
                    "tokens_remaining": tokens_remaining,
                },
            )
        ]

    def _handle_prefill_decode_complete(self, event: Event) -> List[Event]:
        """Handle decode iteration completing on prefill instance."""
        req: SimReq = event.data["request"]
        instance: PrefillInstance = event.data["instance"]
        tokens_remaining: int = event.data["tokens_remaining"]

        # Generate 1 token
        req.output_ids.append(req.decode_tokens_generated)
        req.decode_tokens_generated += 1
        tokens_remaining -= 1

        # Check if request is fully done
        if req.decode_tokens_generated >= req.generated_tokens:
            req.decode_end_time = self.current_time
            req.continuation_end_time = self.current_time
            instance.free_memory(req)
            return [
                Event(
                    timestamp=self.current_time,
                    event_type=EventType.REQUEST_COMPLETE,
                    data={"request": req},
                )
            ]

        # Check if continuation is done
        if tokens_remaining <= 0:
            req.continuation_end_time = self.current_time
            # Free prefill memory before transferring to decode
            instance.free_memory(req)
            req.stage = RequestStage.PREFILL_TRANSFER_KV_CACHE
            events = self._schedule_kv_transfer(req)
            # Try scheduling next prefill batch (memory freed)
            events.extend(self._try_schedule_prefill(instance))
            return events

        # Continue generating on prefill instance
        decode_time = compute_inference_time(
            total_prefill_tokens=0,
            decode_batch_size=1,
            decode_computed_token_sum=req.context_tokens + len(req.output_ids),
        )

        return [
            Event(
                timestamp=self.current_time,
                event_type=EventType.PREFILL_DECODE_START,
                data={
                    "request": req,
                    "instance": instance,
                    "tokens_remaining": tokens_remaining,
                    "decode_time": decode_time,
                },
            )
        ]

    # ---- PARTIAL OFFLOAD ----

    def _handle_offload_prefill_start(self, event: Event) -> List[Event]:
        """Handle offloaded prefill portion starting on decode instance."""
        req: SimReq = event.data["request"]
        offload_duration: float = event.data["offload_duration"]

        return [
            Event(
                timestamp=self.current_time + offload_duration,
                event_type=EventType.OFFLOAD_PREFILL_COMPLETE,
                data=event.data,
            )
        ]

    def _handle_offload_prefill_complete(self, event: Event) -> List[Event]:
        """Handle offloaded β portion completing on decode instance.

        Sequential flow: prefill (1-β) → KV transfer → offload β → here.
        All KV is now on decode instance. Admit to decode.
        """
        req: SimReq = event.data["request"]
        decode_instance: DecodeInstance = event.data["decode_instance"]

        return self._admit_to_decode(req, decode_instance)

    # ---- KV TRANSFER ----

    def _schedule_kv_transfer(self, req: SimReq) -> List[Event]:
        """Schedule KV transfer for a completed prefill request.

        For offloaded requests (partial_offload_amount > 0):
        - Only transfer the (1-β) portion (offload KV is already on decode)
        - Use the pinned decode instance (assigned at arrival time)
        """
        # Determine which decode instance to use
        if req.assigned_decode_instance is not None:
            # Offloaded request: try the pinned decode instance
            decode_instance = next(
                (inst for inst in self.instance_manager.decode_instances
                 if inst.instance_id == req.assigned_decode_instance),
                None,
            )
            if decode_instance is None:
                # Pinned instance switched roles; fall back to round-robin
                decode_instance = self.instance_manager.select_decode_instance()
        else:
            decode_instance = self.instance_manager.select_decode_instance()

        # Only transfer the non-offloaded portion
        transfer_tokens = req.context_tokens - req.partial_offload_amount
        transfer_time = compute_kv_transfer_time(transfer_tokens)

        return [
            Event(
                timestamp=self.current_time,
                event_type=EventType.KV_TRANSFER_START,
                data={
                    "request": req,
                    "decode_instance": decode_instance,
                    "transfer_duration": transfer_time,
                },
            )
        ]

    def _handle_kv_transfer_start(self, event: Event) -> List[Event]:
        """Handle KV_TRANSFER_START event."""
        req: SimReq = event.data["request"]
        decode_instance: DecodeInstance = event.data["decode_instance"]
        transfer_duration: float = event.data["transfer_duration"]

        decode_instance.transfer_queue.append(req)
        req.kv_transfer_start_time = self.current_time

        return [
            Event(
                timestamp=self.current_time + transfer_duration,
                event_type=EventType.KV_TRANSFER_COMPLETE,
                data={"request": req, "decode_instance": decode_instance},
            )
        ]

    def _handle_kv_transfer_complete(self, event: Event) -> List[Event]:
        """Handle KV_TRANSFER_COMPLETE event.

        For both offloaded and normal requests: admit to decode.
        Offloaded requests carry partial_offload_amount > 0, and the β portion
        will be computed as part of a MIXED decode batch on the decode instance.

        If the target decode instance switched roles during the transfer,
        re-route the request to an active decode instance.
        """
        req: SimReq = event.data["request"]
        decode_instance: DecodeInstance = event.data["decode_instance"]

        # Safely remove from transfer_queue (may already be cleared during role switch)
        if req in decode_instance.transfer_queue:
            decode_instance.transfer_queue.remove(req)
        req.kv_transfer_end_time = self.current_time

        # If instance switched roles during transfer, re-route to active decode
        if decode_instance.instance_type != InstanceType.DECODE:
            new_decode = self.instance_manager.select_least_loaded_decode()
            if new_decode is None:
                print(f"  WARNING: No decode instance for req {req.rid} after role switch")
                return []
            # Release virtual reservation from old instance if still present
            if req in decode_instance.prealloc_reserved:
                decode_instance.prealloc_reserved.remove(req)
            return self._admit_to_decode(req, new_decode)

        return self._admit_to_decode(req, decode_instance)

    def _admit_to_decode(
        self, req: SimReq, decode_instance: DecodeInstance
    ) -> List[Event]:
        """Admit a request to decode after KV transfer (and offload if any) complete.

        Releases virtual reservation from bootstrap admission, then physically
        allocates KV cache on decode side and adds to waiting queue.
        """
        # Release virtual reservation (request has arrived physically)
        if req in decode_instance.prealloc_reserved:
            decode_instance.prealloc_reserved.remove(req)

        total_tokens = req.context_tokens + len(req.output_ids)
        req.extend_input_len = total_tokens
        if not decode_instance.allocate_memory(req):
            # Try eviction on decode side
            decode_instance.tree_cache.evict(
                total_tokens,
                evict_callback=lambda kv_indices: decode_instance.token_to_kv_pool.free(
                    kv_indices
                ),
            )
            if not decode_instance.allocate_memory(req):
                # Still can't allocate - add to prealloc queue for retry later
                decode_instance.prealloc_queue.append(req)
                req.assigned_decode_instance = decode_instance.instance_id
                # Virtual reservation freed — try admitting more from bootstrap
                events = []
                for pi in self.instance_manager.prefill_instances:
                    if pi.bootstrap_queue:
                        events.extend(self._try_admit_from_bootstrap(pi))
                return events

        # Set extend_input_len for decode-side batch computation:
        # Offloaded: needs prefill of β*C tokens in next MIXED decode iteration
        # Normal: KV fully transferred, no prefill needed on decode
        if req.partial_offload_amount > 0:
            req.extend_input_len = req.partial_offload_amount
        else:
            req.extend_input_len = 0

        decode_instance.waiting_queue.append(req)
        req.stage = RequestStage.DECODE_WAITING
        req.assigned_decode_instance = decode_instance.instance_id
        req.decode_queue_entry_time = self.current_time

        return self._try_schedule_decode(decode_instance)

    # ---- DECODE ----

    def _handle_decode_batch_start(self, event: Event) -> List[Event]:
        """Handle DECODE_BATCH_START event."""
        batch = event.data["batch"]
        instance: DecodeInstance = event.data["instance"]

        instance.busy = True
        instance.busy_until = self.current_time + batch.estimated_duration
        instance.current_batch = batch

        for req in batch.reqs:
            if req.decode_tokens_generated == 0:
                req.decode_start_time = self.current_time
            req.stage = RequestStage.DECODE_FORWARD
            # Track offload: first MIXED batch = offload execution
            if req.partial_offload_amount > 0 and req.offload_start_time == 0.0:
                req.offload_start_time = self.current_time

        return [
            Event(
                timestamp=self.current_time + batch.estimated_duration,
                event_type=EventType.DECODE_BATCH_COMPLETE,
                data={"batch": batch, "instance": instance},
            )
        ]

    def _handle_decode_batch_complete(self, event: Event) -> List[Event]:
        """Handle DECODE_BATCH_COMPLETE event."""
        batch = event.data["batch"]
        instance: DecodeInstance = event.data["instance"]

        if self.iteration_logger:
            batch_start_time = self.current_time - batch.estimated_duration
            self.iteration_logger.log_batch_execution(
                batch=batch,
                instance_id=instance.instance_id,
                instance_type="decode",
                start_time=batch_start_time,
                end_time=self.current_time,
                iteration=self.iteration_count,
            )
            self.iteration_count += 1

        instance.busy = False
        instance.current_batch = None

        self.metrics_collector.record_decode_busy(batch.estimated_duration)

        # Accumulate decode tokens for throughput tracking
        self._interval_decode_tokens += batch.decode_batch_size
        if self.time_series_monitor:
            self.time_series_monitor.record_throughput(
                prefill_tokens=0,
                decode_tokens=batch.decode_batch_size
            )
        self._interval_decode_computed_token_sum += batch.decode_computed_token_sum
        self._interval_decode_batch_count += 1

        events = []

        for req in batch.reqs[:]:
            req.output_ids.append(req.decode_tokens_generated)
            req.decode_tokens_generated += 1

            # Track offload end: first batch after offload start
            if (req.partial_offload_amount > 0
                    and req.offload_start_time > 0
                    and req.offload_end_time == 0.0):
                req.offload_end_time = self.current_time

            if req.decode_tokens_generated >= req.generated_tokens:
                req.decode_end_time = self.current_time
                instance.running_batch.reqs.remove(req)
                events.append(
                    Event(
                        timestamp=self.current_time,
                        event_type=EventType.REQUEST_COMPLETE,
                        data={"request": req},
                    )
                )

        # Try to process any prealloc queue requests that were deferred
        # Skip during drain to avoid adding new work
        if instance.prealloc_queue and not instance.draining:
            for req in instance.prealloc_queue[:]:
                total_tokens = req.context_tokens + len(req.output_ids)
                req.extend_input_len = total_tokens
                if instance.allocate_memory(req):
                    instance.prealloc_queue.remove(req)
                    instance.waiting_queue.append(req)
                    req.stage = RequestStage.DECODE_WAITING
                    req.decode_queue_entry_time = self.current_time
                else:
                    break  # Still no memory

        events.extend(self._try_schedule_decode(instance))

        # Decode memory freed — try admitting from bootstrap queues
        for prefill_instance in self.instance_manager.prefill_instances:
            if prefill_instance.bootstrap_queue:
                events.extend(self._try_admit_from_bootstrap(prefill_instance))

        # Check if draining and now empty (and haven't already scheduled unblock)
        if instance.draining and not instance.has_running_requests() and instance.blocked_until == 0.0:
            events.extend(self._complete_drain(instance))

        return events

    # ---- REQUEST COMPLETE ----

    def _handle_request_complete(self, event: Event) -> List[Event]:
        """Handle REQUEST_COMPLETE event."""
        req: SimReq = event.data["request"]
        req.finished = True
        req.stage = RequestStage.COMPLETED
        req.completion_time = self.current_time

        # Free memory on decode instance
        decode_instance = None
        if req.assigned_decode_instance:
            decode_instance = next(
                (
                    inst
                    for inst in self.instance_manager.decode_instances
                    if inst.instance_id == req.assigned_decode_instance
                ),
                None,
            )
            if decode_instance:
                decode_instance.free_memory(req)

        self.metrics_collector.record_completion(req, self.current_time)
        if self.request_trace_logger:
            self.request_trace_logger.log_request(req)
        if self.time_series_monitor:
            self.time_series_monitor.record_completion(req, self.current_time)

        # Memory freed — try to schedule waiting requests that were blocked.
        # In real SGLang, completed requests free KV cache which allows
        # queued requests to proceed.
        events = []

        # Re-try decode scheduling (prealloc_queue may have deferred requests)
        if decode_instance and not decode_instance.busy:
            if decode_instance.prealloc_queue:
                for prealloc_req in decode_instance.prealloc_queue[:]:
                    total_tokens = prealloc_req.context_tokens + len(prealloc_req.output_ids)
                    prealloc_req.extend_input_len = total_tokens
                    if decode_instance.allocate_memory(prealloc_req):
                        decode_instance.prealloc_queue.remove(prealloc_req)
                        decode_instance.waiting_queue.append(prealloc_req)
                        prealloc_req.stage = RequestStage.DECODE_WAITING
                        prealloc_req.decode_queue_entry_time = self.current_time
                    else:
                        break
            events.extend(self._try_schedule_decode(decode_instance))

        # Decode memory freed — try admitting from bootstrap queues
        for prefill_instance in self.instance_manager.prefill_instances:
            if prefill_instance.bootstrap_queue:
                events.extend(self._try_admit_from_bootstrap(prefill_instance))

        # Re-try prefill scheduling on all idle prefill instances
        for prefill_instance in self.instance_manager.prefill_instances:
            if not prefill_instance.busy and prefill_instance.waiting_queue:
                events.extend(self._try_schedule_prefill(prefill_instance))

        return events

    # ---- SCHEDULING HELPERS ----

    def _try_schedule_prefill(self, instance: PrefillInstance) -> List[Event]:
        """Try to schedule prefill batch on instance."""
        if instance.busy:
            return []

        # Instance may have switched roles; skip if no longer a prefill instance
        if instance.instance_id not in self.prefill_schedulers:
            return []

        # Apply dynamic LP betas before batch formation
        lp_delay = 0.0
        if self.policy.dynamic_lp_enabled and instance.waiting_queue:
            lp_delay = self._apply_dynamic_lp_betas(instance)

        scheduler = self.prefill_schedulers[instance.instance_id]
        batch = scheduler.get_next_batch()

        if batch is None:
            return []

        instance.busy = True

        return [
            Event(
                timestamp=self.current_time + lp_delay,
                event_type=EventType.PREFILL_BATCH_START,
                data={"batch": batch, "instance": instance},
            )
        ]

    def _apply_dynamic_lp_betas(self, instance: PrefillInstance) -> float:
        """Apply LP-computed betas to requests in the waiting queue.

        Checks decode instance availability, runs the LP solver on eligible
        requests, and sets per-request partial_offload_amount / extend_input_len.

        Args:
            instance: Prefill instance whose waiting queue to process.

        Returns:
            Wall-clock LP solve time in seconds (for TTFT accounting).
        """
        waiting = instance.waiting_queue
        if not waiting:
            return 0.0

        max_window = self.policy.lp_solver.max_window_size
        window = waiting[:max_window]

        # Check decode instance availability for each request in the window.
        # Only requests whose decode instance can accept offloaded work are
        # eligible for the LP solver; others get beta=0.
        eligible_requests: list[SimReq] = []
        ineligible_indices: set[int] = set()

        for i, req in enumerate(window):
            decode_instance = self._get_decode_instance_for_req(req)
            if decode_instance is None or not self._decode_can_accept_offload(
                decode_instance, req
            ):
                ineligible_indices.add(i)
            else:
                eligible_requests.append(req)

        # Run LP solver on eligible requests
        solve_time = 0.0
        if eligible_requests:
            # Use the first eligible request's decode instance for state query
            first_decode = self._get_decode_instance_for_req(eligible_requests[0])
            if first_decode is not None:
                decode_bs = first_decode.running_batch.decode_batch_size
                decode_token_sum = first_decode.running_batch.decode_computed_token_sum
            else:
                decode_bs = 0
                decode_token_sum = 0

            betas, feasible, solve_time = self.policy.solve_dynamic_betas(
                eligible_requests, self.current_time, decode_bs, decode_token_sum
            )

            # Apply betas to eligible requests
            for req, beta in zip(eligible_requests, betas):
                if beta > 0.0:
                    offload_tokens = int(beta * req.context_tokens)
                    prefill_tokens = req.context_tokens - offload_tokens

                    # Ensure at least 1 token on prefill side
                    if prefill_tokens == 0 and req.context_tokens > 0:
                        prefill_tokens = 1
                        offload_tokens = req.context_tokens - 1

                    req.partial_offload_amount = offload_tokens
                    req.extend_input_len = prefill_tokens
                    req.lp_beta = beta
                else:
                    req.partial_offload_amount = 0
                    req.extend_input_len = req.context_tokens
                    req.lp_beta = 0.0

        # Ineligible requests get beta=0 (full prefill, no offload)
        for i in ineligible_indices:
            req = window[i]
            req.partial_offload_amount = 0
            req.extend_input_len = req.context_tokens
            req.lp_beta = 0.0

        return solve_time

    def _get_decode_instance_for_req(self, req: SimReq):
        """Get the decode instance assigned to a request, or None."""
        if req.assigned_decode_instance is not None:
            for inst in self.instance_manager.decode_instances:
                if inst.instance_id == req.assigned_decode_instance:
                    return inst
        return None

    def _decode_can_accept_offload(
        self, decode_instance: DecodeInstance, req: SimReq
    ) -> bool:
        """Check if a decode instance can accept offloaded prefill work.

        Returns False if decode memory is too tight to handle additional
        offloaded tokens (i.e., decode is effectively "batch full").
        """
        available = decode_instance.token_to_kv_pool.available_size()
        evictable = decode_instance.tree_cache.evictable_size

        # Account for virtual reservations and in-flight requests
        reserved = sum(
            r.context_tokens for r in decode_instance.prealloc_reserved
        )
        transfer_tokens = sum(
            r.context_tokens for r in decode_instance.transfer_queue
        )

        effective_available = available + evictable - reserved - transfer_tokens

        # Need enough space for the request's context tokens plus some headroom
        return effective_available > req.context_tokens

    def _try_schedule_decode(self, instance: DecodeInstance) -> List[Event]:
        """Try to schedule decode batch on instance.

        Handles retracted requests by adding them to prealloc_queue
        for re-admission when memory is available.
        """
        if instance.busy:
            return []

        # Instance may have switched roles; skip if no longer a decode instance
        if instance.instance_id not in self.decode_schedulers:
            return []

        scheduler = self.decode_schedulers[instance.instance_id]

        # During drain: don't admit new requests from waiting_queue,
        # but still schedule existing running_batch to completion
        if not instance.draining:
            for req in instance.waiting_queue[:]:
                instance.waiting_queue.remove(req)
                scheduler.add_to_running_batch([req])

        batch = scheduler.get_next_decode_batch()

        if batch is None:
            return []

        # Handle retracted requests: add them to prealloc_queue
        # for re-admission when memory frees up
        if scheduler.last_retracted_reqs:
            for req in scheduler.last_retracted_reqs:
                instance.prealloc_queue.append(req)
            scheduler.last_retracted_reqs = []

        instance.busy = True

        return [
            Event(
                timestamp=self.current_time,
                event_type=EventType.DECODE_BATCH_START,
                data={"batch": batch, "instance": instance},
            )
        ]

    # ---- ROLE SWITCHING ----

    def _handle_role_switch(self, event: Event) -> List[Event]:
        """Handle ROLE_SWITCH event - manually triggered switch.

        Steps:
        1. Find target instance by ID
        2. Validate switch is possible (not already draining)
        3. Initiate drain process
        """
        instance_id = event.data["instance_id"]
        target_role = event.data["target_role"]

        # Find instance
        all_instances = (
            self.instance_manager.prefill_instances +
            self.instance_manager.decode_instances
        )
        instance = next(
            (i for i in all_instances if i.instance_id == instance_id),
            None
        )

        if instance is None:
            print(f"ERROR: Instance {instance_id} not found for role switch")
            return []

        if instance.draining:
            # Silently skip if already draining
            return []

        if instance.instance_type == target_role:
            print(f"WARNING: Instance {instance_id} already {target_role.name}, skipping")
            return []

        # Initiate switch
        return self._initiate_role_switch(instance, target_role)

    def _initiate_role_switch(
        self,
        instance,
        target_role: InstanceType
    ) -> List[Event]:
        """Initiate role switch by starting drain process.

        1. Mark instance as draining, stop accepting new requests
        2. Migrate all queued requests to other instances with priority
        3. If no running requests, schedule unblock immediately
        4. Otherwise, wait for running_batch to drain naturally
        """
        print(f"[{self.current_time:.3f}s] Role switch initiated: "
              f"{instance.instance_id} {instance.instance_type.name} → {target_role.name}")

        # Update instance state
        instance.draining = True
        instance.switch_target_role = target_role
        instance.switch_initiated_time = self.current_time
        instance.accepting_requests = False

        events = []

        # Migrate all queued requests (batched to avoid cascading scheduling)
        migrated_count, migration_events = self._migrate_queued_requests(instance)
        events.extend(migration_events)
        print(f"  Migrated {migrated_count} queued requests")

        # Check if instance is already idle
        if not instance.has_running_requests():
            # Can complete drain immediately
            instance.drain_duration = 0.0
            instance.blocked_until = self.current_time + self.config.switch_min_blocking_time

            print(f"  Instance idle, blocking for {self.config.switch_min_blocking_time:.1f}s")

            events.append(Event(
                timestamp=instance.blocked_until,
                event_type=EventType.SWITCH_UNBLOCK,
                data={"instance": instance}
            ))
        else:
            # Must wait for running batch to drain
            num_running = len(instance.running_batch.reqs) if instance.running_batch else 0
            print(f"  Waiting for {num_running} running requests to drain")

        return events

    def _migrate_queued_requests(self, source_instance) -> Tuple[int, List[Event]]:
        """Migrate all queued requests from draining instance.

        Uses batched migration: moves all requests first, then triggers
        scheduling once per affected target instance. This avoids O(n^2)
        list operations and cascading LP solver calls.

        Handles instance-specific queues:
        - PrefillInstance: bootstrap_queue, waiting_queue
        - DecodeInstance: prealloc_queue, prealloc_reserved, waiting_queue

        Does NOT migrate:
        - transfer_queue (requests in flight, will complete)
        - running_batch (will drain naturally)

        Returns: (number_migrated, scheduling_events)
        """
        migrated_count = 0
        events: List[Event] = []

        if source_instance.instance_type == InstanceType.PREFILL:
            migrated_count += self._migrate_prefill_bootstrap_queue(source_instance)
            migrated_count += self._migrate_prefill_waiting_queue(source_instance)

            # Schedule once per affected prefill target (not per request)
            affected_targets: set = set()
            for inst in self.instance_manager.prefill_instances:
                if inst.instance_id != source_instance.instance_id and inst.accepting_requests:
                    if inst.bootstrap_queue or inst.waiting_queue:
                        affected_targets.add(inst.instance_id)
            for inst in self.instance_manager.prefill_instances:
                if inst.instance_id in affected_targets:
                    events.extend(self._try_admit_from_bootstrap(inst))
                    events.extend(self._try_schedule_prefill(inst))

        elif source_instance.instance_type == InstanceType.DECODE:
            migrated_count += self._migrate_decode_prealloc_reserved(source_instance)
            migrated_count += self._migrate_decode_prealloc_queue(source_instance)
            migrated_count += self._migrate_decode_waiting_queue(source_instance)

            # Schedule once per affected decode target
            for inst in self.instance_manager.decode_instances:
                if inst.instance_id != source_instance.instance_id and inst.accepting_requests:
                    if inst.waiting_queue:
                        events.extend(self._try_schedule_decode(inst))

        return migrated_count, events

    def _migrate_prefill_bootstrap_queue(self, source: PrefillInstance) -> int:
        """Migrate bootstrap queue requests with priority (batched)."""
        if not source.bootstrap_queue:
            return 0

        # Take all requests from source at once
        requests = source.bootstrap_queue[:]
        source.bootstrap_queue.clear()

        count = 0
        unmigrated = []
        for req in requests:
            target = self.instance_manager.select_instance_for_migrated(
                InstanceType.PREFILL, source.instance_id
            )
            if target is None:
                print(f"  WARNING: Cannot migrate req {req.rid}, no available instances")
                unmigrated.append(req)
                continue

            req.migrated_from_switch = True
            req.migration_timestamp = self.current_time
            req.assigned_prefill_instance = target.instance_id

            # Insert at head for priority
            target.bootstrap_queue.insert(0, req)
            count += 1

        # Put back any unmigrated requests
        source.bootstrap_queue.extend(unmigrated)
        return count

    def _migrate_prefill_waiting_queue(self, source: PrefillInstance) -> int:
        """Migrate waiting queue requests with priority (batched)."""
        if not source.waiting_queue:
            return 0

        requests = source.waiting_queue[:]
        source.waiting_queue.clear()

        count = 0
        unmigrated = []
        for req in requests:
            target = self.instance_manager.select_instance_for_migrated(
                InstanceType.PREFILL, source.instance_id
            )
            if target is None:
                unmigrated.append(req)
                continue

            source.free_memory(req)

            req.migrated_from_switch = True
            req.migration_timestamp = self.current_time
            req.assigned_prefill_instance = target.instance_id

            target.waiting_queue.insert(0, req)
            count += 1

        source.waiting_queue.extend(unmigrated)
        return count

    def _migrate_decode_prealloc_reserved(self, source: DecodeInstance) -> int:
        """Migrate virtual reservations (batched)."""
        if not source.prealloc_reserved:
            return 0

        requests = source.prealloc_reserved[:]
        source.prealloc_reserved.clear()

        count = 0
        unmigrated = []
        for req in requests:
            target = self.instance_manager.select_instance_for_migrated(
                InstanceType.DECODE, source.instance_id
            )
            if target is None:
                unmigrated.append(req)
                continue

            req.migrated_from_switch = True
            req.migration_timestamp = self.current_time
            req.assigned_decode_instance = target.instance_id

            target.prealloc_reserved.insert(0, req)
            count += 1

        source.prealloc_reserved.extend(unmigrated)
        return count

    def _migrate_decode_prealloc_queue(self, source: DecodeInstance) -> int:
        """Migrate prealloc queue requests (batched)."""
        if not source.prealloc_queue:
            return 0

        requests = source.prealloc_queue[:]
        source.prealloc_queue.clear()

        count = 0
        unmigrated = []
        for req in requests:
            target = self.instance_manager.select_instance_for_migrated(
                InstanceType.DECODE, source.instance_id
            )
            if target is None:
                unmigrated.append(req)
                continue

            req.migrated_from_switch = True
            req.migration_timestamp = self.current_time
            req.assigned_decode_instance = target.instance_id

            target.prealloc_queue.insert(0, req)
            count += 1

        source.prealloc_queue.extend(unmigrated)
        return count

    def _migrate_decode_waiting_queue(self, source: DecodeInstance) -> int:
        """Migrate decode waiting queue requests (batched)."""
        if not source.waiting_queue:
            return 0

        requests = source.waiting_queue[:]
        source.waiting_queue.clear()

        count = 0
        unmigrated = []
        for req in requests:
            target = self.instance_manager.select_instance_for_migrated(
                InstanceType.DECODE, source.instance_id
            )
            if target is None:
                unmigrated.append(req)
                continue

            source.free_memory(req)

            req.migrated_from_switch = True
            req.migration_timestamp = self.current_time
            req.assigned_decode_instance = target.instance_id

            target.waiting_queue.insert(0, req)
            count += 1

        source.waiting_queue.extend(unmigrated)
        return count

    def _complete_drain(self, instance) -> List[Event]:
        """Complete drain process after running_batch becomes empty.

        Unblock at max(drain_complete_time, switch_initiated + min_blocking).
        This means drain and the minimum blocking period run concurrently.
        """
        drain_duration = self.current_time - instance.switch_initiated_time
        instance.drain_duration = drain_duration
        # Unblock when both drain is done AND min blocking period has elapsed
        instance.blocked_until = max(
            self.current_time,
            instance.switch_initiated_time + self.config.switch_min_blocking_time,
        )
        remaining = instance.blocked_until - self.current_time

        print(f"[{self.current_time:.3f}s] Drain complete for {instance.instance_id}: "
              f"drain_time={drain_duration:.2f}s, remaining_block={remaining:.2f}s")

        return [Event(
            timestamp=instance.blocked_until,
            event_type=EventType.SWITCH_UNBLOCK,
            data={"instance": instance}
        )]

    def _handle_switch_unblock(self, event: Event) -> List[Event]:
        """Handle SWITCH_UNBLOCK event - instance resumes with new role.

        Updates instance role and recreates scheduler for new role.
        """
        instance = event.data["instance"]

        old_role = instance.instance_type
        new_role = instance.switch_target_role

        print(f"[{self.current_time:.3f}s] Role switch complete: "
              f"{instance.instance_id} {old_role.name} → {new_role.name}")

        # Record role switch in monitor
        if self.time_series_monitor:
            self.time_series_monitor.record_role_switch(
                time=self.current_time,
                instance_id=instance.instance_id,
                from_role=old_role.name,
                to_role=new_role.name
            )

        # Update instance state
        instance.instance_type = new_role
        instance.draining = False
        instance.accepting_requests = True
        instance.switch_target_role = None

        # Migrate any remaining requests that arrived during drain period
        leftover_count = 0
        if instance.waiting_queue:
            leftover_reqs = list(instance.waiting_queue)
            instance.waiting_queue.clear()
            for req in leftover_reqs:
                # Free memory on old instance
                instance.free_memory(req)
                # Re-route to appropriate instance of the OLD role
                if old_role == InstanceType.DECODE:
                    target = self.instance_manager.select_least_loaded_decode()
                else:
                    target = self.instance_manager.select_least_loaded_prefill()
                if target:
                    target.add_request(req)
                    leftover_count += 1
        if instance.running_batch and not instance.running_batch.is_empty():
            orphaned_reqs = list(instance.running_batch.reqs)
            instance.running_batch.reqs.clear()
            for req in orphaned_reqs:
                instance.free_memory(req)
                if old_role == InstanceType.DECODE:
                    target = self.instance_manager.select_least_loaded_decode()
                else:
                    target = self.instance_manager.select_least_loaded_prefill()
                if target:
                    target.add_request(req)
                    leftover_count += 1
        if leftover_count > 0:
            print(f"  Migrated {leftover_count} leftover requests from drain period")

        # Clear old role's queues (should already be empty from migration/drain)
        if old_role == InstanceType.PREFILL:
            instance.bootstrap_queue.clear()
            instance.inflight_queue.clear()
        elif old_role == InstanceType.DECODE:
            instance.prealloc_queue.clear()
            # Don't clear transfer_queue: in-flight KV transfers will complete
            # and be re-routed in _handle_kv_transfer_complete
            instance.prealloc_reserved.clear()

        # Move instance between instance manager lists
        if old_role == InstanceType.PREFILL and new_role == InstanceType.DECODE:
            # Remove from prefill list
            self.instance_manager.prefill_instances.remove(instance)
            # Add to decode list
            self.instance_manager.decode_instances.append(instance)
        elif old_role == InstanceType.DECODE and new_role == InstanceType.PREFILL:
            # Remove from decode list
            self.instance_manager.decode_instances.remove(instance)
            # Add to prefill list
            self.instance_manager.prefill_instances.append(instance)

        # Recreate scheduler for new role
        if new_role == InstanceType.PREFILL:
            # Resolve chunked prefill size
            chunked_prefill_size = None
            if self.config.chunked_prefill_size > 0:
                chunked_prefill_size = self.config.chunked_prefill_size

            self.prefill_schedulers[instance.instance_id] = PrefillScheduler(
                instance,
                max_prefill_tokens=self.config.max_prefill_tokens,
                chunked_prefill_size=chunked_prefill_size,
                new_token_ratio=self.config.schedule_conservativeness,
                schedule_policy=self.config.schedule_policy,
                enable_mixed_chunk=self.config.enable_mixed_chunk,
                max_running_requests=self.config.max_running_requests,
            )
            # Remove from decode schedulers
            self.decode_schedulers.pop(instance.instance_id, None)
        else:
            self.decode_schedulers[instance.instance_id] = SimpleScheduler(
                instance, self.config.max_prefill_tokens
            )
            # Remove from prefill schedulers
            self.prefill_schedulers.pop(instance.instance_id, None)

        # Try to schedule any pending work on new instance and migration targets
        events = []
        if new_role == InstanceType.PREFILL and instance.waiting_queue:
            events.extend(self._try_schedule_prefill(instance))
        elif new_role == InstanceType.DECODE and instance.waiting_queue:
            events.extend(self._try_schedule_decode(instance))

        # Trigger scheduling on instances that received migrated requests
        if leftover_count > 0:
            if old_role == InstanceType.DECODE:
                for di in self.instance_manager.decode_instances:
                    if di.waiting_queue and not di.busy:
                        events.extend(self._try_schedule_decode(di))
            else:
                for pi in self.instance_manager.prefill_instances:
                    if pi.waiting_queue and not pi.busy:
                        events.extend(self._try_schedule_prefill(pi))

        # Record switch in metrics
        self.metrics_collector.record_switch(
            time=self.current_time,
            instance_id=instance.instance_id,
            from_role=old_role.name,
            to_role=new_role.name,
            drain_time=instance.drain_duration,
        )

        return events

    def _handle_monitor_eval(self, event: Event) -> List[Event]:
        """Handle periodic policy monitor evaluation.

        Computes interval throughput metrics and passes them to the policy
        monitor for throughput-based switching decisions.

        Args:
            event: MONITOR_EVAL event

        Returns:
            List of events (may include ROLE_SWITCH events and next MONITOR_EVAL)
        """
        all_instances = (
            self.instance_manager.prefill_instances +
            self.instance_manager.decode_instances
        )

        # Compute interval metrics for throughput-based policy
        interval_duration = self.current_time - self._interval_start_time
        interval_metrics = None
        if interval_duration > 0:
            avg_dcts = (
                self._interval_decode_computed_token_sum / self._interval_decode_batch_count
                if self._interval_decode_batch_count > 0 else 0.0
            )
            avg_dbs = (
                self._interval_decode_tokens / self._interval_decode_batch_count
                if self._interval_decode_batch_count > 0 else 0.0
            )
            interval_metrics = {
                "input_throughput": self._interval_prefill_tokens / interval_duration,
                "output_throughput": self._interval_decode_tokens / interval_duration,
                "avg_decode_computed_token_sum": avg_dcts,
                "avg_decode_batch_size": avg_dbs,
            }

        # Reset counters for next interval
        self._interval_prefill_tokens = 0
        self._interval_decode_tokens = 0
        self._interval_decode_computed_token_sum = 0
        self._interval_decode_batch_count = 0
        self._interval_start_time = self.current_time

        # Evaluate policy (returns List[Event])
        switch_events = self.policy_monitor.evaluate(
            all_instances=all_instances,
            current_time=self.current_time,
            interval_metrics=interval_metrics,
        )

        # Schedule next monitor evaluation
        next_eval_event = Event(
            timestamp=self.current_time + self.config.monitor_interval_s,
            event_type=EventType.MONITOR_EVAL,
            data={},
        )

        events = [next_eval_event]
        events.extend(switch_events)
        return events
