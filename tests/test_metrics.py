"""Tests for the metrics collector and GET /metrics.

A fake clock is used wherever elapsed time matters, so nothing waits.
"""

import threading

import pytest

from app.metrics import (
    LATENCY_WINDOW,
    THROUGHPUT_WINDOW_SECONDS,
    MetricsCollector,
    percentile,
)


class FakeClock:
    """A manually advanced monotonic clock."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeScheduler:
    """Just the gauge surface the collector reads.

    is_busy mirrors the real scheduler: active work only, never a populated
    queue on its own.
    """

    def __init__(self, queue_depth=0, active=0, capacity=4):
        self.queue_depth = queue_depth
        self.active_count = active
        self.max_active_requests = capacity

    @property
    def is_busy(self):
        return self.active_count > 0


def complete(collector, queue=0.1, ttft=0.2, total=0.5):
    """Record one successful completion. Token work is recorded separately."""
    collector.record_completed(
        queue_seconds=queue, ttft_seconds=ttft, total_seconds=total
    )


# --- Percentile helper -----------------------------------------------------


def test_percentile_of_empty_sample_is_none():
    assert percentile([], 0.5) is None


def test_percentile_nearest_rank():
    """rank = ceil(p * N), 1-indexed."""
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert percentile(values, 0.50) == 5  # ceil(5.0) = 5
    assert percentile(values, 0.95) == 10  # ceil(9.5) = 10
    assert percentile(values, 0.10) == 1


def test_percentile_single_value():
    assert percentile([42.0], 0.5) == 42.0
    assert percentile([42.0], 0.95) == 42.0


def test_percentile_handles_unsorted_input():
    assert percentile([9, 1, 5, 3, 7], 0.5) == 5


def test_percentile_never_exceeds_the_sample():
    values = list(range(1, 8))
    assert percentile(values, 1.0) == 7
    assert percentile(values, 0.999) == 7


# --- Empty state -----------------------------------------------------------


def test_snapshot_before_any_traffic():
    """No crash and no misleading zeros where there is simply no data."""
    snapshot = MetricsCollector(clock=FakeClock()).snapshot()

    assert snapshot["requests"] == {
        "submitted": 0,
        "completed": 0,
        "failed": 0,
        "rejected": 0,
    }
    for series in ("queue", "ttft", "total"):
        stats = snapshot["latency_seconds"][series]
        assert stats["avg"] is None
        assert stats["p50"] is None
        assert stats["p95"] is None
        assert stats["count"] == 0
    assert snapshot["throughput"]["requests_per_second_recent"] == 0.0
    assert snapshot["tokens"]["prompt_tokens_processed"] == 0
    assert snapshot["current"]["simulated_gpu_utilization"] == 0.0


# --- Counters --------------------------------------------------------------


def test_counters():
    collector = MetricsCollector(clock=FakeClock())
    for _ in range(5):
        collector.record_submitted()
    complete(collector)
    complete(collector)
    collector.record_failed()
    collector.record_rejected()

    requests = collector.snapshot()["requests"]
    assert requests["submitted"] == 5
    assert requests["completed"] == 2
    assert requests["failed"] == 1
    assert requests["rejected"] == 1


def test_token_totals_accumulate():
    collector = MetricsCollector(clock=FakeClock())
    collector.record_prompt_tokens_processed(100)
    collector.record_prompt_tokens_processed(250)
    collector.record_completion_tokens_generated(30)
    collector.record_completion_tokens_generated(45)

    tokens = collector.snapshot()["tokens"]
    assert tokens["prompt_tokens_processed"] == 350
    assert tokens["completion_tokens_generated"] == 75


def test_token_work_is_independent_of_completion():
    """Work is counted where it happens, not when a request finishes."""
    collector = MetricsCollector(clock=FakeClock())
    collector.record_prompt_tokens_processed(500)
    collector.record_completion_tokens_generated(4)
    collector.record_failed()

    snapshot = collector.snapshot()
    assert snapshot["requests"]["completed"] == 0
    assert snapshot["requests"]["failed"] == 1
    assert snapshot["tokens"]["prompt_tokens_processed"] == 500
    assert snapshot["tokens"]["completion_tokens_generated"] == 4


def test_batch_step_records_tokens_once():
    """One call per decode step, carrying the whole batch's tokens."""
    collector = MetricsCollector(clock=FakeClock())
    for _ in range(10):  # ten steps of a four-wide batch
        collector.record_completion_tokens_generated(4)
    assert collector.snapshot()["tokens"]["completion_tokens_generated"] == 40


# --- Latency statistics ----------------------------------------------------


def test_single_completed_request():
    collector = MetricsCollector(clock=FakeClock())
    complete(collector, queue=0.25, ttft=0.40, total=1.10)

    latency = collector.snapshot()["latency_seconds"]
    for series, value in (("queue", 0.25), ("ttft", 0.40), ("total", 1.10)):
        assert latency[series]["avg"] == pytest.approx(value)
        assert latency[series]["p50"] == pytest.approx(value)
        assert latency[series]["p95"] == pytest.approx(value)
        assert latency[series]["count"] == 1


def test_latency_statistics_over_controlled_samples():
    collector = MetricsCollector(clock=FakeClock())
    for index in range(1, 11):  # queue 0.1 .. 1.0
        complete(collector, queue=index / 10, ttft=index / 5, total=index)

    latency = collector.snapshot()["latency_seconds"]
    assert latency["queue"]["avg"] == pytest.approx(0.55)
    assert latency["queue"]["p50"] == pytest.approx(0.5)
    assert latency["queue"]["p95"] == pytest.approx(1.0)
    assert latency["total"]["avg"] == pytest.approx(5.5)
    assert latency["total"]["p50"] == pytest.approx(5)
    assert latency["total"]["p95"] == pytest.approx(10)
    assert latency["ttft"]["p50"] == pytest.approx(1.0)


def test_missing_ttft_samples_are_skipped():
    """A request that failed before decoding has no TTFT to report."""
    collector = MetricsCollector(clock=FakeClock())
    complete(collector, ttft=0.3)
    complete(collector, ttft=None)

    latency = collector.snapshot()["latency_seconds"]
    assert latency["ttft"]["count"] == 1
    assert latency["total"]["count"] == 2


def test_latency_window_is_bounded():
    """Only the most recent LATENCY_WINDOW completions are kept."""
    collector = MetricsCollector(clock=FakeClock())
    for index in range(LATENCY_WINDOW + 250):
        complete(collector, queue=float(index), ttft=None, total=float(index))

    latency = collector.snapshot()["latency_seconds"]
    assert latency["total"]["count"] == LATENCY_WINDOW
    # The oldest 250 samples have been dropped, so the minimum has moved up.
    assert latency["total"]["p50"] > 250
    # Cumulative counters are unaffected by the window.
    assert collector.snapshot()["requests"]["completed"] == LATENCY_WINDOW + 250


# --- Throughput ------------------------------------------------------------


def test_throughput_uses_elapsed_runtime_before_the_window_fills():
    """Dividing by 60 during the first seconds would understate the rate."""
    clock = FakeClock()
    collector = MetricsCollector(clock=clock)
    clock.advance(10.0)
    for _ in range(20):
        complete(collector)

    throughput = collector.snapshot()["throughput"]
    assert throughput["window_seconds"] == pytest.approx(10.0)
    assert throughput["requests_per_second_recent"] == pytest.approx(2.0)


def test_throughput_uses_the_full_window_once_elapsed():
    clock = FakeClock()
    collector = MetricsCollector(clock=clock)
    clock.advance(120.0)
    for _ in range(30):
        complete(collector)

    throughput = collector.snapshot()["throughput"]
    assert throughput["window_seconds"] == pytest.approx(THROUGHPUT_WINDOW_SECONDS)
    assert throughput["requests_per_second_recent"] == pytest.approx(0.5)


def test_old_completions_leave_the_throughput_window():
    clock = FakeClock()
    collector = MetricsCollector(clock=clock)
    clock.advance(100.0)
    for _ in range(10):
        complete(collector)

    assert collector.snapshot()["throughput"]["requests_per_second_recent"] > 0

    clock.advance(THROUGHPUT_WINDOW_SECONDS + 1)
    assert collector.snapshot()["throughput"]["requests_per_second_recent"] == 0.0
    # But the cumulative counter still remembers them.
    assert collector.snapshot()["requests"]["completed"] == 10


# --- Simulated GPU utilization ---------------------------------------------


def test_utilization_is_zero_when_never_busy():
    clock = FakeClock()
    collector = MetricsCollector(clock=clock)
    clock.advance(10.0)
    assert collector.snapshot()["current"]["simulated_gpu_utilization"] == 0.0


def test_utilization_counts_busy_intervals():
    clock = FakeClock()
    collector = MetricsCollector(clock=clock)

    collector.mark_busy()
    clock.advance(4.0)
    collector.mark_idle()
    clock.advance(6.0)  # 4 busy out of 10 elapsed

    assert collector.snapshot()["current"]["simulated_gpu_utilization"] == pytest.approx(
        0.4
    )


def test_utilization_includes_an_in_progress_busy_interval():
    """Querying /metrics mid-work must not under-report utilization."""
    clock = FakeClock()
    collector = MetricsCollector(clock=clock)

    collector.mark_busy()
    clock.advance(5.0)  # still busy, never marked idle

    assert collector.snapshot()["current"]["simulated_gpu_utilization"] == pytest.approx(
        1.0
    )


def test_utilization_stays_within_zero_and_one():
    clock = FakeClock()
    collector = MetricsCollector(clock=clock)
    for _ in range(5):
        collector.mark_busy()
        clock.advance(2.0)
        collector.mark_idle()
        clock.advance(1.0)
        value = collector.snapshot()["current"]["simulated_gpu_utilization"]
        assert 0.0 <= value <= 1.0


def test_repeated_mark_busy_does_not_double_count():
    clock = FakeClock()
    collector = MetricsCollector(clock=clock)
    collector.mark_busy()
    clock.advance(2.0)
    collector.mark_busy()  # already busy
    clock.advance(2.0)
    collector.mark_idle()
    clock.advance(4.0)  # 4 busy out of 8

    assert collector.snapshot()["current"]["simulated_gpu_utilization"] == pytest.approx(
        0.5
    )


def test_mark_idle_when_already_idle_is_harmless():
    collector = MetricsCollector(clock=FakeClock())
    collector.mark_idle()
    collector.mark_idle()
    assert collector.snapshot()["current"]["simulated_gpu_utilization"] == 0.0


# --- Current gauges --------------------------------------------------------


def test_gauges_reflect_the_scheduler():
    collector = MetricsCollector(clock=FakeClock())
    scheduler = FakeScheduler(queue_depth=7, active=3, capacity=4)

    current = collector.snapshot(scheduler)["current"]
    assert current["queue_depth"] == 7
    assert current["active_requests"] == 3
    assert current["max_active_requests"] == 4
    assert current["batch_occupancy"] == pytest.approx(0.75)
    assert current["gpu_busy"] is True


def test_gpu_busy_is_false_when_only_the_queue_is_populated():
    """Waiting in the queue is not GPU work."""
    collector = MetricsCollector(clock=FakeClock())
    scheduler = FakeScheduler(queue_depth=5, active=0)

    assert collector.snapshot(scheduler)["current"]["gpu_busy"] is False


def test_gpu_busy_is_true_when_requests_are_active():
    collector = MetricsCollector(clock=FakeClock())
    scheduler = FakeScheduler(queue_depth=0, active=1)

    assert collector.snapshot(scheduler)["current"]["gpu_busy"] is True


def test_batch_occupancy_never_exceeds_one():
    collector = MetricsCollector(clock=FakeClock())
    for active in range(0, 9):
        scheduler = FakeScheduler(active=active, capacity=4)
        occupancy = collector.snapshot(scheduler)["current"]["batch_occupancy"]
        assert 0.0 <= occupancy <= 1.0


# --- Thread safety ---------------------------------------------------------


def test_concurrent_updates_are_not_lost():
    """`+= 1` on an int is not safe to assume atomic - the lock must hold."""
    collector = MetricsCollector()
    threads_count, per_thread = 8, 500

    def worker():
        for _ in range(per_thread):
            collector.record_submitted()
            collector.record_prompt_tokens_processed(1)
            collector.record_completion_tokens_generated(1)
            complete(collector)

    threads = [threading.Thread(target=worker) for _ in range(threads_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)

    expected = threads_count * per_thread
    snapshot = collector.snapshot()
    assert snapshot["requests"]["submitted"] == expected
    assert snapshot["requests"]["completed"] == expected
    assert snapshot["tokens"]["prompt_tokens_processed"] == expected
    assert snapshot["tokens"]["completion_tokens_generated"] == expected


def test_snapshot_while_updates_are_in_flight():
    """/metrics must be safe to query during live traffic."""
    collector = MetricsCollector()
    stop = threading.Event()
    errors: list[BaseException] = []

    def writer():
        try:
            while not stop.is_set():
                collector.record_submitted()
                collector.record_prompt_tokens_processed(10)
                collector.record_completion_tokens_generated(4)
                complete(collector)
                collector.mark_busy()
                collector.mark_idle()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def reader():
        try:
            for _ in range(500):
                snapshot = collector.snapshot(FakeScheduler(active=2))
                assert snapshot["requests"]["completed"] >= 0
                assert 0.0 <= snapshot["current"]["simulated_gpu_utilization"] <= 1.0
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    writers = [threading.Thread(target=writer) for _ in range(3)]
    readers = [threading.Thread(target=reader) for _ in range(3)]
    for thread in writers + readers:
        thread.start()
    for thread in readers:
        thread.join(30)
    stop.set()
    for thread in writers:
        thread.join(30)

    assert not errors, errors
