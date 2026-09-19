"""Tests for GET /metrics and the scheduler's metrics integration."""

import threading

import pytest
from fastapi.testclient import TestClient

from app.metrics import MetricsCollector
from app.simulation.gpu import DEFAULT_GPU_PROFILE, SimulatedGPU
from app.simulation.request import RequestState, SimulatedRequest
from app.simulation.scheduler import SingleGPUScheduler

TIMEOUT = 5.0


def make_waiting_request(prompt_id=1, target=5, prompt_tokens=100):
    request = SimulatedRequest(
        prompt_id=prompt_id,
        category="summarization",
        prompt_tokens=prompt_tokens,
        target_completion_tokens=target,
    )
    request.transition_to(RequestState.TOKENIZED)
    request.transition_to(RequestState.WAITING)
    return request


def instant_scheduler(**kwargs):
    return SingleGPUScheduler(
        gpu=SimulatedGPU(DEFAULT_GPU_PROFILE), sleeper=lambda seconds: None, **kwargs
    )


# --- Scheduler integration -------------------------------------------------


def test_scheduler_records_a_completed_request():
    scheduler = instant_scheduler()
    scheduler.start()
    try:
        job = scheduler.submit(make_waiting_request(target=5))
        assert job.wait(TIMEOUT)
    finally:
        scheduler.stop()

    snapshot = scheduler.metrics.snapshot(scheduler)
    assert snapshot["requests"]["submitted"] == 1
    assert snapshot["requests"]["completed"] == 1
    assert snapshot["requests"]["failed"] == 0
    assert snapshot["tokens"]["prompt_tokens_processed"] == 100
    assert snapshot["tokens"]["completion_tokens_generated"] == 5
    for series in ("queue", "ttft", "total"):
        assert snapshot["latency_seconds"][series]["count"] == 1
        assert snapshot["latency_seconds"][series]["avg"] >= 0


def test_ttft_is_recorded_and_ordered_sensibly():
    """queue <= ttft <= total: TTFT spans queue + prefill + first decode."""
    scheduler = instant_scheduler()
    scheduler.start()
    try:
        job = scheduler.submit(make_waiting_request(target=20, prompt_tokens=500))
        assert job.wait(TIMEOUT)
    finally:
        scheduler.stop()

    timing = job.timing
    assert timing.first_token_at is not None
    assert timing.ttft_seconds is not None
    assert timing.queue_seconds <= timing.ttft_seconds + 1e-6
    assert timing.ttft_seconds <= timing.total_seconds + 1e-6


def test_first_token_timestamp_is_set_once():
    """TTFT must describe the *first* token, not the latest one."""
    scheduler = instant_scheduler()
    scheduler.start()
    try:
        job = scheduler.submit(make_waiting_request(target=50))
        assert job.wait(TIMEOUT)
    finally:
        scheduler.stop()

    # With 50 decode steps, a first-token stamp that kept being overwritten
    # would sit at the very end of service instead of near its start.
    assert job.timing.first_token_at <= job.timing.completed_at


def test_scheduler_records_a_failure():
    class FailingRequest(SimulatedRequest):
        def add_generated_tokens(self, count):
            raise RuntimeError("simulated decode failure")

    scheduler = instant_scheduler()
    scheduler.start()
    try:
        bad = FailingRequest(1, "summarization", 100, 5)
        bad.transition_to(RequestState.TOKENIZED)
        bad.transition_to(RequestState.WAITING)
        job = scheduler.submit(bad)
        assert job.wait(TIMEOUT)
    finally:
        scheduler.stop()

    snapshot = scheduler.metrics.snapshot(scheduler)
    assert snapshot["requests"]["failed"] == 1
    assert snapshot["requests"]["completed"] == 0
    # A failed request contributes no latency sample.
    assert snapshot["latency_seconds"]["total"]["count"] == 0


def test_ttft_includes_the_first_decode_step_duration():
    """The step's compute happens before its token becomes visible.

    A controlled clock and sleeper make the durations exact: the sleeper
    advances the clock by exactly what it was asked to wait for, so TTFT must
    come out as prefill + one decode step, not prefill alone.
    """
    clock = {"now": 1000.0}

    def fake_monotonic():
        return clock["now"]

    def advancing_sleeper(seconds):
        clock["now"] += seconds

    prompt_tokens = 500
    gpu = SimulatedGPU(DEFAULT_GPU_PROFILE)
    prefill = gpu.prefill_setup_seconds(
        SimulatedRequest(1, "summarization", prompt_tokens, 4)
    )
    step = gpu.decode_step_seconds
    assert prefill == pytest.approx(0.11)  # 0.01 overhead + 0.10 prefill
    assert step == pytest.approx(0.01)

    scheduler = SingleGPUScheduler(gpu=gpu, sleeper=advancing_sleeper)

    with pytest.MonkeyPatch.context() as patch:
        # The scheduler stamps timings with time.monotonic; drive it from the
        # same fake clock the sleeper advances.
        patch.setattr(
            "app.simulation.scheduler.time.monotonic", fake_monotonic, raising=False
        )
        scheduler.start()
        try:
            request = make_waiting_request(target=4, prompt_tokens=prompt_tokens)
            job = scheduler.submit(request)
            assert job.wait(TIMEOUT)
        finally:
            scheduler.stop()

    timing = job.timing
    # Queue wait is zero here; TTFT must therefore be prefill + one step.
    assert timing.queue_seconds == pytest.approx(0.0)
    assert timing.ttft_seconds == pytest.approx(prefill + step)
    assert timing.ttft_seconds > prefill, "TTFT excluded the first decode step"
    # Four decode steps in total.
    assert timing.total_seconds == pytest.approx(prefill + 4 * step)
    # The invariant still holds.
    assert timing.queue_seconds <= timing.ttft_seconds <= timing.total_seconds


def test_failed_request_still_counts_the_work_it_performed():
    """Prefill and partial decode really happened, so they are counted.

    Prompt 100 tokens: prefill succeeds, three decode tokens are generated,
    then the fourth attempt fails.
    """

    class FailsAfterThreeTokens(SimulatedRequest):
        def add_generated_tokens(self, count):
            if self.generated_tokens >= 3:
                raise RuntimeError("simulated decode failure")
            super().add_generated_tokens(count)

    scheduler = instant_scheduler()
    scheduler.start()
    try:
        request = FailsAfterThreeTokens(1, "summarization", 100, 20)
        request.transition_to(RequestState.TOKENIZED)
        request.transition_to(RequestState.WAITING)
        job = scheduler.submit(request)
        assert job.wait(TIMEOUT)
    finally:
        scheduler.stop()

    assert not job.succeeded
    assert request.state == RequestState.FAILED
    assert request.generated_tokens == 3

    snapshot = scheduler.metrics.snapshot(scheduler)
    assert snapshot["requests"]["submitted"] == 1
    assert snapshot["requests"]["completed"] == 0
    assert snapshot["requests"]["failed"] == 1
    assert snapshot["tokens"]["prompt_tokens_processed"] == 100
    assert snapshot["tokens"]["completion_tokens_generated"] == 3


def test_gpu_busy_gauge_ignores_a_queue_with_no_active_work():
    """A populated queue alone must not report the GPU as busy."""
    scheduler = instant_scheduler()
    assert scheduler.active_count == 0
    assert scheduler.is_busy is False

    # Directly exercise the gauge: queued but nothing admitted yet.
    class QueuedOnly:
        queue_depth = 5
        active_count = 0
        max_active_requests = 4
        is_busy = False

    snapshot = scheduler.metrics.snapshot(QueuedOnly())
    assert snapshot["current"]["queue_depth"] == 5
    assert snapshot["current"]["active_requests"] == 0
    assert snapshot["current"]["gpu_busy"] is False


def test_gpu_busy_gauge_is_true_while_a_request_is_in_service():
    """Observed on the real scheduler, held mid-prefill."""
    entered = threading.Event()
    release = threading.Event()

    def gated_sleeper(seconds):
        entered.set()
        assert release.wait(TIMEOUT)

    scheduler = SingleGPUScheduler(
        gpu=SimulatedGPU(DEFAULT_GPU_PROFILE), sleeper=gated_sleeper
    )
    scheduler.start()
    job = None
    try:
        assert scheduler.is_busy is False
        job = scheduler.submit(make_waiting_request(target=3))
        assert entered.wait(TIMEOUT)

        assert scheduler.active_count > 0
        assert scheduler.is_busy is True
        assert scheduler.metrics.snapshot(scheduler)["current"]["gpu_busy"] is True
    finally:
        release.set()
        if job is not None:
            job.wait(TIMEOUT)
        scheduler.stop()

    assert scheduler.is_busy is False


def test_broken_metrics_do_not_break_the_simulator():
    """Observability is not allowed to take inference down."""

    class BrokenMetrics(MetricsCollector):
        def record_submitted(self):
            raise RuntimeError("metrics exploded")

        def record_completed(self, **kwargs):
            raise RuntimeError("metrics exploded")

        def record_prompt_tokens_processed(self, count):
            raise RuntimeError("metrics exploded")

        def record_completion_tokens_generated(self, count):
            raise RuntimeError("metrics exploded")

        def mark_busy(self):
            raise RuntimeError("metrics exploded")

        def mark_idle(self):
            raise RuntimeError("metrics exploded")

    scheduler = instant_scheduler(metrics=BrokenMetrics())
    scheduler.start()
    try:
        request = make_waiting_request(target=3)
        job = scheduler.submit(request)
        assert job.wait(TIMEOUT), "a metrics failure must not hang the request"
        assert job.succeeded
        assert request.state == RequestState.COMPLETED
    finally:
        scheduler.stop()


def test_utilization_after_real_work_is_within_bounds():
    scheduler = instant_scheduler()
    scheduler.start()
    try:
        jobs = [scheduler.submit(make_waiting_request(prompt_id=i)) for i in range(5)]
        for job in jobs:
            assert job.wait(TIMEOUT)
    finally:
        scheduler.stop()

    utilization = scheduler.metrics.snapshot(scheduler)["current"][
        "simulated_gpu_utilization"
    ]
    assert 0.0 <= utilization <= 1.0


# --- HTTP endpoint ---------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def patched_scheduler():
    import app.main as main_module

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(main_module, "scheduler", instant_scheduler())
        yield


@pytest.fixture(scope="module")
def client():
    from app.main import app

    with TestClient(app) as started_client:
        yield started_client


def test_metrics_endpoint_before_traffic(client: TestClient):
    response = client.get("/metrics")
    assert response.status_code == 200
    body = response.json()
    assert body["requests"]["completed"] == 0
    assert body["latency_seconds"]["total"]["avg"] is None


def test_metrics_endpoint_structure(client: TestClient):
    client.post("/generate", json={"prompt_id": 1})
    body = client.get("/metrics").json()

    assert set(body) == {
        "uptime_seconds",
        "requests",
        "current",
        "latency_seconds",
        "throughput",
        "tokens",
    }
    assert set(body["requests"]) == {"submitted", "completed", "failed", "rejected"}
    assert set(body["current"]) == {
        "queue_depth",
        "active_requests",
        "max_active_requests",
        "batch_occupancy",
        "gpu_busy",
        "simulated_gpu_utilization",
    }
    assert set(body["latency_seconds"]) == {"queue", "ttft", "total"}
    for series in body["latency_seconds"].values():
        assert set(series) == {"avg", "p50", "p95", "count"}
    assert set(body["tokens"]) == {
        "prompt_tokens_processed",
        "completion_tokens_generated",
    }


def test_metrics_reflect_generate_traffic(client: TestClient):
    before = client.get("/metrics").json()
    for prompt_id in (10, 11, 12):
        assert client.post("/generate", json={"prompt_id": prompt_id}).status_code == 200
    after = client.get("/metrics").json()

    assert after["requests"]["completed"] >= before["requests"]["completed"] + 3
    assert (
        after["tokens"]["prompt_tokens_processed"]
        > before["tokens"]["prompt_tokens_processed"]
    )
    assert after["latency_seconds"]["total"]["avg"] is not None
    assert after["latency_seconds"]["ttft"]["count"] > 0
    assert after["throughput"]["requests_per_second_recent"] > 0


def test_metrics_gauges_are_idle_between_requests(client: TestClient):
    body = client.get("/metrics").json()
    assert body["current"]["queue_depth"] == 0
    assert body["current"]["active_requests"] == 0
    assert body["current"]["batch_occupancy"] == 0.0
    assert body["current"]["max_active_requests"] >= 1
    assert 0.0 <= body["current"]["simulated_gpu_utilization"] <= 1.0


def test_failed_lookup_does_not_count_as_a_failure(client: TestClient):
    """A 404 is a client error, not a simulated inference failure."""
    before = client.get("/metrics").json()["requests"]
    assert client.post("/generate", json={"prompt_id": 999999}).status_code == 404
    after = client.get("/metrics").json()["requests"]

    assert after["failed"] == before["failed"]
    assert after["submitted"] == before["submitted"]


def test_metrics_can_be_read_while_traffic_is_running(client: TestClient):
    """The endpoint must be safe to query concurrently with live load."""
    errors: list[BaseException] = []

    def generate_load():
        try:
            for prompt_id in range(20, 30):
                client.post("/generate", json={"prompt_id": prompt_id})
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def read_metrics():
        try:
            for _ in range(30):
                body = client.get("/metrics").json()
                assert body["current"]["batch_occupancy"] <= 1.0
                assert 0.0 <= body["current"]["simulated_gpu_utilization"] <= 1.0
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    loader = threading.Thread(target=generate_load)
    reader = threading.Thread(target=read_metrics)
    loader.start()
    reader.start()
    loader.join(30)
    reader.join(30)

    assert not errors, errors
