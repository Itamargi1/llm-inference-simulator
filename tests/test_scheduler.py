"""Tests for the continuous-batching single-GPU scheduler.

Behaviour is proved with recording and gated sleepers plus Events, never with
real multi-second waits, so the suite stays fast and deterministic.
"""

import threading

import pytest

from app.simulation.gpu import DEFAULT_GPU_PROFILE, SimulatedGPU
from app.simulation.request import (
    InvalidStateTransition,
    RequestState,
    SimulatedRequest,
)
from app.simulation.scheduler import JobTiming, SingleGPUScheduler

# Generous upper bound for "this should have happened by now". Only reached
# when something is genuinely broken, so it never slows a passing run.
TIMEOUT = 5.0


def make_waiting_request(
    prompt_id: int = 1, target: int = 10, prompt_tokens: int = 100
) -> SimulatedRequest:
    request = SimulatedRequest(
        prompt_id=prompt_id,
        category="summarization",
        prompt_tokens=prompt_tokens,
        target_completion_tokens=target,
    )
    request.transition_to(RequestState.TOKENIZED)
    request.transition_to(RequestState.WAITING)
    return request


class RecordingSleeper:
    """A no-op sleeper that remembers every duration it was asked for."""

    def __init__(self) -> None:
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)

    @property
    def total(self) -> float:
        return sum(self.calls)


class StepGate:
    """Parks the scheduler inside each sleep so tests can inspect the batch.

    The worker blocks *inside* a sleep, which is a stable quiescent point:
    for a decode step, the tokens for that step have already been added. The
    test inspects state while parked, then calls advance() to release the
    sleep and wait until the worker parks in the next one. Without that
    second wait the worker could run several steps before an assertion ran.
    """

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.proceed = threading.Event()
        self.durations: list[float] = []
        self.opened = False

    def __call__(self, seconds: float) -> None:
        self.durations.append(seconds)
        if self.opened:
            return
        self.entered.set()
        assert self.proceed.wait(TIMEOUT), "sleeper was never released"
        self.proceed.clear()

    def wait_parked(self) -> None:
        """Block until the worker is inside a sleep."""
        assert self.entered.wait(TIMEOUT), "scheduler never reached a sleep"
        self.entered.clear()

    def advance(self) -> None:
        """Release the current sleep, then wait until the next one is reached."""
        self.proceed.set()
        self.wait_parked()

    def open_all(self) -> None:
        """Stop gating, so the scheduler can drain and shut down."""
        self.opened = True
        self.proceed.set()


# Non-zero prompt so a prefill wait (0.01 + 0.10) is distinguishable from a
# decode step wait (0.01).
PROMPT_TOKENS = 500
PREFILL_WAIT = 0.11
DECODE_WAIT = 0.01


def scheduler_with(sleeper, max_active: int = 4) -> SingleGPUScheduler:
    return SingleGPUScheduler(
        gpu=SimulatedGPU(DEFAULT_GPU_PROFILE),
        max_active_requests=max_active,
        sleeper=sleeper,
    )


@pytest.fixture
def scheduler():
    """A started scheduler that never waits, always stopped afterwards."""
    instance = scheduler_with(lambda seconds: None)
    instance.start()
    try:
        yield instance
    finally:
        instance.stop()


# --- Single-request timing -------------------------------------------------


def test_single_request_matches_the_isolated_timing_model():
    """prompt 1000, target 100 -> 0.010 + 0.200 + 1.000 = 1.210 s of work.

    A single active request must cost exactly what the non-batched model
    predicted: batching extends that model rather than replacing it.
    """
    sleeper = RecordingSleeper()
    scheduler = scheduler_with(sleeper)
    scheduler.start()
    try:
        request = make_waiting_request(prompt_tokens=1000, target=100)
        job = scheduler.submit(request)
        assert job.wait(TIMEOUT)
    finally:
        scheduler.stop()

    isolated = SimulatedGPU(DEFAULT_GPU_PROFILE).estimate_timing(request)
    assert isolated.total_seconds == pytest.approx(1.21)

    # One prefill wait of 0.21, then 100 decode steps of 0.01.
    assert sleeper.calls[0] == pytest.approx(0.21)
    decode_calls = sleeper.calls[1:101]
    assert len(decode_calls) == 100
    assert all(call == pytest.approx(0.01) for call in decode_calls)
    assert sum(sleeper.calls[:101]) == pytest.approx(1.21)


def test_single_request_completes(scheduler: SingleGPUScheduler):
    request = make_waiting_request(target=25)
    job = scheduler.submit(request)

    assert job.wait(TIMEOUT)
    assert request.state == RequestState.COMPLETED
    assert request.generated_tokens == 25
    assert job.succeeded


def test_job_records_observed_timing(scheduler: SingleGPUScheduler):
    job = scheduler.submit(make_waiting_request())
    assert job.wait(TIMEOUT)

    timing = job.timing
    assert timing.queue_seconds >= 0
    assert timing.service_seconds >= 0
    assert timing.total_seconds == pytest.approx(
        timing.queue_seconds + timing.service_seconds, abs=1e-6
    )


def test_submit_requires_a_waiting_request(scheduler: SingleGPUScheduler):
    with pytest.raises(InvalidStateTransition, match="may be submitted"):
        scheduler.submit(SimulatedRequest(1, "summarization", 100, 10))


# --- Batched decode --------------------------------------------------------


def test_two_active_requests_share_one_decode_step():
    """Both advance by one token, for a single shared decode wait.

    Note the ordering: a decode step sleeps first and applies its tokens
    afterwards, so the tokens from step 1 are visible once the worker has
    parked in step 2's sleep.
    """
    gate = StepGate()
    scheduler = scheduler_with(gate, max_active=4)
    scheduler.start()
    try:
        a = make_waiting_request(prompt_id=1, target=10, prompt_tokens=PROMPT_TOKENS)
        b = make_waiting_request(prompt_id=2, target=10, prompt_tokens=PROMPT_TOKENS)
        scheduler.submit(a)
        scheduler.submit(b)

        gate.wait_parked()  # inside A's prefill
        gate.advance()      # inside B's prefill
        gate.advance()      # inside decode step 1 - its tokens not applied yet
        assert a.generated_tokens == 0
        assert b.generated_tokens == 0

        gate.advance()      # step 1 released, step 2 entered
        assert a.generated_tokens == 1
        assert b.generated_tokens == 1

        # Two tokens of work for one completed decode wait.
        assert gate.durations == [
            pytest.approx(PREFILL_WAIT),
            pytest.approx(PREFILL_WAIT),
            pytest.approx(DECODE_WAIT),
            pytest.approx(DECODE_WAIT),
        ]
    finally:
        gate.open_all()
        scheduler.stop()


def test_four_active_requests_produce_four_tokens_per_step():
    """Aggregate throughput scales with occupancy, by assumption."""
    gate = StepGate()
    scheduler = scheduler_with(gate, max_active=4)
    scheduler.start()
    try:
        requests = [
            make_waiting_request(prompt_id=i, target=10, prompt_tokens=PROMPT_TOKENS)
            for i in range(4)
        ]
        for request in requests:
            scheduler.submit(request)

        gate.wait_parked()
        for _ in range(5):  # three remaining prefills, then two decode sleeps
            gate.advance()

        assert [r.generated_tokens for r in requests] == [1, 1, 1, 1]
        assert sum(r.generated_tokens for r in requests) == 4
        decode_waits = [d for d in gate.durations if d == pytest.approx(DECODE_WAIT)]
        # One decode step completed (the second is still in progress).
        assert len(decode_waits) == 2
    finally:
        gate.open_all()
        scheduler.stop()


# --- Continuous admission -------------------------------------------------


def test_finished_request_is_replaced_while_others_keep_decoding():
    """A (target 2) finishes, C is admitted, and B is still decoding.

    This is what makes the batch continuous rather than static: C waits only
    for a free slot, not for the whole batch to drain.
    """
    gate = StepGate()
    scheduler = scheduler_with(gate, max_active=2)
    scheduler.start()
    job_b = job_c = None
    try:
        a = make_waiting_request(prompt_id=1, target=2, prompt_tokens=PROMPT_TOKENS)
        b = make_waiting_request(prompt_id=2, target=6, prompt_tokens=PROMPT_TOKENS)
        c = make_waiting_request(prompt_id=3, target=6, prompt_tokens=PROMPT_TOKENS)

        job_a = scheduler.submit(a)
        job_b = scheduler.submit(b)
        job_c = scheduler.submit(c)

        gate.wait_parked()  # A prefill
        assert c.state == RequestState.WAITING, "C must wait for a free slot"

        gate.advance()  # B prefill
        assert c.state == RequestState.WAITING

        gate.advance()  # inside decode step 1 (tokens applied on release)
        gate.advance()  # step 1 applied, inside step 2
        assert a.generated_tokens == 1
        assert b.generated_tokens == 1

        # Releasing step 2 applies its tokens: A reaches its target of 2,
        # completes, and C is admitted into the freed slot.
        gate.advance()
        assert a.generated_tokens == 2
        assert b.generated_tokens == 2
        assert job_a.wait(TIMEOUT), "A should complete as soon as it finishes"
        assert a.state == RequestState.COMPLETED
        assert gate.durations[-1] == pytest.approx(PREFILL_WAIT)

        # The decisive assertions: C is entering service while B is still
        # active and unfinished.
        assert c.state == RequestState.PREFILL
        assert b.state == RequestState.DECODING
        assert not job_b.done.is_set(), "B must still be running when C joins"
        assert not b.is_decode_complete

        gate.advance()  # C now decoding alongside B
        assert c.state == RequestState.DECODING
        assert b.state == RequestState.DECODING
        assert scheduler.active_count == 2
    finally:
        gate.open_all()
        if job_b is not None:
            job_b.wait(TIMEOUT)
        if job_c is not None:
            job_c.wait(TIMEOUT)
        scheduler.stop()


def test_completion_order_may_differ_from_admission_order():
    """Admitted A then B; B finishes first because it needs fewer tokens.

    Checked structurally rather than by comparing timestamps: with an instant
    sleeper both completions fall inside one clock tick, so the ordering is
    asserted by observing that B is COMPLETED while A is still DECODING.
    """
    gate = StepGate()
    scheduler = scheduler_with(gate, max_active=4)
    scheduler.start()
    try:
        a = make_waiting_request(prompt_id=1, target=50, prompt_tokens=PROMPT_TOKENS)
        b = make_waiting_request(prompt_id=2, target=3, prompt_tokens=PROMPT_TOKENS)
        job_a = scheduler.submit(a)
        job_b = scheduler.submit(b)

        gate.wait_parked()  # A prefill
        gate.advance()      # B prefill - both now in the batch
        assert scheduler.active_count == 2
        assert job_a.timing.service_started_at <= job_b.timing.service_started_at

        # Four advances: enter step 1, then apply steps 1, 2 and 3. B reaches
        # its target of 3 on step 3 and completes in that same iteration.
        for _ in range(4):
            gate.advance()

        assert b.generated_tokens == 3
        assert job_b.done.is_set(), "B should complete as soon as it finishes"
        assert b.state == RequestState.COMPLETED

        # A was admitted first but is still running.
        assert not job_a.done.is_set()
        assert a.state == RequestState.DECODING
        assert a.generated_tokens == 3
        assert scheduler.active_count == 1, "B's slot must be released"
    finally:
        gate.open_all()
        job_a.wait(TIMEOUT)
        scheduler.stop()


# --- Batch cap -------------------------------------------------------------


def test_never_more_than_max_active_requests_in_service():
    gate = StepGate()
    max_active = 4
    scheduler = scheduler_with(gate, max_active=max_active)
    scheduler.start()
    observed_peak = 0
    try:
        requests = [
            make_waiting_request(prompt_id=i, target=6, prompt_tokens=PROMPT_TOKENS)
            for i in range(10)
        ]
        for request in requests:
            scheduler.submit(request)

        gate.wait_parked()
        for _ in range(25):
            active = scheduler.active_count
            observed_peak = max(observed_peak, active)
            assert active <= max_active, "too many requests active at once"
            in_service = [
                r
                for r in requests
                if r.state in (RequestState.PREFILL, RequestState.DECODING)
            ]
            assert len(in_service) <= max_active
            gate.advance()
    finally:
        gate.open_all()
        scheduler.stop()

    assert observed_peak == max_active, "the batch should actually fill up"


def test_extra_requests_stay_waiting():
    gate = StepGate()
    scheduler = scheduler_with(gate, max_active=2)
    scheduler.start()
    try:
        requests = [
            make_waiting_request(prompt_id=i, target=20, prompt_tokens=PROMPT_TOKENS)
            for i in range(5)
        ]
        for request in requests:
            scheduler.submit(request)

        gate.wait_parked()  # first prefill
        gate.advance()      # second prefill
        gate.advance()      # first decode step

        waiting = [r for r in requests if r.state == RequestState.WAITING]
        assert len(waiting) == 3
        assert scheduler.queue_depth == 3
    finally:
        gate.open_all()
        scheduler.stop()


def test_max_active_must_be_positive():
    with pytest.raises(ValueError, match="max_active_requests"):
        SingleGPUScheduler(max_active_requests=0)


# --- FIFO admission --------------------------------------------------------


def test_admission_order_is_fifo():
    """Admission follows queue order, even though completion need not."""
    gate = StepGate()
    scheduler = scheduler_with(gate, max_active=2)
    scheduler.start()
    admission_order = []
    jobs = []
    try:
        requests = [
            make_waiting_request(prompt_id=i, target=2, prompt_tokens=PROMPT_TOKENS)
            for i in range(1, 6)
        ]
        jobs = [scheduler.submit(r) for r in requests]

        seen = set()
        gate.wait_parked()
        for _ in range(30):
            for request in requests:
                if (
                    request.prompt_id not in seen
                    and request.state is not RequestState.WAITING
                ):
                    seen.add(request.prompt_id)
                    admission_order.append(request.prompt_id)
            if len(seen) == len(requests):
                break
            gate.advance()
    finally:
        gate.open_all()
        for job in jobs:
            job.wait(TIMEOUT)
        scheduler.stop()

    assert admission_order == [1, 2, 3, 4, 5]


# --- Failures --------------------------------------------------------------


def test_failure_releases_the_slot_and_the_caller():
    """A failing request must not hold its batch slot or hang its caller."""

    class FailingRequest(SimulatedRequest):
        def add_generated_tokens(self, count: int) -> None:
            raise RuntimeError("simulated decode failure")

    scheduler = scheduler_with(lambda seconds: None, max_active=2)
    scheduler.start()
    try:
        bad = FailingRequest(1, "summarization", 100, 10)
        bad.transition_to(RequestState.TOKENIZED)
        bad.transition_to(RequestState.WAITING)

        job = scheduler.submit(bad)
        assert job.wait(TIMEOUT), "caller was not released after failure"
        assert not job.succeeded
        assert isinstance(job.error, RuntimeError)
        assert bad.state == RequestState.FAILED
        assert scheduler.active_count == 0

        # The scheduler survives and serves the next request.
        good = make_waiting_request(prompt_id=2, target=5)
        good_job = scheduler.submit(good)
        assert good_job.wait(TIMEOUT)
        assert good.state == RequestState.COMPLETED
    finally:
        scheduler.stop()


def test_failure_during_prefill_is_handled():
    class BadPrefillGPU(SimulatedGPU):
        def prefill_setup_seconds(self, request):
            raise RuntimeError("prefill blew up")

    scheduler = SingleGPUScheduler(
        gpu=BadPrefillGPU(DEFAULT_GPU_PROFILE), sleeper=lambda s: None
    )
    scheduler.start()
    try:
        request = make_waiting_request()
        job = scheduler.submit(request)
        assert job.wait(TIMEOUT)
        assert not job.succeeded
        assert request.state == RequestState.FAILED
        assert scheduler.active_count == 0
    finally:
        scheduler.stop()


# --- Lifecycle and observable state ----------------------------------------


def test_start_and_stop_terminate_the_worker():
    scheduler = scheduler_with(lambda seconds: None)
    assert not scheduler.is_running

    scheduler.start()
    assert scheduler.is_running
    worker = scheduler._worker
    assert worker is not None and worker.is_alive()

    scheduler.stop()
    assert not scheduler.is_running
    worker.join(TIMEOUT)
    assert not worker.is_alive(), "worker thread leaked after stop()"


def test_idle_worker_blocks_then_wakes_on_submission(scheduler: SingleGPUScheduler):
    """No busy-waiting: an idle worker still picks up new work promptly."""
    assert not scheduler.is_busy
    job = scheduler.submit(make_waiting_request(target=3))
    assert job.wait(TIMEOUT)
    assert not scheduler.is_busy


def test_start_is_idempotent():
    scheduler = scheduler_with(lambda seconds: None)
    scheduler.start()
    worker = scheduler._worker
    scheduler.start()
    try:
        assert scheduler._worker is worker
    finally:
        scheduler.stop()


def test_stop_is_safe_when_not_running():
    scheduler_with(lambda seconds: None).stop()  # must not raise


def test_submitting_to_a_stopped_scheduler_raises():
    """Better a loud error than a caller blocked forever on a dead worker."""
    scheduler = scheduler_with(lambda seconds: None)
    with pytest.raises(RuntimeError, match="not running"):
        scheduler.submit(make_waiting_request())

    scheduler.start()
    scheduler.stop()
    with pytest.raises(RuntimeError, match="not running"):
        scheduler.submit(make_waiting_request())


def test_busy_and_idle_state():
    """Busy state reflects active simulated compute, not queued work."""
    gate = StepGate()
    scheduler = scheduler_with(gate, max_active=2)
    scheduler.start()
    job = None
    try:
        assert not scheduler.is_busy
        assert scheduler.active_count == 0

        job = scheduler.submit(
            make_waiting_request(target=3, prompt_tokens=PROMPT_TOKENS)
        )
        # A request being prefilled already occupies its slot, so the GPU
        # must not look idle here.
        gate.wait_parked()
        assert scheduler.is_busy
        assert scheduler.active_count == 1

        gate.advance()  # now decoding
        assert scheduler.active_count == 1
        assert scheduler.is_busy
    finally:
        gate.open_all()
        if job is not None:
            job.wait(TIMEOUT)
        scheduler.stop()


# --- JobTiming -------------------------------------------------------------


def test_job_timing_is_none_until_measured():
    timing = JobTiming(enqueued_at=100.0)
    assert timing.queue_seconds is None
    assert timing.service_seconds is None
    assert timing.total_seconds is None


def test_job_timing_arithmetic():
    timing = JobTiming(enqueued_at=100.0, service_started_at=100.5, completed_at=102.0)
    assert timing.queue_seconds == pytest.approx(0.5)
    assert timing.service_seconds == pytest.approx(1.5)
    assert timing.total_seconds == pytest.approx(2.0)
