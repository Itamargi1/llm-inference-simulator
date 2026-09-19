"""A single-GPU scheduler with simplified continuous batching.

Up to MAX_ACTIVE_REQUESTS sequences decode together. A finished request is
replaced as soon as its slot becomes available, so admission does not wait for
the rest of the active batch to drain.

The loop
--------
    while running:
        fill free slots from the FIFO queue (prefill each admitted request)
        if any request is active:
            run one batched decode step
        else:
            block until new work arrives

One decode step gives **every** active request one token and then waits once
for `gpu.decode_step_seconds`. Aggregate throughput therefore scales with
batch occupancy - see SimulatedGPU.decode_step_seconds for why that linear
assumption is deliberate and where it departs from real hardware.

The queue, the worker thread and the events are real application
synchronization. Only GPU timing is simulated: no real GPU, CUDA, model or
vLLM process is involved.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from app.config import MAX_ACTIVE_REQUESTS
from app.metrics import MetricsCollector
from app.simulation.gpu import SimulatedGPU
from app.simulation.request import (
    InvalidStateTransition,
    RequestState,
    SimulatedRequest,
)

# Waits for a number of seconds. time.sleep in the application; a recording
# no-op in tests, so the suite never waits on simulated time.
Sleeper = Callable[[float], None]


@dataclass
class JobTiming:
    """Observed timing for one request, measured with time.monotonic().

    Monotonic rather than wall-clock: the system clock can jump backwards and
    produce negative latencies.

    Under batching, `service_seconds` covers this request's own prefill, the
    decode steps it shared with other active sequences, and any pause while a
    newly admitted request was prefilled. It is therefore *not* expected to
    equal the GPU's isolated StageTiming estimate.
    """

    enqueued_at: float
    service_started_at: Optional[float] = None
    first_token_at: Optional[float] = None
    completed_at: Optional[float] = None

    @property
    def ttft_seconds(self) -> Optional[float]:
        """Time to first token: enqueue until the first decode token.

        Spans queue wait, prefill, and the wait until decode first produces
        output - which is what a user actually perceives as responsiveness.
        """
        if self.first_token_at is None:
            return None
        return self.first_token_at - self.enqueued_at

    @property
    def queue_seconds(self) -> Optional[float]:
        """How long the request waited before it was admitted."""
        if self.service_started_at is None:
            return None
        return self.service_started_at - self.enqueued_at

    @property
    def service_seconds(self) -> Optional[float]:
        """How long it was in service, sharing the GPU with others."""
        if self.service_started_at is None or self.completed_at is None:
            return None
        return self.completed_at - self.service_started_at

    @property
    def total_seconds(self) -> Optional[float]:
        """End to end: queue wait plus service."""
        if self.completed_at is None:
            return None
        return self.completed_at - self.enqueued_at


@dataclass
class ScheduledJob:
    """One queued request, plus the means to wake its waiting caller."""

    request: SimulatedRequest
    timing: JobTiming
    # Set when the job finishes, successfully or not. The HTTP thread blocks
    # on this rather than polling - no busy-waiting.
    done: threading.Event = field(default_factory=threading.Event)
    error: Optional[BaseException] = None

    @property
    def succeeded(self) -> bool:
        return self.error is None

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Block until the scheduler finishes this job."""
        return self.done.wait(timeout)


class SingleGPUScheduler:
    """FIFO admission into a continuous batch on one simulated GPU.

    One GPU and one worker, serving several sequences at once.

    Admission is strictly FIFO. Completion order need not match admission
    order, because requests have different completion targets and a short one
    admitted later can finish first.

    The queue is unbounded: nothing is rejected for load, so REJECTED remains
    unused. Bounded queues and memory-based admission are not modelled.
    """

    def __init__(
        self,
        gpu: Optional[SimulatedGPU] = None,
        max_active_requests: int = MAX_ACTIVE_REQUESTS,
        sleeper: Sleeper = time.sleep,
        metrics: Optional[MetricsCollector] = None,
    ) -> None:
        if max_active_requests < 1:
            raise ValueError(
                f"max_active_requests must be >= 1, got {max_active_requests}"
            )
        self.gpu = gpu if gpu is not None else SimulatedGPU()
        self.max_active_requests = max_active_requests
        self._sleep = sleeper
        self.metrics = metrics if metrics is not None else MetricsCollector()

        self._queue: queue.Queue = queue.Queue()
        self._active: list[ScheduledJob] = []
        self._lock = threading.Lock()  # guards _active for observers
        # Signals the idle worker that there is something to do.
        self._wake = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._running = False

    # --- Metrics -----------------------------------------------------------

    def _record(self, name: str, *args, **kwargs) -> None:
        """Call a metrics method, swallowing any failure.

        Observability must never take the simulator down: a broken counter is
        a reporting problem, not a reason to fail a request.
        """
        try:
            getattr(self.metrics, name)(*args, **kwargs)
        except Exception:  # noqa: BLE001 - deliberately non-fatal
            pass

    # --- Lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Start the worker thread. Idempotent."""
        if self._running:
            return
        self._running = True
        self._wake.clear()
        self._worker = threading.Thread(
            target=self._run_worker, name="simulated-gpu-worker", daemon=True
        )
        self._worker.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the worker cleanly, so no thread leaks after shutdown."""
        if not self._running:
            return
        self._running = False
        self._wake.set()  # release the worker if it is idle
        if self._worker is not None:
            self._worker.join(timeout)
            self._worker = None

    @property
    def is_running(self) -> bool:
        return self._running

    # --- Observable state --------------------------------------------------

    @property
    def queue_depth(self) -> int:
        """Requests waiting for a free batch slot."""
        return self._queue.qsize()

    @property
    def active_count(self) -> int:
        """Requests currently in PREFILL or DECODING."""
        with self._lock:
            return len(self._active)

    @property
    def active_requests(self) -> list[SimulatedRequest]:
        with self._lock:
            return [job.request for job in self._active]

    @property
    def is_busy(self) -> bool:
        """True when the simulated GPU is doing prefill or decode work.

        Requests join the active set before their prefill begins, so an
        active count above zero means compute is happening now. A request
        merely sitting in the queue is *not* GPU work - counting it would
        report the GPU as busy while it is actually idle behind a full batch.
        """
        return self.active_count > 0

    # --- Submission --------------------------------------------------------

    def submit(self, request: SimulatedRequest) -> ScheduledJob:
        """Enqueue a request that is already in WAITING.

        The HTTP layer owns the request up to WAITING; from here the
        scheduler owns it and drives PREFILL, DECODING and COMPLETED.
        """
        if not self._running:
            # Fail loudly rather than queueing into a scheduler with no
            # worker: that job would never run and its caller would block
            # forever.
            raise RuntimeError(
                "Scheduler is not running; start() it before submitting work."
            )
        if request.state is not RequestState.WAITING:
            raise InvalidStateTransition(
                f"Only a request in {RequestState.WAITING.value!r} may be "
                f"submitted; request {request.request_id} is in "
                f"{request.state.value!r}."
            )

        job = ScheduledJob(
            request=request, timing=JobTiming(enqueued_at=time.monotonic())
        )
        self._queue.put(job)
        self._record("record_submitted")
        self._wake.set()  # an idle worker has work now
        return job

    # --- Worker loop -------------------------------------------------------

    def _run_worker(self) -> None:
        while self._running:
            # Busy means prefill or decode work is pending, never merely that
            # HTTP requests exist.
            if self._active or not self._queue.empty():
                self._record("mark_busy")

            # WAITING -> PREFILL: admit queued requests into free batch slots.
            self._admit_waiting_requests()

            if self._active:
                self._run_decode_step()
            else:
                # Nothing active and nothing queued: block rather than spin.
                self._record("mark_idle")
                self._wake.wait()
                self._wake.clear()
        self._record("mark_idle")

    def _admit_waiting_requests(self) -> None:
        """Fill free batch slots from the FIFO queue, prefilling each.

        Prefills are performed one at a time and are *not* overlapped with
        decode: a long prompt admitted now delays the next decode step for
        the sequences already active. Real engines mitigate this with chunked
        prefill, which this simulator deliberately does not model.
        """
        while self._running and len(self._active) < self.max_active_requests:
            try:
                job = self._queue.get_nowait()
            except queue.Empty:
                return

            job.timing.service_started_at = time.monotonic()
            try:
                # PREFILL: process the input prompt before token generation.
                # The job joins the active set before the wait because it has
                # left the queue and is occupying a slot; counting it only
                # afterwards would make the GPU look idle mid-prefill.
                job.request.transition_to(RequestState.PREFILL)
                with self._lock:
                    self._active.append(job)
                self._sleep(self.gpu.prefill_setup_seconds(job.request))
                job.request.transition_to(RequestState.DECODING)
            except BaseException as exc:  # noqa: BLE001 - recorded, surfaced
                self._fail(job, exc)
                continue

            # Prompt tokens are counted here, not at completion: the prefill
            # work happened even if the request later fails during decode.
            self._record(
                "record_prompt_tokens_processed", job.request.prompt_tokens
            )

            # A request needing no decode work is finished already.
            if job.request.is_decode_complete:
                self._complete(job)

    def _run_decode_step(self) -> None:
        """One batched decode iteration: one token for every active request.

        Order matters: the step's compute is spent first, and only then does
        the resulting token become visible. Stamping the token before the
        wait would make time-to-first-token exclude the step that produced
        it, understating what a user waits for.

        The single wait is shared by the whole batch, which is what makes
        aggregate throughput rise with occupancy.
        """
        with self._lock:
            batch = list(self._active)

        # DECODE: one simulated token step, shared by the active batch.
        self._sleep(self.gpu.decode_step_seconds)

        # The tokens this step produced now become visible.
        failed: list[tuple[ScheduledJob, BaseException]] = []
        generated = 0
        for job in batch:
            try:
                job.request.add_generated_tokens(1)
            except BaseException as exc:  # noqa: BLE001
                failed.append((job, exc))
                continue
            generated += 1
            if job.timing.first_token_at is None:
                job.timing.first_token_at = time.monotonic()

        # Decode work performed, counted whether or not these requests go on
        # to complete. One call per step rather than one per token.
        if generated:
            self._record("record_completion_tokens_generated", generated)

        for job, exc in failed:
            self._fail(job, exc)

        # COMPLETED: a request that reached its target leaves at once, and
        # its slot is refilled on the next iteration. Not waiting for the
        # rest of the batch is what makes the batching continuous.
        for job in batch:
            if job.error is None and job.request.is_decode_complete:
                self._complete(job)

    # --- Job termination ---------------------------------------------------

    def _complete(self, job: ScheduledJob) -> None:
        try:
            job.request.transition_to(RequestState.COMPLETED)
        except BaseException as exc:  # noqa: BLE001
            self._fail(job, exc)
            return
        self._finish(job)

    def _fail(self, job: ScheduledJob, exc: BaseException) -> None:
        job.error = exc
        try:
            job.request.transition_to(RequestState.FAILED)
        except InvalidStateTransition:
            # Already terminal; must not mask the original error.
            pass
        self._finish(job)
        self._record("record_failed")

    def _finish(self, job: ScheduledJob) -> None:
        """Release the batch slot and wake the caller."""
        with self._lock:
            if job in self._active:
                self._active.remove(job)
        job.timing.completed_at = time.monotonic()
        if job.error is None:
            # Token work is accounted for where it happens (prefill and each
            # decode step), not here, so failed requests still contribute the
            # resource work they actually consumed.
            self._record(
                "record_completed",
                queue_seconds=job.timing.queue_seconds or 0.0,
                ttft_seconds=job.timing.ttft_seconds,
                total_seconds=job.timing.total_seconds or 0.0,
            )
        job.done.set()
