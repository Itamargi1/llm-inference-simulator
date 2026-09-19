"""In-memory metrics for the simulator.

Deliberately small: a thread-safe collector plus a JSON snapshot. No
Prometheus, OpenTelemetry or time-series database - for a local simulator the
extra dependency would cost more than it explains.

**Everything here describes the simulation.** `simulated_gpu_utilization` in
particular is derived from the scheduler's own busy/idle bookkeeping, not from
any hardware API. No GPU is queried, and nvidia-smi/NVML are never involved.

Two kinds of numbers, kept clearly apart:

* **Cumulative counters** (requests submitted/completed/failed, tokens) grow
  for the process lifetime.
* **Latency statistics** are computed over a bounded window of the most
  recent completed requests, so memory does not grow without limit and the
  numbers reflect recent behaviour rather than an average since boot.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Sequence

# Latency statistics are computed over at most this many recent requests.
LATENCY_WINDOW = 1000

# Recent throughput is measured over this trailing window.
THROUGHPUT_WINDOW_SECONDS = 60.0


def percentile(values: Sequence[float], fraction: float) -> Optional[float]:
    """Nearest-rank percentile: rank = ceil(p * N), 1-indexed.

    Chosen because it is trivial to explain and needs no interpolation or
    third-party library. Returns None for an empty sample, so callers can
    report "no data yet" rather than a misleading zero.
    """
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def _summarise(values: Sequence[float]) -> dict[str, Optional[float]]:
    """avg / p50 / p95 for one latency series."""
    if not values:
        return {"avg": None, "p50": None, "p95": None, "count": 0}
    return {
        "avg": sum(values) / len(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "count": len(values),
    }


@dataclass
class _CompletedSample:
    """One finished request's observed latencies, in seconds."""

    queue_seconds: float
    ttft_seconds: Optional[float]
    total_seconds: float
    completed_at: float


class MetricsCollector:
    """Thread-safe metrics state.

    The scheduler calls the record_* methods from its worker thread while the
    HTTP layer calls snapshot() from request threads, so every mutation and
    every read happens under one lock. Nothing here relies on an operation
    being "probably atomic" - `+= 1` on an int is not safe to assume.

    This object stores what it is told. It does not re-derive scheduling
    decisions: current gauges are read from the scheduler at snapshot time.
    """

    def __init__(self, clock=time.monotonic) -> None:
        # Injectable clock so tests can control time without sleeping.
        self._clock = clock
        self._lock = threading.Lock()
        self._started_at = clock()

        # Cumulative counters.
        self._submitted = 0
        self._completed = 0
        self._failed = 0
        self._rejected = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0

        # Bounded recent samples.
        self._samples: deque[_CompletedSample] = deque(maxlen=LATENCY_WINDOW)
        self._completion_times: deque[float] = deque()

        # Simulated GPU busy accounting.
        self._busy_seconds = 0.0
        self._busy_since: Optional[float] = None

    # --- Recording (called by the scheduler) -------------------------------

    def record_submitted(self) -> None:
        with self._lock:
            self._submitted += 1

    def record_rejected(self) -> None:
        with self._lock:
            self._rejected += 1

    def record_prompt_tokens_processed(self, count: int) -> None:
        """Prefill work performed, recorded when the prefill finishes.

        Independent of whether the request goes on to complete: the prompt
        was processed either way, and the resource cost was real.
        """
        with self._lock:
            self._prompt_tokens += count

    def record_completion_tokens_generated(self, count: int) -> None:
        """Decode work performed in one batch step.

        Called once per decode iteration with the number of tokens actually
        generated across the batch, so a request that later fails still
        contributes the decode work it consumed.
        """
        with self._lock:
            self._completion_tokens += count

    def record_completed(
        self,
        queue_seconds: float,
        ttft_seconds: Optional[float],
        total_seconds: float,
    ) -> None:
        """One request finished successfully.

        Handles the completion count, the latency sample and the throughput
        timestamp. Token totals are *not* its responsibility - they are
        recorded where the work happens.
        """
        with self._lock:
            # Read the clock under the lock: a timestamp taken outside it can
            # be older than state another thread sets in between.
            now = self._clock()
            self._completed += 1
            self._samples.append(
                _CompletedSample(queue_seconds, ttft_seconds, total_seconds, now)
            )
            self._completion_times.append(now)
            self._trim_completion_times(now)

    def record_failed(self) -> None:
        with self._lock:
            self._failed += 1

    def mark_busy(self) -> None:
        """The simulated GPU started doing prefill or decode work."""
        with self._lock:
            if self._busy_since is None:
                self._busy_since = self._clock()

    def mark_idle(self) -> None:
        """The simulated GPU ran out of work."""
        with self._lock:
            if self._busy_since is not None:
                self._busy_seconds += max(self._clock() - self._busy_since, 0.0)
                self._busy_since = None

    # --- Reading -----------------------------------------------------------

    def snapshot(self, scheduler=None) -> dict:
        """Build the /metrics payload.

        Current gauges are read from the scheduler rather than mirrored here,
        so the collector cannot drift out of step with the real queue.
        """
        with self._lock:
            # Read the clock under the lock. Taking it beforehand allows
            # another thread to mark the GPU busy *after* this timestamp,
            # which would make the in-progress interval below negative.
            now = self._clock()
            elapsed = max(now - self._started_at, 1e-9)

            # Include the in-progress busy interval, so utilization is
            # correct even when /metrics is queried mid-work.
            busy = self._busy_seconds
            if self._busy_since is not None:
                busy += max(now - self._busy_since, 0.0)
            utilization = min(max(busy / elapsed, 0.0), 1.0)

            self._trim_completion_times(now)
            recent_completions = len(self._completion_times)

            queue_values = [s.queue_seconds for s in self._samples]
            ttft_values = [
                s.ttft_seconds for s in self._samples if s.ttft_seconds is not None
            ]
            total_values = [s.total_seconds for s in self._samples]

            counters = {
                "submitted": self._submitted,
                "completed": self._completed,
                "failed": self._failed,
                "rejected": self._rejected,
            }
            # Actual simulated work performed, not a target restated: a
            # request that fails mid-decode still contributed these tokens.
            tokens = {
                "prompt_tokens_processed": self._prompt_tokens,
                "completion_tokens_generated": self._completion_tokens,
            }

        # Before the throughput window has elapsed, dividing by 60 would
        # understate the rate, so the denominator is the elapsed runtime.
        denominator = min(elapsed, THROUGHPUT_WINDOW_SECONDS)
        requests_per_second = recent_completions / denominator

        current = {
            "queue_depth": 0,
            "active_requests": 0,
            "max_active_requests": 0,
            "batch_occupancy": 0.0,
            "gpu_busy": False,
            "simulated_gpu_utilization": utilization,
        }
        if scheduler is not None:
            active = scheduler.active_count
            capacity = max(scheduler.max_active_requests, 1)
            current.update(
                {
                    "queue_depth": scheduler.queue_depth,
                    "active_requests": active,
                    "max_active_requests": scheduler.max_active_requests,
                    "batch_occupancy": min(active / capacity, 1.0),
                    "gpu_busy": scheduler.is_busy,
                }
            )

        return {
            "uptime_seconds": elapsed,
            "requests": counters,
            "current": current,
            "latency_seconds": {
                "queue": _summarise(queue_values),
                "ttft": _summarise(ttft_values),
                "total": _summarise(total_values),
            },
            "throughput": {
                "requests_per_second_recent": requests_per_second,
                "window_seconds": denominator,
            },
            "tokens": tokens,
        }

    # --- Internal ----------------------------------------------------------

    def _trim_completion_times(self, now: float) -> None:
        """Drop completions older than the throughput window.

        Caller must hold the lock.
        """
        cutoff = now - THROUGHPUT_WINDOW_SECONDS
        while self._completion_times and self._completion_times[0] < cutoff:
            self._completion_times.popleft()
