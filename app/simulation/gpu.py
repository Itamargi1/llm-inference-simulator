"""A simulated GPU: stage timing, and the wall-clock delay that models it.

**Nothing here touches real hardware.** No GPU is detected, accessed or
required, and no model is loaded. This is a software model of how long an
inference request would take, built from two throughput numbers.

Why prefill and decode are separate
-----------------------------------
They are different kinds of work:

* **Prefill** processes the entire input prompt. The whole sequence is
  available at once, so the work parallelises well and the per-token cost is
  low.
* **Decode** generates the output autoregressively - one token at a time,
  each depending on the last. It parallelises far worse, so the per-token
  cost is much higher.

Modelling them with one combined throughput would hide exactly the behavior
this simulator exists to reason about, so they get separate parameters.

This module is **pure**: every function here computes a duration and nothing
here ever waits. Execution - actually spending that time, and driving request
state - belongs to the scheduler, which owns the batching loop and the
injected sleeper. Keeping the timing model free of waiting is what lets the
scheduler plan with it and lets tests assert on it instantly.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import (
    DEFAULT_DECODE_TOKENS_PER_SECOND,
    DEFAULT_FIXED_OVERHEAD_SECONDS,
    DEFAULT_GPU_NAME,
    DEFAULT_PREFILL_TOKENS_PER_SECOND,
)
from app.simulation.request import SimulatedRequest


@dataclass(frozen=True)
class GPUProfile:
    """Serving-relevant properties of one simulated GPU.

    Only what the timing model needs. VRAM, KV-cache capacity and batch-token
    limits are outside this simulator's scope; scheduler concurrency is a
    separate assumed parameter.

    All values are assumed simulation parameters, not measurements.
    """

    name: str
    prefill_tokens_per_second: float
    decode_tokens_per_second: float
    fixed_overhead_seconds: float

    def __post_init__(self) -> None:
        if self.prefill_tokens_per_second <= 0:
            raise ValueError(
                f"prefill_tokens_per_second must be > 0, got "
                f"{self.prefill_tokens_per_second}"
            )
        if self.decode_tokens_per_second <= 0:
            raise ValueError(
                f"decode_tokens_per_second must be > 0, got "
                f"{self.decode_tokens_per_second}"
            )
        if self.fixed_overhead_seconds < 0:
            raise ValueError(
                f"fixed_overhead_seconds cannot be negative, got "
                f"{self.fixed_overhead_seconds}"
            )


DEFAULT_GPU_PROFILE = GPUProfile(
    name=DEFAULT_GPU_NAME,
    prefill_tokens_per_second=DEFAULT_PREFILL_TOKENS_PER_SECOND,
    decode_tokens_per_second=DEFAULT_DECODE_TOKENS_PER_SECOND,
    fixed_overhead_seconds=DEFAULT_FIXED_OVERHEAD_SECONDS,
)


@dataclass(frozen=True)
class StageTiming:
    """Predicted service time for one request, broken down by stage.

    This is *service* time only - the time spent being served. Queue waiting
    is not included.
    """

    fixed_overhead_seconds: float
    prefill_seconds: float
    decode_seconds: float

    @property
    def total_seconds(self) -> float:
        return self.fixed_overhead_seconds + self.prefill_seconds + self.decode_seconds


class SimulatedGPU:
    """The timing model of one simulated GPU. Computes only; never waits."""

    def __init__(self, profile: GPUProfile = DEFAULT_GPU_PROFILE) -> None:
        self.profile = profile

    # --- Pure calculations -------------------------------------------------

    @property
    def decode_step_seconds(self) -> float:
        """How long one batched decode iteration takes.

        SIMPLIFYING ASSUMPTION: one iteration costs the same whether one
        sequence or MAX_ACTIVE_REQUESTS sequences are active, so aggregate
        token throughput scales linearly with batch occupancy. Real batching
        does not scale linearly - kernel efficiency, sequence lengths and
        memory bandwidth all matter - but a linear model captures the
        qualitative point (batching raises throughput) without inventing a
        scaling curve we have not measured.

        With one active request this reduces exactly to the isolated model:
        target_tokens steps x (1 / decode_tokens_per_second).
        """
        return 1.0 / self.profile.decode_tokens_per_second

    def prefill_setup_seconds(self, request: SimulatedRequest) -> float:
        """Fixed overhead plus prefill: the cost of admitting one request."""
        return self.profile.fixed_overhead_seconds + self.prefill_seconds(request)

    def prefill_seconds(self, request: SimulatedRequest) -> float:
        """Time to process the whole prompt. Independent of output length."""
        if request.prompt_tokens < 0:
            raise ValueError(f"prompt_tokens cannot be negative: {request.prompt_tokens}")
        return request.prompt_tokens / self.profile.prefill_tokens_per_second

    def decode_seconds(self, request: SimulatedRequest) -> float:
        """Time to generate the output. Independent of prompt length.

        Uses target_completion_tokens - the simulated decode *work* - rather
        than a token count measured from the placeholder text, which is only
        an artefact of how the fake content is built.
        """
        if request.target_completion_tokens < 0:
            raise ValueError(
                f"target_completion_tokens cannot be negative: "
                f"{request.target_completion_tokens}"
            )
        return request.target_completion_tokens / self.profile.decode_tokens_per_second

    def estimate_timing(self, request: SimulatedRequest) -> StageTiming:
        """Predict isolated service time: this request alone on the GPU.

        A per-request work estimate. Under continuous batching it is not
        what the request observes, because decode steps are shared with the
        other active sequences and a newly admitted prefill can delay a step.
        JobTiming records what actually happened.
        """
        return StageTiming(
            fixed_overhead_seconds=self.profile.fixed_overhead_seconds,
            prefill_seconds=self.prefill_seconds(request),
            decode_seconds=self.decode_seconds(request),
        )
