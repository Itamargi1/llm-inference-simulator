"""The lifecycle of one simulated inference request.

This is simulation-domain logic only. It knows nothing about FastAPI, HTTP
status codes or Pydantic - the API layer translates between the two, while
the scheduler drives service-side transitions.

Lifecycle:

    RECEIVED -> TOKENIZED -> WAITING -> PREFILL -> DECODING -> COMPLETED

REJECTED and FAILED are terminal side states. REJECTED represents a request
turned away before inference begins, although the current unbounded queue does
not use it. FAILED represents an unexpected failure during simulated
inference. There is no preemption state.

Nothing here measures or simulates time; the scheduler owns timing and batching.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum


class RequestState(str, Enum):
    """States a simulated request can occupy."""

    RECEIVED = "received"
    TOKENIZED = "tokenized"
    WAITING = "waiting"
    PREFILL = "prefill"
    DECODING = "decoding"
    COMPLETED = "completed"
    REJECTED = "rejected"
    FAILED = "failed"


class InvalidStateTransition(Exception):
    """Raised when a request is asked to make a transition that is not allowed.

    Deliberately loud: a scheduler bug that moved requests through impossible
    states would quietly corrupt every number the simulator reports.
    """


TERMINAL_STATES = frozenset(
    {RequestState.COMPLETED, RequestState.REJECTED, RequestState.FAILED}
)

# The allowed transitions, written out explicitly rather than inferred.
#
# FAILED is reachable from any active state: a simulated failure can occur at
# any point before the request finishes. REJECTED is only reachable *before*
# inference begins, which is what "rejected" means here - the request was
# turned away, not abandoned midway. Terminal states have no outgoing edges.
ALLOWED_TRANSITIONS: dict[RequestState, frozenset[RequestState]] = {
    RequestState.RECEIVED: frozenset(
        {RequestState.TOKENIZED, RequestState.REJECTED, RequestState.FAILED}
    ),
    RequestState.TOKENIZED: frozenset(
        {RequestState.WAITING, RequestState.REJECTED, RequestState.FAILED}
    ),
    RequestState.WAITING: frozenset(
        {RequestState.PREFILL, RequestState.REJECTED, RequestState.FAILED}
    ),
    RequestState.PREFILL: frozenset({RequestState.DECODING, RequestState.FAILED}),
    RequestState.DECODING: frozenset({RequestState.COMPLETED, RequestState.FAILED}),
    RequestState.COMPLETED: frozenset(),
    RequestState.REJECTED: frozenset(),
    RequestState.FAILED: frozenset(),
}


def new_request_id() -> str:
    """A unique identifier for one invocation of /generate.

    Distinct from prompt_id, which identifies dataset content: two calls with
    the same prompt_id are two different requests. uuid4 is enough - the ID
    only has to be unique within a running simulator, so no persistence or
    coordination is needed.
    """
    return str(uuid.uuid4())


@dataclass
class SimulatedRequest:
    """One inference request moving through the simulated lifecycle.

    GPU assignment, KV-cache accounting, queue timestamps and latency are not
    part of this request-domain object. Scheduler jobs hold observed timing.

    Note the distinction between two token counts:

    * ``target_completion_tokens`` is the simulated amount of decode work.
      The scheduler uses this, and ``generated_tokens`` tracks progress
      against it.
    * The ``completion_tokens`` the API reports is measured separately, by
      applying the tokenizer to the text actually returned. The two can
      differ by roughly two tokens because the placeholder text is trimmed on
      a word boundary. Timing logic uses the target and the
      generated-token bookkeeping, never a string length.
    """

    prompt_id: int
    category: str
    prompt_tokens: int
    target_completion_tokens: int
    request_id: str = field(default_factory=new_request_id)
    state: RequestState = RequestState.RECEIVED
    generated_tokens: int = 0

    # --- State machine -----------------------------------------------------

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    def transition_to(self, new_state: RequestState) -> None:
        """Move to `new_state`, or raise InvalidStateTransition.

        Beyond the transition table there is one invariant: a request may
        only complete once its decode work is actually finished. Without it
        a scheduler bug could mark a half-decoded request COMPLETED, and the
        simulator would under-report decode work while looking healthy.
        """
        allowed = ALLOWED_TRANSITIONS[self.state]
        if new_state not in allowed:
            raise InvalidStateTransition(
                f"Cannot move request {self.request_id} from "
                f"{self.state.value!r} to {new_state.value!r}. "
                f"Allowed: {sorted(s.value for s in allowed) or 'none (terminal state)'}."
            )

        if new_state is RequestState.COMPLETED and not self.is_decode_complete:
            raise InvalidStateTransition(
                f"Cannot complete request {self.request_id}: decode is "
                f"unfinished ({self.generated_tokens}/"
                f"{self.target_completion_tokens} tokens generated)."
            )

        self.state = new_state

    # --- Decode progress ---------------------------------------------------

    @property
    def remaining_tokens(self) -> int:
        """Decode work still outstanding."""
        return self.target_completion_tokens - self.generated_tokens

    @property
    def is_decode_complete(self) -> bool:
        """True once the target is reached, so the request may be COMPLETED."""
        return self.generated_tokens >= self.target_completion_tokens

    def add_generated_tokens(self, count: int) -> None:
        """Record decode progress. Only valid while DECODING.

        Pure bookkeeping for the scheduler - no real token generation or
        timing happens here, and reaching the target does
        **not** transition the request. The scheduler stays responsible for
        observing `is_decode_complete` and calling transition_to(COMPLETED)
        itself, which keeps doing work separate from changing state.

        A lifecycle violation raises InvalidStateTransition; bad numbers
        raise ValueError.
        """
        if self.state is not RequestState.DECODING:
            raise InvalidStateTransition(
                f"Cannot generate tokens for request {self.request_id} in "
                f"state {self.state.value!r}: only "
                f"{RequestState.DECODING.value!r} may record decode progress."
            )
        if count < 0:
            raise ValueError(f"Generated token count cannot be negative: {count}")
        if self.generated_tokens + count > self.target_completion_tokens:
            raise ValueError(
                f"Generated tokens would exceed the target: "
                f"{self.generated_tokens} + {count} > {self.target_completion_tokens}"
            )
        self.generated_tokens += count
