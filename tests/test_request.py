"""Tests for the simulated request lifecycle."""

import pytest

from app.simulation.request import (
    ALLOWED_TRANSITIONS,
    TERMINAL_STATES,
    InvalidStateTransition,
    RequestState,
    SimulatedRequest,
)

HAPPY_PATH = [
    RequestState.TOKENIZED,
    RequestState.WAITING,
    RequestState.PREFILL,
    RequestState.DECODING,
    RequestState.COMPLETED,
]


def make_request(target: int = 100) -> SimulatedRequest:
    return SimulatedRequest(
        prompt_id=1,
        category="summarization",
        prompt_tokens=400,
        target_completion_tokens=target,
    )


def advance_to(request: SimulatedRequest, state: RequestState) -> SimulatedRequest:
    """Walk the happy path until `state` is reached (pre-COMPLETED only)."""
    for next_state in HAPPY_PATH:
        if request.state == state:
            break
        request.transition_to(next_state)
    return request


def decoding_request(target: int = 100) -> SimulatedRequest:
    """A request sitting in DECODING, ready to record decode progress."""
    return advance_to(make_request(target), RequestState.DECODING)


# --- Normal path -----------------------------------------------------------


def test_starts_in_received():
    assert make_request().state == RequestState.RECEIVED


def test_full_happy_path():
    """The real lifecycle: decode work must be done before completing."""
    request = make_request(target=100)

    request.transition_to(RequestState.TOKENIZED)
    request.transition_to(RequestState.WAITING)
    request.transition_to(RequestState.PREFILL)
    request.transition_to(RequestState.DECODING)

    request.add_generated_tokens(100)

    request.transition_to(RequestState.COMPLETED)
    assert request.state == RequestState.COMPLETED
    assert request.is_terminal
    assert request.is_decode_complete


# --- Invalid transitions ---------------------------------------------------


@pytest.mark.parametrize(
    "start, target",
    [
        (RequestState.RECEIVED, RequestState.PREFILL),
        (RequestState.RECEIVED, RequestState.DECODING),
        (RequestState.RECEIVED, RequestState.COMPLETED),
        (RequestState.TOKENIZED, RequestState.DECODING),
        (RequestState.WAITING, RequestState.COMPLETED),
        (RequestState.PREFILL, RequestState.COMPLETED),
        (RequestState.DECODING, RequestState.PREFILL),
        (RequestState.DECODING, RequestState.WAITING),
    ],
)
def test_invalid_skips_are_rejected(start: RequestState, target: RequestState):
    request = advance_to(make_request(), start)
    assert request.state == start
    with pytest.raises(InvalidStateTransition):
        request.transition_to(target)
    # The failed transition must not have changed anything.
    assert request.state == start


def test_error_message_names_both_states():
    request = make_request()
    with pytest.raises(InvalidStateTransition, match="received.*prefill"):
        request.transition_to(RequestState.PREFILL)


def test_cannot_transition_to_itself():
    request = make_request()
    with pytest.raises(InvalidStateTransition):
        request.transition_to(RequestState.RECEIVED)


def test_cannot_go_backwards():
    request = advance_to(make_request(), RequestState.PREFILL)
    with pytest.raises(InvalidStateTransition):
        request.transition_to(RequestState.WAITING)


# --- Terminal states -------------------------------------------------------


@pytest.mark.parametrize("terminal", sorted(TERMINAL_STATES, key=lambda s: s.value))
def test_terminal_states_have_no_outgoing_transitions(terminal: RequestState):
    assert ALLOWED_TRANSITIONS[terminal] == frozenset()


@pytest.mark.parametrize("terminal", sorted(TERMINAL_STATES, key=lambda s: s.value))
def test_nothing_leaves_a_terminal_state(terminal: RequestState):
    request = make_request()
    request.state = terminal  # place directly, to test every exit attempt
    assert request.is_terminal
    for state in RequestState:
        with pytest.raises(InvalidStateTransition):
            request.transition_to(state)
    assert request.state == terminal


def test_completed_cannot_return_to_decoding():
    request = decoding_request(target=10)
    request.add_generated_tokens(10)
    request.transition_to(RequestState.COMPLETED)
    with pytest.raises(InvalidStateTransition):
        request.transition_to(RequestState.DECODING)


# --- Failure and rejection paths -------------------------------------------


@pytest.mark.parametrize(
    "state",
    [
        RequestState.RECEIVED,
        RequestState.TOKENIZED,
        RequestState.WAITING,
        RequestState.PREFILL,
        RequestState.DECODING,
    ],
)
def test_failed_is_reachable_from_every_active_state(state: RequestState):
    request = advance_to(make_request(), state)
    request.transition_to(RequestState.FAILED)
    assert request.state == RequestState.FAILED
    assert request.is_terminal


@pytest.mark.parametrize(
    "state",
    [RequestState.RECEIVED, RequestState.TOKENIZED, RequestState.WAITING],
)
def test_rejected_is_reachable_before_inference_begins(state: RequestState):
    request = advance_to(make_request(), state)
    request.transition_to(RequestState.REJECTED)
    assert request.state == RequestState.REJECTED


@pytest.mark.parametrize("state", [RequestState.PREFILL, RequestState.DECODING])
def test_rejected_is_not_reachable_once_inference_started(state: RequestState):
    """Rejection means turned away, not abandoned midway - that is FAILED."""
    request = advance_to(make_request(), state)
    with pytest.raises(InvalidStateTransition):
        request.transition_to(RequestState.REJECTED)


def test_every_state_has_a_transition_rule():
    assert set(ALLOWED_TRANSITIONS) == set(RequestState)


# --- Generated-token bookkeeping -------------------------------------------


def test_generated_tokens_start_at_zero():
    request = make_request()
    assert request.generated_tokens == 0
    assert request.remaining_tokens == 100
    assert not request.is_decode_complete


def test_adding_tokens_accumulates():
    request = decoding_request(target=100)
    request.add_generated_tokens(30)
    assert request.generated_tokens == 30
    assert request.remaining_tokens == 70
    request.add_generated_tokens(20)
    assert request.generated_tokens == 50
    assert not request.is_decode_complete


def test_adding_zero_is_allowed():
    request = decoding_request()
    request.add_generated_tokens(0)
    assert request.generated_tokens == 0


def test_negative_increment_is_rejected():
    request = decoding_request()
    with pytest.raises(ValueError, match="cannot be negative"):
        request.add_generated_tokens(-1)
    assert request.generated_tokens == 0


def test_exceeding_target_is_rejected():
    request = decoding_request(target=100)
    request.add_generated_tokens(90)
    with pytest.raises(ValueError, match="exceed the target"):
        request.add_generated_tokens(11)
    assert request.generated_tokens == 90


def test_reaching_exact_target_works():
    request = decoding_request(target=100)
    request.add_generated_tokens(100)
    assert request.generated_tokens == 100
    assert request.remaining_tokens == 0
    assert request.is_decode_complete


def test_reaching_target_in_steps():
    """The scheduler may reach the target across multiple decode steps."""
    request = decoding_request(target=10)
    for _ in range(10):
        assert not request.is_decode_complete
        request.add_generated_tokens(1)
    assert request.is_decode_complete
    assert request.generated_tokens == 10


def test_reaching_target_does_not_auto_complete():
    """Finishing decode work must not move the state by itself.

    The scheduler observes is_decode_complete and transitions explicitly,
    which keeps doing work separate from changing state.
    """
    request = decoding_request(target=10)
    request.add_generated_tokens(10)
    assert request.is_decode_complete
    assert request.state == RequestState.DECODING
    assert not request.is_terminal


# --- Invariant: tokens may only be generated while DECODING ----------------


@pytest.mark.parametrize(
    "state",
    [
        RequestState.RECEIVED,
        RequestState.TOKENIZED,
        RequestState.WAITING,
        RequestState.PREFILL,
    ],
)
def test_cannot_generate_tokens_before_decoding(state: RequestState):
    request = advance_to(make_request(target=100), state)
    with pytest.raises(InvalidStateTransition, match="only 'decoding'"):
        request.add_generated_tokens(10)
    assert request.generated_tokens == 0
    assert request.state == state


@pytest.mark.parametrize(
    "state",
    [RequestState.COMPLETED, RequestState.REJECTED, RequestState.FAILED],
)
def test_cannot_generate_tokens_in_terminal_states(state: RequestState):
    request = decoding_request(target=100)
    request.add_generated_tokens(100)
    request.state = state  # place directly, including the terminal cases
    with pytest.raises(InvalidStateTransition):
        request.add_generated_tokens(0)
    assert request.generated_tokens == 100


def test_generation_blocked_in_every_non_decoding_state():
    """Exhaustive: only DECODING may record decode progress."""
    for state in RequestState:
        request = make_request(target=100)
        request.state = state
        if state is RequestState.DECODING:
            request.add_generated_tokens(5)
            assert request.generated_tokens == 5
            continue
        with pytest.raises(InvalidStateTransition):
            request.add_generated_tokens(5)
        assert request.generated_tokens == 0


# --- Invariant: decode must finish before COMPLETED ------------------------


def test_cannot_complete_with_no_tokens_generated():
    request = decoding_request(target=100)
    assert request.generated_tokens == 0
    with pytest.raises(InvalidStateTransition, match="decode is unfinished"):
        request.transition_to(RequestState.COMPLETED)
    assert request.state == RequestState.DECODING


def test_cannot_complete_one_token_short():
    request = decoding_request(target=100)
    request.add_generated_tokens(99)
    with pytest.raises(InvalidStateTransition, match=r"99/100"):
        request.transition_to(RequestState.COMPLETED)
    assert request.state == RequestState.DECODING
    assert request.generated_tokens == 99


def test_completes_at_exactly_the_target():
    request = decoding_request(target=100)
    request.add_generated_tokens(80)
    with pytest.raises(InvalidStateTransition):
        request.transition_to(RequestState.COMPLETED)
    assert request.state == RequestState.DECODING

    request.add_generated_tokens(20)
    request.transition_to(RequestState.COMPLETED)
    assert request.state == RequestState.COMPLETED


def test_stepwise_decode_then_complete():
    """Scheduler-style: several partial steps, then an explicit completion."""
    request = decoding_request(target=10)
    for step in (3, 3, 4):
        request.add_generated_tokens(step)
    assert request.generated_tokens == 10
    request.transition_to(RequestState.COMPLETED)
    assert request.state == RequestState.COMPLETED


def test_unfinished_decode_can_still_fail():
    """The completion invariant must not trap a request that failed."""
    request = decoding_request(target=100)
    request.add_generated_tokens(40)
    request.transition_to(RequestState.FAILED)
    assert request.state == RequestState.FAILED


# --- Identity --------------------------------------------------------------


def test_each_request_gets_a_unique_id():
    ids = {make_request().request_id for _ in range(100)}
    assert len(ids) == 100


def test_same_prompt_id_yields_different_request_ids():
    first, second = make_request(), make_request()
    assert first.prompt_id == second.prompt_id
    assert first.request_id != second.request_id


def test_request_id_is_a_string():
    assert isinstance(make_request().request_id, str)
    assert len(make_request().request_id) == 36  # uuid4 with hyphens


# --- Domain isolation ------------------------------------------------------


def test_module_has_no_web_framework_dependency():
    """The state machine is simulation logic, not an HTTP-aware object.

    Checked against the actual import statements rather than the file text,
    so prose in the docstrings does not trigger a false failure.
    """
    import ast

    import app.simulation.request as module

    with open(module.__file__, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    assert imported.isdisjoint({"fastapi", "pydantic", "starlette"}), imported
