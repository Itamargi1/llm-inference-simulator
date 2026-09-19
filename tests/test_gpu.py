"""Tests for the simulated GPU timing model.

This module is pure - it computes durations and never waits - so every test
here runs instantly. Execution (spending that time, driving request state)
lives in the scheduler and is tested in test_scheduler.py.
"""

import pytest

from app.simulation.gpu import (
    DEFAULT_GPU_PROFILE,
    GPUProfile,
    SimulatedGPU,
    StageTiming,
)
from app.simulation.request import SimulatedRequest


def make_request(prompt_tokens: int = 1000, target: int = 100) -> SimulatedRequest:
    return SimulatedRequest(
        prompt_id=1,
        category="summarization",
        prompt_tokens=prompt_tokens,
        target_completion_tokens=target,
    )


@pytest.fixture
def gpu() -> SimulatedGPU:
    return SimulatedGPU(DEFAULT_GPU_PROFILE)


# --- Profile validation ----------------------------------------------------


def test_default_profile_values():
    """The assumed simulation parameters, pinned."""
    assert DEFAULT_GPU_PROFILE.name == "simulated-default-gpu"
    assert DEFAULT_GPU_PROFILE.prefill_tokens_per_second == 5000.0
    assert DEFAULT_GPU_PROFILE.decode_tokens_per_second == 100.0
    assert DEFAULT_GPU_PROFILE.fixed_overhead_seconds == 0.010


@pytest.mark.parametrize("bad", [0.0, -1.0, -5000.0])
def test_invalid_prefill_throughput_is_rejected(bad: float):
    with pytest.raises(ValueError, match="prefill_tokens_per_second"):
        GPUProfile("x", bad, 100.0, 0.01)


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_invalid_decode_throughput_is_rejected(bad: float):
    with pytest.raises(ValueError, match="decode_tokens_per_second"):
        GPUProfile("x", 5000.0, bad, 0.01)


def test_negative_overhead_is_rejected():
    with pytest.raises(ValueError, match="fixed_overhead_seconds"):
        GPUProfile("x", 5000.0, 100.0, -0.001)


def test_zero_overhead_is_allowed():
    assert GPUProfile("x", 5000.0, 100.0, 0.0).fixed_overhead_seconds == 0.0


# --- Pure timing formulas --------------------------------------------------


def test_worked_example(gpu: SimulatedGPU):
    """prompt 1000, target 100, default profile -> 1.21 s total."""
    timing = gpu.estimate_timing(make_request(prompt_tokens=1000, target=100))
    assert timing.prefill_seconds == pytest.approx(0.2)  # 1000 / 5000
    assert timing.decode_seconds == pytest.approx(1.0)  # 100 / 100
    assert timing.fixed_overhead_seconds == pytest.approx(0.01)
    assert timing.total_seconds == pytest.approx(1.21)


def test_gpu_module_never_waits():
    """The timing model must be pure, so the scheduler can plan with it."""
    import app.simulation.gpu as gpu_module

    assert not hasattr(gpu_module.SimulatedGPU, "run_request")
    assert "time" not in dir(gpu_module)


def test_decode_step_seconds(gpu: SimulatedGPU):
    """One batched decode iteration: 1 / decode_tokens_per_second."""
    assert gpu.decode_step_seconds == pytest.approx(0.01)


def test_prefill_setup_seconds_includes_overhead(gpu: SimulatedGPU):
    request = make_request(prompt_tokens=1000)
    assert gpu.prefill_setup_seconds(request) == pytest.approx(0.21)  # 0.01 + 0.20


def test_isolated_decode_equals_steps_times_step_time(gpu: SimulatedGPU):
    """The batched step model reduces to the isolated model for one request."""
    request = make_request(target=100)
    assert gpu.decode_seconds(request) == pytest.approx(
        request.target_completion_tokens * gpu.decode_step_seconds
    )


def test_longer_prompt_increases_prefill(gpu: SimulatedGPU):
    short = gpu.prefill_seconds(make_request(prompt_tokens=100))
    long = gpu.prefill_seconds(make_request(prompt_tokens=10000))
    assert long > short
    assert long == pytest.approx(short * 100)


def test_longer_target_increases_decode(gpu: SimulatedGPU):
    small = gpu.decode_seconds(make_request(target=10))
    large = gpu.decode_seconds(make_request(target=200))
    assert large > small
    assert large == pytest.approx(small * 20)


def test_prefill_is_independent_of_completion_length(gpu: SimulatedGPU):
    a = gpu.prefill_seconds(make_request(prompt_tokens=800, target=10))
    b = gpu.prefill_seconds(make_request(prompt_tokens=800, target=300))
    assert a == b


def test_decode_is_independent_of_prompt_length(gpu: SimulatedGPU):
    a = gpu.decode_seconds(make_request(prompt_tokens=10, target=120))
    b = gpu.decode_seconds(make_request(prompt_tokens=9000, target=120))
    assert a == b


def test_zero_prompt_tokens_gives_zero_prefill(gpu: SimulatedGPU):
    assert gpu.prefill_seconds(make_request(prompt_tokens=0)) == 0.0


def test_zero_target_gives_zero_decode(gpu: SimulatedGPU):
    assert gpu.decode_seconds(make_request(target=0)) == 0.0


def test_no_negative_stage_durations(gpu: SimulatedGPU):
    for prompt_tokens in (0, 1, 500, 20000):
        for target in (0, 1, 300):
            timing = gpu.estimate_timing(make_request(prompt_tokens, target))
            assert timing.prefill_seconds >= 0
            assert timing.decode_seconds >= 0
            assert timing.total_seconds >= 0


def test_negative_token_counts_are_rejected(gpu: SimulatedGPU):
    bad_prompt = SimulatedRequest(1, "summarization", -5, 100)
    with pytest.raises(ValueError, match="prompt_tokens"):
        gpu.prefill_seconds(bad_prompt)
    bad_target = SimulatedRequest(1, "summarization", 100, -5)
    with pytest.raises(ValueError, match="target_completion_tokens"):
        gpu.decode_seconds(bad_target)


def test_decode_token_is_more_expensive_than_prefill_token(gpu: SimulatedGPU):
    """The key qualitative property the parameters are meant to encode."""
    per_prefill_token = gpu.prefill_seconds(make_request(prompt_tokens=1000)) / 1000
    per_decode_token = gpu.decode_seconds(make_request(target=1000)) / 1000
    assert per_decode_token > per_prefill_token
    assert per_decode_token == pytest.approx(per_prefill_token * 50)


def test_timing_total_is_the_sum_of_stages():
    timing = StageTiming(0.01, 0.2, 1.0)
    assert timing.total_seconds == pytest.approx(1.21)


def test_custom_profile_changes_timing():
    fast = SimulatedGPU(GPUProfile("fast", 10000.0, 200.0, 0.0))
    timing = fast.estimate_timing(make_request(prompt_tokens=1000, target=100))
    assert timing.prefill_seconds == pytest.approx(0.1)
    assert timing.decode_seconds == pytest.approx(0.5)
    assert timing.total_seconds == pytest.approx(0.6)
