"""Simulation coefficients.

Every number here is an **assumed simulation parameter**, not a measured fact
and not a property of any real tokenizer or model. They live in one place so
the simulation's assumptions can be changed, reviewed or later calibrated
without hunting through the code.

Nothing here is a configuration framework on purpose: plain module-level
constants are enough, and they stay easy to read in an interview.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Token approximation
# --------------------------------------------------------------------------

# ASSUMED. Average characters per token. Roughly in line with the widely
# quoted "~4 characters per token" rule of thumb for English text, but it is
# used here as a simulation coefficient, not as a claim about any specific
# tokenizer. Code that estimates token counts reads this shared coefficient.
CHARS_PER_TOKEN = 4.0


# --------------------------------------------------------------------------
# Completion length model
# --------------------------------------------------------------------------
#
# We have no real model generating text, so output length is simulated from
# input length:
#
#     target = min(max_tokens, round(base_tokens + prompt_tokens * ratio))
#
# The three terms each model something:
#   - base_tokens: the answer work a request costs even when the prompt is
#     tiny. This also acts as the floor, so no separate minimum is needed for
#     non-negative prompt sizes.
#   - ratio * prompt_tokens: workload-dependent growth of the output.
#   - max_tokens: output cannot grow without limit just because the source
#     document is enormous.
#
# ASSUMED values, chosen for plausible workload shape rather than measured:
#   - summarization output grows most with input, but stays much shorter than
#     the source material
#   - document understanding answers a specific question, so it is shortest
#     and flattest
#   - customer support sits in between
#
# These are simulation assumptions, not measured model behavior, and are not
# derived from any benchmark. The base + ratio form keeps output varying with
# input across the dataset.

COMPLETION_RULES: dict[str, dict[str, float | int]] = {
    "summarization": {"base_tokens": 20, "ratio": 0.12, "max_tokens": 300},
    "document_understanding": {"base_tokens": 12, "ratio": 0.05, "max_tokens": 180},
    "customer_support": {"base_tokens": 20, "ratio": 0.10, "max_tokens": 220},
}


# --------------------------------------------------------------------------
# Simulated GPU
# --------------------------------------------------------------------------
#
# ASSUMED SIMULATION PARAMETERS. These are **not** measurements of an A100,
# H100, B200 or any other hardware, and not a benchmark of any model. They are
# chosen to produce plausible *relative* behavior:
#
#   - a longer prompt costs more prefill time
#   - a longer completion costs more decode time
#   - a decode token is far more expensive than a prefill token, because
#     prefill processes the whole prompt in parallel while decode emits
#     tokens one at a time
#
# The 50x gap between the two throughputs is what encodes that last point.
# They can be replaced with calibrated values if measurements become available.

DEFAULT_GPU_NAME = "simulated-default-gpu"
DEFAULT_PREFILL_TOKENS_PER_SECOND = 5000.0
DEFAULT_DECODE_TOKENS_PER_SECOND = 100.0
DEFAULT_FIXED_OVERHEAD_SECONDS = 0.010


# --------------------------------------------------------------------------
# Continuous batching
# --------------------------------------------------------------------------

# ASSUMED SIMULATOR PARAMETER. How many requests may occupy the simulated
# GPU scheduler at once. This is a property of *our simulation*, not a real
# vLLM setting and not a limit derived from any GPU's memory or kernels.
#
# In a real engine the practical ceiling comes mostly from KV-cache capacity,
# which this simulator does not model, so the number is simply fixed.
MAX_ACTIVE_REQUESTS = 4
