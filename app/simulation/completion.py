"""Simulated completion: how long the output is, and what it contains.

There is no real model here, so two things are simulated:

1. **How many tokens the model would emit.** Derived deterministically from
   the prompt size using a per-category ratio and cap (see app/config.py).
   Deterministic rather than random, so the same request always produces the
   same result and the tests can assert on it.

2. **The text itself.** Deterministic placeholder content. The text is built to
   the target size, so the completion-token count reported by the API is
   measured from the real returned string rather than asserted.

Both are plain functions with no external model dependency.
"""

from __future__ import annotations

from app.config import CHARS_PER_TOKEN, COMPLETION_RULES

# Placeholder sentences per category. Cycled deterministically and numbered,
# so longer completions read like a structured answer rather than one
# sentence copied repeatedly. Content quality is explicitly not graded.
SENTENCE_POOLS: dict[str, list[str]] = {
    "summarization": [
        "The source material is condensed into its main points.",
        "Key decisions and their owners are listed in order of importance.",
        "Open questions are separated from settled items.",
        "Numbers quoted in the original are carried over unchanged.",
        "Items requiring follow-up are called out explicitly.",
        "Background detail that does not affect the outcome is dropped.",
        "The summary keeps the original ordering where it aids comprehension.",
        "Risks raised in the source are grouped together at the end.",
    ],
    "document_understanding": [
        "The answer is taken directly from the relevant section.",
        "The supporting clause is quoted to show where the answer comes from.",
        "Conditions and thresholds stated in the document are preserved.",
        "Where the document is silent, that is stated rather than inferred.",
        "Any figures cited are reproduced exactly as written.",
        "Cross-references to other sections are noted where relevant.",
    ],
    "customer_support": [
        "The reported issue is acknowledged and restated for confirmation.",
        "The most likely cause is identified from the details provided.",
        "Concrete next steps are given in the order they should be attempted.",
        "Account-specific information from the request is taken into account.",
        "If the steps do not help, the escalation path is explained.",
        "An expected timeframe is given where one can reasonably be offered.",
        "Previous troubleshooting already attempted is not repeated.",
    ],
}

OPENINGS: dict[str, str] = {
    "summarization": "Simulated summary for prompt {prompt_id}.",
    "document_understanding": "Simulated document answer for prompt {prompt_id}.",
    "customer_support": "Simulated support response for prompt {prompt_id}.",
}


def target_completion_tokens(prompt_tokens: int, category: str) -> int:
    """How many completion tokens this request is assumed to produce.

    target = min(max_tokens, round(base_tokens + prompt_tokens * ratio))

    base_tokens is the answer work a request costs even for a tiny prompt, so
    it doubles as the floor and no separate minimum bound is needed. The base,
    ratio and cap are simulation assumptions held in app/config.py.

    Note that Python's round() is half-to-even, which only matters for exact
    .5 values and is deterministic either way.
    """
    rule = COMPLETION_RULES[category]
    scaled = round(rule["base_tokens"] + prompt_tokens * rule["ratio"])
    return min(int(rule["max_tokens"]), scaled)


def build_content(category: str, prompt_id: int, target_tokens: int) -> str:
    """Build deterministic placeholder text of approximately `target_tokens`.

    The text is grown to the character length that the tokenizer
    approximation maps to the target, then trimmed on a word boundary. The
    caller re-measures the finished string with estimate_tokens(), so the
    reported completion-token count always describes the text actually
    returned - never a number asserted independently of it.

    Note the coupling: this helper reads CHARS_PER_TOKEN directly in order to
    convert a token target into a character length. It is therefore a second
    place (besides app/tokenizer.py) that depends on the character/token
    assumption, and swapping in a real tokenizer would require adapting this
    sizing logic too. That coupling is accepted deliberately - the simple
    version is appropriate for a simulator whose content is not graded.
    """
    target_chars = max(1, round(target_tokens * CHARS_PER_TOKEN))

    pool = SENTENCE_POOLS[category]
    parts = [OPENINGS[category].format(prompt_id=prompt_id)]
    index = 0
    while len(" ".join(parts)) < target_chars:
        parts.append("({}) {}".format(index + 1, pool[index % len(pool)]))
        index += 1

    text = " ".join(parts)
    if len(text) <= target_chars:
        return text

    # Cut on a word boundary rather than mid-word, which would look like a
    # truncation bug. Both the boundary before and the boundary after the
    # target are considered, and the closer one wins - always cutting at the
    # earlier boundary can drop a long word and undershoot by several tokens.
    before = text.rfind(" ", 0, target_chars + 1)
    after = text.find(" ", target_chars)

    candidates = [len(text) if index == -1 else index for index in (before, after)]
    candidates = [length for length in candidates if length > 0]
    best = min(candidates, key=lambda length: abs(length - target_chars))
    return text[:best].rstrip()
