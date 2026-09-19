"""Tests for the simulated completion length and content."""

import pytest

from app.config import COMPLETION_RULES
from app.simulation.completion import build_content, target_completion_tokens
from app.tokenizer import estimate_tokens

CATEGORIES = list(COMPLETION_RULES)


# --- Completion length -----------------------------------------------------
#
# target = min(max_tokens, round(base_tokens + prompt_tokens * ratio))


@pytest.mark.parametrize("category", CATEGORIES)
def test_base_applies_to_zero_and_tiny_prompts(category: str):
    """base_tokens is the floor - no separate minimum bound exists."""
    rule = COMPLETION_RULES[category]
    assert target_completion_tokens(0, category) == rule["base_tokens"]
    # A 1-token prompt adds less than half a token, so it still rounds to base.
    assert target_completion_tokens(1, category) == rule["base_tokens"]


@pytest.mark.parametrize("category", CATEGORIES)
def test_maximum_cap_applies_to_huge_prompts(category: str):
    rule = COMPLETION_RULES[category]
    assert target_completion_tokens(1_000_000, category) == rule["max_tokens"]


@pytest.mark.parametrize("category", CATEGORIES)
def test_proportional_growth_below_the_cap(category: str):
    """Below the cap, the target is base + prompt_tokens * ratio."""
    rule = COMPLETION_RULES[category]
    prompt_tokens = int((rule["max_tokens"] - rule["base_tokens"]) / rule["ratio"] / 2)
    expected = round(rule["base_tokens"] + prompt_tokens * rule["ratio"])
    assert rule["base_tokens"] < expected < rule["max_tokens"]
    assert target_completion_tokens(prompt_tokens, category) == expected


@pytest.mark.parametrize("category", CATEGORIES)
def test_output_grows_with_input_below_the_cap(category: str):
    """The floor no longer flattens small and medium prompts together."""
    rule = COMPLETION_RULES[category]
    small = target_completion_tokens(50, category)
    medium = target_completion_tokens(500, category)
    assert small > 0
    assert medium > small, (category, small, medium)
    assert medium < rule["max_tokens"]


def test_specific_known_values():
    """The exact formulas, spelled out."""
    # summarization: base 20, ratio 0.12, cap 300
    assert target_completion_tokens(0, "summarization") == 20
    assert target_completion_tokens(100, "summarization") == 32  # 20 + 12
    assert target_completion_tokens(1000, "summarization") == 140  # 20 + 120
    assert target_completion_tokens(5000, "summarization") == 300  # 620 -> cap
    # document_understanding: base 12, ratio 0.05, cap 180
    assert target_completion_tokens(0, "document_understanding") == 12
    assert target_completion_tokens(100, "document_understanding") == 17  # 12 + 5
    assert target_completion_tokens(2000, "document_understanding") == 112
    assert target_completion_tokens(10000, "document_understanding") == 180
    # customer_support: base 20, ratio 0.10, cap 220
    assert target_completion_tokens(0, "customer_support") == 20
    assert target_completion_tokens(100, "customer_support") == 30  # 20 + 10
    assert target_completion_tokens(1000, "customer_support") == 120
    assert target_completion_tokens(5000, "customer_support") == 220


@pytest.mark.parametrize("category", CATEGORIES)
def test_cap_is_reached_exactly_at_the_expected_prompt_size(category: str):
    rule = COMPLETION_RULES[category]
    at_cap = (rule["max_tokens"] - rule["base_tokens"]) / rule["ratio"]
    assert target_completion_tokens(int(at_cap) + 1, category) == rule["max_tokens"]
    assert target_completion_tokens(int(at_cap) - 20, category) < rule["max_tokens"]


@pytest.mark.parametrize("category", CATEGORIES)
def test_is_deterministic(category: str):
    values = {target_completion_tokens(777, category) for _ in range(5)}
    assert len(values) == 1


@pytest.mark.parametrize("category", CATEGORIES)
def test_is_monotonic_non_decreasing(category: str):
    previous = 0
    for prompt_tokens in range(0, 6000, 50):
        current = target_completion_tokens(prompt_tokens, category)
        assert current >= previous
        previous = current


# --- Content ---------------------------------------------------------------


@pytest.mark.parametrize("category", CATEGORIES)
def test_content_is_close_to_target_size(category: str):
    """The text is really built to the target, not merely labelled with it."""
    for target in (12, 20, 30, 50, 100, 180, 300):
        content = build_content(category, 1, target)
        actual = estimate_tokens(content)
        assert abs(actual - target) <= 2, (category, target, actual)


@pytest.mark.parametrize("category", CATEGORIES)
def test_content_is_deterministic(category: str):
    assert build_content(category, 42, 100) == build_content(category, 42, 100)


@pytest.mark.parametrize("category", CATEGORIES)
def test_content_mentions_prompt_id_and_category_wording(category: str):
    content = build_content(category, 314, 60)
    assert "314" in content
    assert content.startswith("Simulated")


def test_content_differs_between_categories():
    texts = {build_content(category, 1, 80) for category in CATEGORIES}
    assert len(texts) == len(CATEGORIES)


@pytest.mark.parametrize("category", CATEGORIES)
def test_content_does_not_end_mid_word(category: str):
    content = build_content(category, 7, 95)
    assert not content.endswith(" ")
    assert content == content.rstrip()
