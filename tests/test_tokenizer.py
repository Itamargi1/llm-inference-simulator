"""Tests for the character-based token approximation."""

import pytest

from app.config import CHARS_PER_TOKEN
from app.tokenizer import estimate_tokens


@pytest.mark.parametrize(
    "text, expected",
    [
        ("", 0),
        ("a", 1),
        ("abcd", 1),
        ("abcde", 2),
        ("abcdefgh", 2),
        ("abcdefghi", 3),
    ],
)
def test_boundary_cases(text: str, expected: int):
    assert estimate_tokens(text) == expected


def test_longer_strings():
    assert estimate_tokens("x" * 400) == 100
    assert estimate_tokens("x" * 401) == 101
    assert estimate_tokens("x" * 1000) == 250


def test_uses_configured_coefficient():
    """The coefficient lives in config, not hard-coded in the tokenizer."""
    assert CHARS_PER_TOKEN == 4.0
    assert estimate_tokens("x" * int(CHARS_PER_TOKEN)) == 1


def test_is_monotonic_and_deterministic():
    previous = 0
    for length in range(0, 200):
        tokens = estimate_tokens("x" * length)
        assert tokens >= previous
        assert tokens == estimate_tokens("x" * length)
        previous = tokens


def test_whitespace_counts_as_characters():
    """No word splitting is involved - this is purely a length formula."""
    assert estimate_tokens("    ") == 1
