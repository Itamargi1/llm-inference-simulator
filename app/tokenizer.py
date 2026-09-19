"""Token approximation.

This simulator uses a deliberately simple character-count approximation:

    tokens = ceil(characters / CHARS_PER_TOKEN)

Runtime token *estimation* is isolated here: every caller that needs to count
tokens goes through estimate_tokens().

Note, however, that this is not the only place the character/token assumption
is used. app/simulation/completion.py also reads CHARS_PER_TOKEN to size the
synthetic placeholder text it builds, so that the generated content really is
about as long as the target it claims. Replacing this approximation with a
real tokenizer would therefore mean adapting that sizing helper as well, not
just this file.
"""

from __future__ import annotations

import math

from app.config import CHARS_PER_TOKEN


def estimate_tokens(text: str) -> int:
    """Approximate the token count of a piece of text.

    Empty text is 0 tokens; any non-empty text is at least 1, since ceil()
    of any positive value is at least 1.
    """
    if not text:
        return 0
    return math.ceil(len(text) / CHARS_PER_TOKEN)
