"""API request and response models.

Kept minimal on purpose. Latency, queue time, GPU assignment, request state
and reasoning tokens are internal concerns and are deliberately absent from
the public generation contract.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """Input to POST /generate."""

    prompt_id: int = Field(
        ...,
        description="ID of a prompt in the loaded dataset.",
        examples=[123],
    )


class GenerateResponse(BaseModel):
    """Result of a simulated inference request.

    Internal lifecycle state is deliberately not exposed: by the time a
    response exists the request has completed, so reporting `state` would add
    nothing. Queue time, latency and GPU assignment are available only through
    aggregate simulator metrics where applicable.
    """

    request_id: str = Field(
        ...,
        description="Unique ID for this invocation. Two calls with the same "
        "prompt_id are two different requests.",
    )
    prompt_id: int = Field(
        ..., description="Echoed so a response can be matched to its request."
    )
    content: str = Field(..., description="Simulated generated content.")
    prompt_tokens: int = Field(
        ..., description="Approximate token count of the dataset prompt."
    )
    completion_tokens: int = Field(
        ..., description="Approximate token count of the returned content."
    )
