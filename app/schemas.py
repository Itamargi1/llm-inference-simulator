"""API request and response models.

Kept minimal on purpose. Request state, queue time and latency are internal
concerns, reported through /metrics rather than per request.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """Input for one simulated inference request."""

    prompt_id: int = Field(
        ...,
        description="ID of the prompt to load from data/prompts.json.",
        examples=[30],
    )


class GenerateResponse(BaseModel):
    """Result returned after one simulated inference request completes.

    Queue and latency information are reported through /metrics rather than
    as per-request response fields.
    """

    request_id: str = Field(..., description="Unique ID for this API call.")
    prompt_id: int = Field(
        ..., description="Dataset prompt ID used for the simulation."
    )
    content: str = Field(
        ...,
        description="Deterministic placeholder content produced by the simulator.",
    )
    prompt_tokens: int = Field(
        ...,
        description="Approximate input-token count using the simulator token formula.",
    )
    completion_tokens: int = Field(
        ...,
        description="Approximate token count of the returned placeholder content.",
    )
