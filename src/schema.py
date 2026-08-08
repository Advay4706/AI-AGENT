"""Shared structured-output schema for the screening engine.

Every screening result — from the LLM agent, the API, and the eval harness —
validates against `ScreeningResult`. This is the single source of truth for the
output contract.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

MatchTier = Literal[
    "CONFIRMED_SANCTIONS",
    "ADVERSE_NEWS",
    "NAME_SIMILARITY_ONLY",
    "NO_MATCH",
]


class ScreeningResult(BaseModel):
    """The disambiguation verdict for a single counterparty screening."""

    counterparty_name: str
    match_tier: MatchTier
    confidence: float = Field(ge=0.0, le=1.0)  # 0.0-1.0
    matched_entry: Optional[str] = None  # which corpus entry it matched, if any
    reasoning: str  # why this tier/confidence, referencing specific comparison points
