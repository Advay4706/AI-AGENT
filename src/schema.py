"""The core output contract for the screening engine.

Every screening decision — from the deterministic filter, the LLM agent, the
API, and the evaluation harness — is represented as a `ScreeningResult`. Pydantic
validates it at runtime, so malformed data (an out-of-range confidence, an unknown
tier) fails loudly here instead of silently corrupting a downstream stage.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

# The four possible verdicts. Literal restricts the field to exactly these strings;
# anything else raises a validation error.
MatchTier = Literal[
    "CONFIRMED_SANCTIONS",    # on an official sanctions list
    "ADVERSE_NEWS",           # negative media / open investigation, not sanctioned
    "NAME_SIMILARITY_ONLY",   # benign party whose name merely collides with a listed one
    "NO_MATCH",               # not on any list
]


class ScreeningResult(BaseModel):
    """The disambiguation verdict for a single counterparty screening."""

    counterparty_name: str
    match_tier: MatchTier
    confidence: float = Field(ge=0.0, le=1.0)  # calibrated 0.0-1.0
    matched_entry: Optional[str] = None        # which corpus entry matched, if any
    reasoning: str                             # why this tier/confidence
