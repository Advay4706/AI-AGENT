"""
Step 4 — FastAPI service.

POST /screen accepts a counterparty name plus optional DOB/nationality and
returns a `ScreeningResult` produced by the full two-stage pipeline
(deterministic filter -> LLM disambiguation).

The Anthropic client is resolved through a FastAPI dependency so it can be
overridden in tests (no key/network needed). At runtime the dependency returns
None, which makes `screen()` construct a real client lazily — and if the key is
missing or the LLM call fails, the endpoint responds with 503 instead of a 500.
"""

from __future__ import annotations

from typing import Optional

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.disambiguator import DEFAULT_MODEL, screen
from src.schema import ScreeningResult

app = FastAPI(
    title="Sanctions & Adverse-Media Name-Match Disambiguation Engine",
    description="Screens counterparty names against a tiered watchlist and decides "
    "whether a match is a real hit or a coincidental collision.",
    version="1.0.0",
)


class ScreenRequest(BaseModel):
    name: str = Field(..., min_length=1, description="Counterparty name to screen.")
    dob: Optional[str] = Field(None, description="Date of birth, ISO YYYY-MM-DD (optional).")
    nationality: Optional[str] = Field(None, description="Nationality (optional).")
    top_n: int = Field(5, ge=1, le=25, description="Max deterministic candidates to consider.")


def get_screening_client():
    """Injectable client seam. Returns None at runtime so screen() builds the real one.

    Tests override this dependency to supply a mock client.
    """
    return None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": DEFAULT_MODEL}


@app.post("/screen", response_model=ScreeningResult)
def screen_counterparty(
    req: ScreenRequest,
    client=Depends(get_screening_client),
) -> ScreeningResult:
    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="name must not be blank")
    try:
        return screen(
            name,
            dob=req.dob,
            nationality=req.nationality,
            client=client,
            top_n=req.top_n,
        )
    except Exception as exc:  # missing API key, auth failure, upstream LLM error
        raise HTTPException(
            status_code=503,
            detail=f"Screening backend unavailable: {type(exc).__name__}: {exc}",
        ) from exc
