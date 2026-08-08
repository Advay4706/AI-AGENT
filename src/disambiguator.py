"""
Step 3 — LLM disambiguation agent.

Takes the narrowed candidates from Step 2 plus any available counterparty
DOB/nationality and asks Claude to decide whether a name match is a real hit or
a coincidental collision. The decision is returned as a schema-validated
`ScreeningResult`, enforced via forced tool-calling (not free-text parsing).

Design points:
* No-candidate short-circuit — if the deterministic filter returned nothing,
  we return NO_MATCH / 0.0 WITHOUT calling the LLM.
* The confidence calibration anchors are embedded in the system prompt VERBATIM.
* Deterministic guardrails are applied on top of the LLM output (name-only cap,
  NO_MATCH => 0.0, range clamp) so the calibration rules hold even if the model
  drifts.
* The Anthropic client is injectable so the agent is fully unit-testable with a
  mock and needs no API key until it actually runs live.
"""

from __future__ import annotations

from typing import Optional, Sequence

from src.candidate_filter import Candidate, find_candidates
from src.schema import ScreeningResult

# Default to the latest, most capable Claude model. Overridable per call.
DEFAULT_MODEL = "claude-opus-5"
MAX_TOKENS = 1024
TOOL_NAME = "submit_screening_result"

# --- Confidence calibration anchors — EMBEDDED VERBATIM, do not paraphrase -----
CALIBRATION_ANCHORS = """\
- 0.90+: name matches AND at least one other identifying attribute (DOB, nationality) also matches
- 0.5-0.7: name matches closely, but only partial or missing auxiliary data to confirm
- below 0.3: name matches (exactly or fuzzily) but at least one identifying attribute clearly diverges (different DOB, different nationality)
- A bare name-only string match with no auxiliary data available should never exceed 0.6 confidence"""

SYSTEM_PROMPT = f"""\
You are a sanctions and adverse-media screening analyst. You decide whether a \
transaction counterparty is a genuine match to a watchlist entry, or merely a \
coincidental name collision (a false positive).

You are given the counterparty (name, and optionally date of birth and \
nationality) and a short list of candidate watchlist entries that a deterministic \
filter already flagged as name-similar. Each candidate carries its own name, date \
of birth, nationality, and tier.

Watchlist tiers:
- CONFIRMED_SANCTIONS: the entry is on an official sanctions list.
- ADVERSE_NEWS: the entry is subject to negative media / open investigation, not formally sanctioned.
- NAME_SIMILARITY_ONLY: a benign party whose name merely resembles a watchlisted name.

Decide a single match_tier for the counterparty:
- If the counterparty genuinely matches a CONFIRMED_SANCTIONS or ADVERSE_NEWS candidate, return that tier.
- If the name matches a candidate but identifying attributes (DOB, nationality) clearly diverge, it is a coincidental collision: return NAME_SIMILARITY_ONLY.
- If no candidate is a plausible match at all, return NO_MATCH.

Confidence calibration anchors (follow these exactly):
{CALIBRATION_ANCHORS}

Compare the counterparty's DOB and nationality against each candidate's. Cite the \
specific comparison points (which attributes matched, which diverged, or which were \
missing) in your reasoning. Then call the {TOOL_NAME} tool with your decision. \
Always call the tool — never answer in free text."""

SCREENING_TOOL = {
    "name": TOOL_NAME,
    "description": "Submit the final, calibrated screening decision for the counterparty.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "counterparty_name": {
                "type": "string",
                "description": "The counterparty name that was screened.",
            },
            "match_tier": {
                "type": "string",
                "enum": [
                    "CONFIRMED_SANCTIONS",
                    "ADVERSE_NEWS",
                    "NAME_SIMILARITY_ONLY",
                    "NO_MATCH",
                ],
                "description": "The screening tier decision.",
            },
            "confidence": {
                "type": "number",
                "description": "Calibrated confidence 0.0-1.0 per the anchors.",
            },
            "matched_entry": {
                "type": ["string", "null"],
                "description": "The corpus id/name of the matched entry, or null if none.",
            },
            "reasoning": {
                "type": "string",
                "description": "Why this tier and confidence, citing specific DOB/nationality comparison points.",
            },
        },
        "required": [
            "counterparty_name",
            "match_tier",
            "confidence",
            "matched_entry",
            "reasoning",
        ],
        "additionalProperties": False,
    },
}


def _get_client():
    """Lazily construct a real Anthropic client (only when no client is injected)."""
    import anthropic

    return anthropic.Anthropic()


def _format_candidates(candidates: Sequence[Candidate]) -> str:
    lines = []
    for i, c in enumerate(candidates, 1):
        e = c.entry
        lines.append(
            f"{i}. corpus_id={e['id']} | name=\"{e['name']}\" | dob={e['dob']} | "
            f"nationality={e['nationality']} | tier={e['tier']} | "
            f"filter_score={c.combined_score:.2f}"
        )
    return "\n".join(lines)


def _build_user_message(
    name: str,
    dob: Optional[str],
    nationality: Optional[str],
    candidates: Sequence[Candidate],
) -> str:
    return (
        "Counterparty to screen:\n"
        f"- name: \"{name}\"\n"
        f"- dob: {dob if dob else 'NOT PROVIDED'}\n"
        f"- nationality: {nationality if nationality else 'NOT PROVIDED'}\n\n"
        "Candidate watchlist entries:\n"
        f"{_format_candidates(candidates)}\n\n"
        f"Decide the match_tier and confidence, then call {TOOL_NAME}."
    )


def _extract_tool_input(response) -> dict:
    for block in response.content:
        if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == TOOL_NAME:
            return dict(block.input)
    raise ValueError(f"model response did not contain a '{TOOL_NAME}' tool_use block")


def _apply_guardrails(
    result: ScreeningResult,
    dob: Optional[str],
    nationality: Optional[str],
) -> ScreeningResult:
    """Enforce the calibration rules deterministically on top of the LLM output."""
    conf = max(0.0, min(1.0, result.confidence))

    if result.match_tier == "NO_MATCH":
        conf = 0.0
        matched = None
    else:
        matched = result.matched_entry
        # "A bare name-only string match with no auxiliary data available should
        # never exceed 0.6 confidence."
        name_only = not dob and not nationality
        if name_only:
            conf = min(conf, 0.6)

    return result.model_copy(update={"confidence": conf, "matched_entry": matched})


def disambiguate(
    name: str,
    candidates: Sequence[Candidate],
    dob: Optional[str] = None,
    nationality: Optional[str] = None,
    client=None,
    model: str = DEFAULT_MODEL,
) -> ScreeningResult:
    """Disambiguate `name` against the narrowed `candidates` using Claude.

    Returns a schema-validated `ScreeningResult`. If `candidates` is empty, returns
    NO_MATCH / 0.0 without any LLM call.
    """
    if not candidates:
        return ScreeningResult(
            counterparty_name=name,
            match_tier="NO_MATCH",
            confidence=0.0,
            matched_entry=None,
            reasoning="No corpus candidates cleared the deterministic filter, so there is nothing to disambiguate.",
        )

    if client is None:
        client = _get_client()

    response = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        thinking={"type": "disabled"},
        system=SYSTEM_PROMPT,
        tools=[SCREENING_TOOL],
        tool_choice={"type": "tool", "name": TOOL_NAME},
        messages=[
            {"role": "user", "content": _build_user_message(name, dob, nationality, candidates)}
        ],
    )

    tool_input = _extract_tool_input(response)
    # Pin the screened name to the actual query (don't trust the model to echo it).
    tool_input["counterparty_name"] = name
    result = ScreeningResult.model_validate(tool_input)  # raises on schema violation
    return _apply_guardrails(result, dob, nationality)


def screen(
    name: str,
    dob: Optional[str] = None,
    nationality: Optional[str] = None,
    client=None,
    top_n: int = 5,
    model: str = DEFAULT_MODEL,
) -> ScreeningResult:
    """Full two-stage pipeline: deterministic filter (Step 2) -> LLM disambiguation (Step 3)."""
    candidates = find_candidates(name, top_n=top_n)
    return disambiguate(name, candidates, dob=dob, nationality=nationality, client=client, model=model)
