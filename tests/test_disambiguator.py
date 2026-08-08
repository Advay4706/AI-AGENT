"""Step 3 unit tests — the LLM disambiguation agent.

All tests use a mock Anthropic client (no API key, no network). They verify the
control flow, request shape, schema enforcement, and calibration guardrails —
not the model's judgment, which is exercised live in Step 5.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.candidate_filter import Candidate, find_candidates
from src.disambiguator import (
    CALIBRATION_ANCHORS,
    TOOL_NAME,
    disambiguate,
    screen,
)
from src.schema import ScreeningResult

CORPUS = json.loads(
    (Path(__file__).resolve().parent.parent / "data" / "corpus.json").read_text(encoding="utf-8")
)
CORPUS_BY_ID = {e["id"]: e for e in CORPUS["entries"]}


# ---------------------------------------------------------------------------
# Mock Anthropic client
# ---------------------------------------------------------------------------
class _Block:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Response:
    def __init__(self, content):
        self.content = content


class FakeClient:
    """Stands in for anthropic.Anthropic(). Records the last create() kwargs."""

    def __init__(self, tool_input: dict):
        self._tool_input = tool_input
        self.messages = self
        self.calls = 0
        self.last_kwargs: dict = {}

    def create(self, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        block = _Block(type="tool_use", name=TOOL_NAME, input=dict(self._tool_input))
        # Include a stray text block to prove extraction picks the tool block.
        return _Response([_Block(type="text", text="ignore me"), block])


class ExplodingClient:
    """Fails if the LLM is called — used to prove the no-candidate short-circuit."""

    def __init__(self):
        self.messages = self

    def create(self, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("LLM should not be called")


def _candidate(corpus_id: str, score: float = 0.9) -> Candidate:
    return Candidate(entry=CORPUS_BY_ID[corpus_id], combined_score=score, fuzzy_score=score, phonetic_score=score)


def _tool_input(tier, confidence, matched="SANC-001", name="X", reasoning="because"):
    return {
        "counterparty_name": name,
        "match_tier": tier,
        "confidence": confidence,
        "matched_entry": matched,
        "reasoning": reasoning,
    }


# ---------------------------------------------------------------------------
# No-candidate short-circuit
# ---------------------------------------------------------------------------
class TestNoCandidateShortCircuit:
    def test_empty_candidates_returns_no_match_without_llm(self):
        result = disambiguate("Nobody Here", candidates=[], client=ExplodingClient())
        assert result.match_tier == "NO_MATCH"
        assert result.confidence == 0.0
        assert result.matched_entry is None
        assert result.counterparty_name == "Nobody Here"

    def test_screen_no_match_name_never_calls_llm(self):
        # 'Oluwaseun Adeyemi' is a Step-1 true negative; the filter returns nothing.
        result = screen("Oluwaseun Adeyemi", dob="1991-03-04", nationality="Nigeria", client=ExplodingClient())
        assert result.match_tier == "NO_MATCH"
        assert result.confidence == 0.0


# ---------------------------------------------------------------------------
# Schema enforcement via tool-calling
# ---------------------------------------------------------------------------
class TestSchemaEnforcement:
    def test_valid_tool_output_parses_into_screening_result(self):
        client = FakeClient(_tool_input("CONFIRMED_SANCTIONS", 0.95, "SANC-001"))
        result = disambiguate(
            "Elena Volkov", [_candidate("SANC-001")],
            dob="1968-04-15", nationality="Russia", client=client,
        )
        assert isinstance(result, ScreeningResult)
        assert result.match_tier == "CONFIRMED_SANCTIONS"
        assert result.matched_entry == "SANC-001"
        assert client.calls == 1

    def test_out_of_range_confidence_is_rejected(self):
        client = FakeClient(_tool_input("CONFIRMED_SANCTIONS", 1.5, "SANC-001"))
        with pytest.raises(ValidationError):
            disambiguate("Elena Volkov", [_candidate("SANC-001")], dob="1968-04-15", client=client)

    def test_counterparty_name_is_pinned_to_query(self):
        # Model echoes a wrong name; the agent overrides it with the actual query.
        client = FakeClient(_tool_input("ADVERSE_NEWS", 0.9, "NEWS-001", name="WRONG NAME"))
        result = disambiguate("Jose Garcia", [_candidate("NEWS-001")], dob="1979-11-05", nationality="Mexico", client=client)
        assert result.counterparty_name == "Jose Garcia"


# ---------------------------------------------------------------------------
# Request shape
# ---------------------------------------------------------------------------
class TestRequestShape:
    def test_calibration_anchors_embedded_verbatim(self):
        client = FakeClient(_tool_input("NAME_SIMILARITY_ONLY", 0.15, "SIM-001"))
        disambiguate("John Smith", [_candidate("SIM-001")], dob="1990-11-02", nationality="Canada", client=client)
        assert CALIBRATION_ANCHORS in client.last_kwargs["system"]

    def test_tool_choice_is_forced(self):
        client = FakeClient(_tool_input("NAME_SIMILARITY_ONLY", 0.15, "SIM-001"))
        disambiguate("John Smith", [_candidate("SIM-001")], client=client)
        assert client.last_kwargs["tool_choice"] == {"type": "tool", "name": TOOL_NAME}
        assert client.last_kwargs["tools"][0]["name"] == TOOL_NAME

    def test_candidate_attributes_present_in_prompt(self):
        client = FakeClient(_tool_input("CONFIRMED_SANCTIONS", 0.95, "SANC-001"))
        disambiguate("Elena Volkov", [_candidate("SANC-001")], dob="1968-04-15", nationality="Russia", client=client)
        user_msg = client.last_kwargs["messages"][0]["content"]
        assert "SANC-001" in user_msg and "1968-04-15" in user_msg and "Russia" in user_msg


# ---------------------------------------------------------------------------
# Calibration guardrails (applied on top of the LLM output)
# ---------------------------------------------------------------------------
class TestGuardrails:
    def test_name_only_confidence_capped_at_0_6(self):
        # No DOB, no nationality -> even if the model says 0.95, cap to 0.6.
        client = FakeClient(_tool_input("CONFIRMED_SANCTIONS", 0.95, "SANC-002"))
        result = disambiguate("Mohammed Al-Rashid", [_candidate("SANC-002")], client=client)
        assert result.confidence == 0.6

    def test_partial_data_not_capped_to_name_only_rule(self):
        # Nationality provided -> not "name-only", so 0.65 is allowed through.
        client = FakeClient(_tool_input("CONFIRMED_SANCTIONS", 0.65, "SANC-001"))
        result = disambiguate("Elena Volkov", [_candidate("SANC-001")], nationality="Russia", client=client)
        assert result.confidence == 0.65

    def test_no_match_tier_forces_zero_confidence_and_null_entry(self):
        client = FakeClient(_tool_input("NO_MATCH", 0.4, "SANC-001"))
        result = disambiguate("Someone Else", [_candidate("SANC-001")], dob="2000-01-01", client=client)
        assert result.confidence == 0.0
        assert result.matched_entry is None

    def test_full_match_high_confidence_passes_through(self):
        client = FakeClient(_tool_input("CONFIRMED_SANCTIONS", 0.95, "SANC-001"))
        result = disambiguate("Elena Volkov", [_candidate("SANC-001")], dob="1968-04-15", nationality="Russia", client=client)
        assert result.confidence == 0.95


# ---------------------------------------------------------------------------
# Full pipeline integration (Step 2 filter + mocked Step 3)
# ---------------------------------------------------------------------------
class TestScreenIntegration:
    def test_screen_runs_filter_then_disambiguates(self):
        client = FakeClient(_tool_input("CONFIRMED_SANCTIONS", 0.95, "SANC-001"))
        result = screen("Elena Volkov", dob="1968-04-15", nationality="Russia", client=client)
        assert result.match_tier == "CONFIRMED_SANCTIONS"
        assert client.calls == 1

    def test_screen_passes_real_candidates_from_filter(self):
        # Confirm the candidate the filter found is what reaches the prompt.
        assert any(c.id == "SANC-001" for c in find_candidates("Elena Volkov"))
        client = FakeClient(_tool_input("CONFIRMED_SANCTIONS", 0.95, "SANC-001"))
        screen("Elena Volkov", dob="1968-04-15", nationality="Russia", client=client)
        assert "SANC-001" in client.last_kwargs["messages"][0]["content"]
