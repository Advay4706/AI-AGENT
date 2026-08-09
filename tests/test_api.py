"""Step 4 tests — the FastAPI service.

The Anthropic client is overridden with a mock via the get_screening_client
dependency, so these run with no API key and no network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import app, get_screening_client
from src.disambiguator import TOOL_NAME

CORPUS = json.loads(
    (Path(__file__).resolve().parent.parent / "data" / "corpus.json").read_text(encoding="utf-8")
)


# --- Mock Anthropic client (mirrors test_disambiguator) --------------------
class _Block:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Response:
    def __init__(self, content):
        self.content = content


class FakeClient:
    def __init__(self, tool_input: dict):
        self._tool_input = tool_input
        self.messages = self

    def create(self, **kwargs):
        block = _Block(type="tool_use", name=TOOL_NAME, input=dict(self._tool_input))
        return _Response([block])


def _tool_input(tier, confidence, matched, name="X", reasoning="because"):
    return {
        "counterparty_name": name,
        "match_tier": tier,
        "confidence": confidence,
        "matched_entry": matched,
        "reasoning": reasoning,
    }


@pytest.fixture
def client():
    return TestClient(app)


def _override(tool_input: dict):
    app.dependency_overrides[get_screening_client] = lambda: FakeClient(tool_input)


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
class TestHealth:
    def test_health_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


class TestScreenEndpoint:
    def test_true_positive_returns_confirmed(self, client):
        _override(_tool_input("CONFIRMED_SANCTIONS", 0.95, "SANC-001"))
        r = client.post("/screen", json={"name": "Elena Volkov", "dob": "1968-04-15", "nationality": "Russia"})
        assert r.status_code == 200
        body = r.json()
        assert body["match_tier"] == "CONFIRMED_SANCTIONS"
        assert body["counterparty_name"] == "Elena Volkov"
        assert body["matched_entry"] == "SANC-001"
        assert 0.0 <= body["confidence"] <= 1.0

    def test_false_positive_name_only(self, client):
        _override(_tool_input("NAME_SIMILARITY_ONLY", 0.15, "SIM-001"))
        r = client.post("/screen", json={"name": "John Smith", "dob": "1990-11-02", "nationality": "Canada"})
        assert r.status_code == 200
        assert r.json()["match_tier"] == "NAME_SIMILARITY_ONLY"

    def test_no_match_name_short_circuits_without_client(self, client):
        # No dependency override: 'Oluwaseun Adeyemi' finds no candidates, so the
        # real client is never constructed -> NO_MATCH, no key needed.
        r = client.post("/screen", json={"name": "Oluwaseun Adeyemi", "dob": "1991-03-04", "nationality": "Nigeria"})
        assert r.status_code == 200
        body = r.json()
        assert body["match_tier"] == "NO_MATCH"
        assert body["confidence"] == 0.0

    def test_optional_fields_omitted(self, client):
        _override(_tool_input("CONFIRMED_SANCTIONS", 0.9, "SANC-002"))
        r = client.post("/screen", json={"name": "Mohammed Al-Rashid"})
        assert r.status_code == 200
        # name-only (no dob/nationality) -> guardrail caps confidence at 0.6
        assert r.json()["confidence"] <= 0.6

    def test_response_matches_screening_result_schema(self, client):
        _override(_tool_input("ADVERSE_NEWS", 0.9, "NEWS-001"))
        r = client.post("/screen", json={"name": "Jose Garcia", "dob": "1979-11-05", "nationality": "Mexico"})
        body = r.json()
        assert set(body) == {"counterparty_name", "match_tier", "confidence", "matched_entry", "reasoning"}


class TestValidation:
    def test_missing_name_is_422(self, client):
        r = client.post("/screen", json={"dob": "1990-01-01"})
        assert r.status_code == 422

    def test_blank_name_is_422(self, client):
        r = client.post("/screen", json={"name": "   "})
        # min_length passes but the handler rejects whitespace-only -> 422
        assert r.status_code == 422

    def test_bad_top_n_is_422(self, client):
        r = client.post("/screen", json={"name": "Elena Volkov", "top_n": 0})
        assert r.status_code == 422


class TestBackendError:
    def test_llm_failure_maps_to_503(self, client):
        class Boom:
            def __init__(self):
                self.messages = self

            def create(self, **kwargs):
                raise RuntimeError("no api key configured")

        app.dependency_overrides[get_screening_client] = lambda: Boom()
        # A name that DOES find candidates, so the client is actually used.
        r = client.post("/screen", json={"name": "Elena Volkov", "dob": "1968-04-15", "nationality": "Russia"})
        assert r.status_code == 503
        assert "unavailable" in r.json()["detail"].lower()
