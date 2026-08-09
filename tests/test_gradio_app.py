"""Step 4 — Gradio demo smoke tests (no server launch, no API key)."""

from __future__ import annotations

import pytest

pytest.importorskip("gradio")

from src.gradio_app import build_demo, run_screen


def test_demo_builds():
    demo = build_demo()
    assert demo is not None


def test_run_screen_blank_name_returns_error():
    assert "error" in run_screen("", "", "")


def test_run_screen_no_match_short_circuits_without_key():
    # True negative from the eval set: filter finds nothing, so no client is built.
    out = run_screen("Oluwaseun Adeyemi", "1991-03-04", "Nigeria")
    assert out["match_tier"] == "NO_MATCH"
    assert out["confidence"] == 0.0
