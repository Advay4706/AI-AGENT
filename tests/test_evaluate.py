"""Step 5 tests — the evaluation harness (rule-based stand-in, no key/network)."""

from __future__ import annotations

import pytest

from src.baseline import baseline_predict
from src.candidate_filter import find_candidates
from src.evaluate import (
    TIERS,
    compute_metrics,
    count_false_positives,
    false_positive_reduction,
    rule_based_disambiguate,
    run_evaluation,
)


# ---------------------------------------------------------------------------
# Baseline behavior
# ---------------------------------------------------------------------------
class TestBaseline:
    def test_exact_name_returns_corpus_tier(self):
        assert baseline_predict("Elena Volkov") == ("CONFIRMED_SANCTIONS", "SANC-001")

    def test_case_insensitive(self):
        assert baseline_predict("elena volkov")[0] == "CONFIRMED_SANCTIONS"

    def test_phonetic_variant_misses(self):
        # Naive baseline can't match Catherine->Katherine or Mueller->Muller.
        assert baseline_predict("Catherine Mueller") == ("NO_MATCH", None)

    def test_unknown_name_is_no_match(self):
        assert baseline_predict("Nobody At All") == ("NO_MATCH", None)


# ---------------------------------------------------------------------------
# Rule-based stand-in disambiguator
# ---------------------------------------------------------------------------
class TestRuleBasedDisambiguator:
    def test_no_candidates_is_no_match(self):
        r = rule_based_disambiguate("Nobody", [], dob="2000-01-01", nationality="X")
        assert r.match_tier == "NO_MATCH" and r.confidence == 0.0

    def test_full_match_returns_candidate_tier(self):
        cands = find_candidates("Elena Volkov")
        r = rule_based_disambiguate("Elena Volkov", cands, dob="1968-04-15", nationality="Russia")
        assert r.match_tier == "CONFIRMED_SANCTIONS"
        assert r.confidence >= 0.9

    def test_divergent_attributes_downgrade_to_similarity(self):
        cands = find_candidates("Elena Volkov")
        r = rule_based_disambiguate("Elena Volkov", cands, dob="1995-02-10", nationality="Ukraine")
        assert r.match_tier == "NAME_SIMILARITY_ONLY"
        assert r.confidence < 0.3

    def test_name_only_confidence_capped(self):
        cands = find_candidates("Mohammed Al-Rashid")
        r = rule_based_disambiguate("Mohammed Al-Rashid", cands)  # no aux data
        assert r.confidence <= 0.6

    def test_weak_name_overlap_without_corroboration_is_no_match(self):
        # Priya Nair shares only a surname with NEWS-009 (Lakshmi Nair).
        cands = find_candidates("Priya Nair")
        r = rule_based_disambiguate("Priya Nair", cands, dob="1993-12-25", nationality="India")
        assert r.match_tier == "NO_MATCH"


# ---------------------------------------------------------------------------
# Metrics primitives
# ---------------------------------------------------------------------------
class TestMetrics:
    def test_compute_metrics_structure(self):
        y_true = ["CONFIRMED_SANCTIONS", "NO_MATCH", "ADVERSE_NEWS"]
        y_pred = ["CONFIRMED_SANCTIONS", "NO_MATCH", "ADVERSE_NEWS"]
        m = compute_metrics(y_true, y_pred)
        assert set(m["per_tier"]) == set(TIERS)
        assert m["accuracy"] == 1.0
        assert m["macro"]["f1"] > 0

    def test_count_false_positives(self):
        # non-hit ground truth escalated to a hit tier
        y_true = ["NAME_SIMILARITY_ONLY", "NO_MATCH", "CONFIRMED_SANCTIONS"]
        y_pred = ["CONFIRMED_SANCTIONS", "ADVERSE_NEWS", "CONFIRMED_SANCTIONS"]
        assert count_false_positives(y_true, y_pred) == 2

    def test_false_positive_reduction_math(self):
        y_true = ["NAME_SIMILARITY_ONLY", "NAME_SIMILARITY_ONLY"]
        y_base = ["CONFIRMED_SANCTIONS", "ADVERSE_NEWS"]  # 2 FPs
        y_pipe = ["NAME_SIMILARITY_ONLY", "ADVERSE_NEWS"]  # 1 FP
        fp = false_positive_reduction(y_true, y_base, y_pipe)
        assert fp["baseline_false_positives"] == 2
        assert fp["pipeline_false_positives"] == 1
        assert fp["reduction_pct"] == 50.0


# ---------------------------------------------------------------------------
# End-to-end harness (mock backend)
# ---------------------------------------------------------------------------
class TestRunEvaluation:
    def test_pipeline_beats_baseline_and_reduces_false_positives(self):
        report = run_evaluation(mock=True, write=False)
        assert report["backend"] == "rule_based_stand_in"
        assert report["eval_case_count"] == 48
        # Pipeline should not be worse than the naive baseline on macro F1.
        assert report["pipeline"]["macro"]["f1"] >= report["baseline"]["macro"]["f1"]
        # And it should strictly reduce false positives.
        fp = report["false_positive_reduction"]
        assert fp["baseline_false_positives"] > 0
        assert fp["pipeline_false_positives"] < fp["baseline_false_positives"]
