"""Step 2 unit tests — the deterministic candidate filter.

The headline requirement: the filter must catch the phonetic and
transliteration variants defined in the Step 1 eval set, plus exact matches,
while not confidently matching true negatives — all with no LLM in the loop.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from src.candidate_filter import (
    Candidate,
    find_candidates,
    load_corpus,
    normalize_name,
    phonetic_score,
    tokens,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EVAL_SET = json.loads((DATA_DIR / "eval_set.json").read_text(encoding="utf-8"))
CORPUS = load_corpus()

EVAL_CASES = EVAL_SET["cases"]


def _ids(candidates: list[Candidate]) -> list[str]:
    return [c.id for c in candidates]


def _cases(*categories: str) -> list[dict]:
    return [c for c in EVAL_CASES if c["category"] in categories]


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
class TestNormalization:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("José García", "jose garcia"),
            ("Mohammed Al-Rashid", "mohammed al rashid"),
            ("  Elena   Volkov  ", "elena volkov"),
            ("Kim Jong-Su", "kim jong su"),
            ("O'Brien", "o brien"),
        ],
    )
    def test_normalize_name(self, raw, expected):
        assert normalize_name(raw) == expected

    def test_accented_and_ascii_forms_normalize_equal(self):
        assert normalize_name("José García") == normalize_name("Jose Garcia")

    def test_empty_name_normalizes_empty(self):
        assert normalize_name("") == ""
        assert tokens("   ") == []


# ---------------------------------------------------------------------------
# Phonetic scoring
# ---------------------------------------------------------------------------
class TestPhoneticScore:
    @pytest.mark.parametrize(
        "a, b",
        [
            ("Catherine Mueller", "Katherine Muller"),
            ("Dmitry Ivanov", "Dmitri Ivanov"),
            ("Sergey Morozov", "Sergei Morozov"),
            ("Jon Smyth", "John Smith"),
        ],
    )
    def test_phonetic_variants_fully_match(self, a, b):
        assert phonetic_score(tokens(a), tokens(b)) == pytest.approx(1.0)

    def test_unrelated_names_score_low(self):
        assert phonetic_score(tokens("Oluwaseun Adeyemi"), tokens("Elena Volkov")) < 0.5


# ---------------------------------------------------------------------------
# Recall — the core deliverable
# ---------------------------------------------------------------------------
class TestVariantRecall:
    @pytest.mark.parametrize(
        "case",
        _cases("phonetic_variant", "transliteration_variant"),
        ids=[c["case_id"] for c in _cases("phonetic_variant", "transliteration_variant")],
    )
    def test_variant_surfaces_true_entry(self, case):
        """Every phonetic/transliteration variant must surface its corpus entry."""
        candidates = find_candidates(case["counterparty_name"])
        assert case["related_corpus_id"] in _ids(candidates), (
            f"{case['case_id']} ({case['counterparty_name']}): expected "
            f"{case['related_corpus_id']} in {_ids(candidates)}"
        )

    @pytest.mark.parametrize(
        "case",
        _cases("true_positive_exact"),
        ids=[c["case_id"] for c in _cases("true_positive_exact")],
    )
    def test_exact_match_is_top_candidate(self, case):
        candidates = find_candidates(case["counterparty_name"])
        assert candidates, f"{case['case_id']}: no candidates returned"
        assert candidates[0].id == case["related_corpus_id"]

    @pytest.mark.parametrize(
        "case",
        _cases("false_positive_name_only"),
        ids=[c["case_id"] for c in _cases("false_positive_name_only")],
    )
    def test_name_only_false_positive_still_surfaces_entry(self, case):
        """Name collisions must reach the LLM stage (filter can't see DOB/nationality)."""
        candidates = find_candidates(case["counterparty_name"])
        assert case["related_corpus_id"] in _ids(candidates)


# ---------------------------------------------------------------------------
# Precision guardrails — true negatives should not match confidently
# ---------------------------------------------------------------------------
class TestTrueNegatives:
    @pytest.mark.parametrize(
        "case",
        _cases("no_match"),
        ids=[c["case_id"] for c in _cases("no_match")],
    )
    def test_no_match_has_no_confident_candidate(self, case):
        """A name not on the list must not produce a near-exact (>=0.85) candidate."""
        candidates = find_candidates(case["counterparty_name"])
        assert all(c.combined_score < 0.85 for c in candidates), (
            f"{case['case_id']} ({case['counterparty_name']}) matched too confidently: "
            f"{[(c.id, round(c.combined_score, 2)) for c in candidates]}"
        )


# ---------------------------------------------------------------------------
# Behavior / robustness
# ---------------------------------------------------------------------------
class TestBehavior:
    def test_word_order_swap_is_matched(self):
        # Corpus has SIM-004 "Wei Zhang"; a reversed order should still surface it.
        ids = _ids(find_candidates("Zhang Wei"))
        assert "SIM-004" in ids

    def test_top_n_is_respected(self):
        assert len(find_candidates("John Smith", top_n=2)) <= 2

    def test_empty_name_returns_nothing(self):
        assert find_candidates("") == []
        assert find_candidates("   ") == []

    def test_deterministic(self):
        a = _ids(find_candidates("Catherine Mueller"))
        b = _ids(find_candidates("Catherine Mueller"))
        assert a == b

    def test_results_sorted_by_combined_score_desc(self):
        cands = find_candidates("John Smith")
        scores = [c.combined_score for c in cands]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Performance — must run in milliseconds
# ---------------------------------------------------------------------------
class TestPerformance:
    def test_filter_is_fast(self):
        queries = [c["counterparty_name"] for c in EVAL_CASES]
        start = time.perf_counter()
        runs = 5
        for _ in range(runs):
            for q in queries:
                find_candidates(q)
        elapsed = time.perf_counter() - start
        per_query_ms = elapsed / (runs * len(queries)) * 1000
        # Generous ceiling for CI noise; typically well under 1 ms/query.
        assert per_query_ms < 15, f"{per_query_ms:.2f} ms/query too slow"
