"""Step 10 — unit tests for the deterministic candidate filter.

Headline requirement: the filter must surface the phonetic and transliteration
variants from the eval set within the top-N, and must NOT confidently match true
negatives — all with no LLM in the loop.
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
EVAL_CASES = EVAL_SET["cases"]


def _ids(cands: list[Candidate]) -> list[str]:
    return [c.id for c in cands]


def _cases(*categories: str) -> list[dict]:
    return [c for c in EVAL_CASES if c["category"] in categories]


# ---------------------------------------------------------------------------
class TestNormalization:
    @pytest.mark.parametrize("raw, expected", [
        ("José García", "jose garcia"),
        ("Mohammed Al-Rashid", "mohammed al rashid"),
        ("  Elena   Volkov  ", "elena volkov"),
        ("Kim Jong-Su", "kim jong su"),
    ])
    def test_normalize_name(self, raw, expected):
        assert normalize_name(raw) == expected

    def test_accented_and_ascii_normalize_equal(self):
        assert normalize_name("José García") == normalize_name("Jose Garcia")

    def test_empty(self):
        assert normalize_name("") == ""
        assert tokens("   ") == []


class TestPhoneticScore:
    @pytest.mark.parametrize("a, b", [
        ("Catherine Mueller", "Katherine Muller"),
        ("Dmitry Ivanov", "Dmitri Ivanov"),
        ("Sergey Morozov", "Sergei Morozov"),
        ("Jon Smyth", "John Smith"),
    ])
    def test_variants_fully_match(self, a, b):
        assert phonetic_score(tokens(a), tokens(b)) == pytest.approx(1.0)

    def test_unrelated_low(self):
        assert phonetic_score(tokens("Oluwaseun Adeyemi"), tokens("Elena Volkov")) < 0.5


class TestVariantRecall:
    @pytest.mark.parametrize("case", _cases("phonetic_variant", "transliteration_variant"),
                             ids=[c["case_id"] for c in _cases("phonetic_variant", "transliteration_variant")])
    def test_variant_surfaces_true_entry(self, case):
        cands = find_candidates(case["counterparty_name"])
        assert case["related_corpus_id"] in _ids(cands), (
            f"{case['case_id']} ({case['counterparty_name']}): expected "
            f"{case['related_corpus_id']} in {_ids(cands)}"
        )

    @pytest.mark.parametrize("case", _cases("true_positive_exact"),
                             ids=[c["case_id"] for c in _cases("true_positive_exact")])
    def test_exact_match_is_top_candidate(self, case):
        cands = find_candidates(case["counterparty_name"])
        assert cands and cands[0].id == case["related_corpus_id"]

    @pytest.mark.parametrize("case", _cases("false_positive_name_only"),
                             ids=[c["case_id"] for c in _cases("false_positive_name_only")])
    def test_name_collision_still_surfaces_entry(self, case):
        assert case["related_corpus_id"] in _ids(find_candidates(case["counterparty_name"]))


class TestTrueNegatives:
    @pytest.mark.parametrize("case", _cases("no_match"),
                             ids=[c["case_id"] for c in _cases("no_match")])
    def test_no_confident_candidate(self, case):
        cands = find_candidates(case["counterparty_name"])
        assert all(c.combined_score < 0.85 for c in cands), (
            f"{case['case_id']} ({case['counterparty_name']}) matched too confidently: "
            f"{[(c.id, round(c.combined_score, 2)) for c in cands]}"
        )


class TestBehavior:
    def test_word_order_swap(self):
        assert "SIM-004" in _ids(find_candidates("Zhang Wei"))  # corpus has "Wei Zhang"

    def test_top_n_respected(self):
        assert len(find_candidates("John Smith", top_n=2)) <= 2

    def test_empty_returns_nothing(self):
        assert find_candidates("") == []
        assert find_candidates("   ") == []

    def test_deterministic(self):
        assert _ids(find_candidates("Catherine Mueller")) == _ids(find_candidates("Catherine Mueller"))

    def test_sorted_desc(self):
        scores = [c.combined_score for c in find_candidates("John Smith")]
        assert scores == sorted(scores, reverse=True)


class TestPerformance:
    def test_fast(self):
        queries = [c["counterparty_name"] for c in EVAL_CASES]
        start = time.perf_counter()
        runs = 5
        for _ in range(runs):
            for q in queries:
                find_candidates(q)
        per_query_ms = (time.perf_counter() - start) / (runs * len(queries)) * 1000
        assert per_query_ms < 15, f"{per_query_ms:.2f} ms/query too slow"
