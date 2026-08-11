"""
Naive exact-string-matching baseline (Step 5).

The point of comparison for the two-stage pipeline. It does what a screening
system with no disambiguation does: if the counterparty name exactly matches a
watchlist entry, flag it as that entry's tier — with no DOB/nationality check.

Consequences this baseline deliberately exhibits:
* It escalates coincidental name collisions to CONFIRMED_SANCTIONS / ADVERSE_NEWS
  (false positives), because it can't tell two people with the same name apart.
* It misses phonetic and transliteration variants (Catherine/Katherine,
  José/Jose), because the strings don't match exactly -> NO_MATCH.

Normalization is intentionally minimal (casefold + whitespace collapse) — no
accent folding, no phonetics — so it stays a genuinely naive baseline.
"""

from __future__ import annotations

from functools import lru_cache

from src.candidate_filter import load_corpus


def _baseline_key(name: str) -> str:
    return " ".join((name or "").casefold().split())


@lru_cache(maxsize=8)
def _exact_index() -> dict[str, dict]:
    """Map of baseline-normalized corpus name -> entry."""
    index: dict[str, dict] = {}
    for entry in load_corpus():
        index.setdefault(_baseline_key(entry["name"]), entry)
    return index


def baseline_predict(name: str) -> tuple[str, str | None]:
    """Return (predicted_tier, matched_corpus_id) for the naive baseline."""
    entry = _exact_index().get(_baseline_key(name))
    if entry is None:
        return "NO_MATCH", None
    return entry["tier"], entry["id"]
