"""
Step 2 — Deterministic candidate filter (NO LLM).

Given a counterparty name, narrow the 60-100 entry watchlist corpus down to a
handful of plausible candidates in milliseconds, so the expensive LLM stage in
Step 3 only ever reasons over a short list.

Three complementary signals, combined per corpus entry:

* unidecode  -> normalization: strip accents/transliteration noise so
               "Jose Garcia" and "Jose Garcia" compare equal.
* rapidfuzz  -> edit-distance / token similarity: catches typos, dropped
               hyphens, word-order swaps, and minor spelling drift.
* jellyfish  -> phonetic codes (Metaphone + NYSIIS + Soundex): catches
               same-sounding spellings like Catherine/Katherine,
               Mueller/Muller, Dmitri/Dmitry, Sergei/Sergey.

The combined score blends fuzzy + phonetic. A candidate is kept if it clears a
lenient cutoff (recall-oriented — better to hand the LLM a spurious candidate
than to drop a real one), then the top-N by score are returned.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import jellyfish
from rapidfuzz import fuzz
from unidecode import unidecode

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_CORPUS_PATH = DATA_DIR / "corpus.json"

# Scoring weights and gates. Tuned so every phonetic/transliteration variant in
# the Step 1 eval set surfaces its true corpus entry within the top-N.
FUZZY_WEIGHT = 0.65
PHONETIC_WEIGHT = 0.35
DEFAULT_TOP_N = 5
DEFAULT_SCORE_CUTOFF = 0.45  # combined score in [0, 1]
# A candidate also survives the cutoff if either single signal is strong enough,
# so a perfect phonetic match with weak string overlap (or vice versa) is kept.
STRONG_FUZZY = 0.60
STRONG_PHONETIC = 0.99


@dataclass(frozen=True)
class Candidate:
    """A corpus entry that plausibly matches the query, with its scores."""

    entry: dict
    combined_score: float
    fuzzy_score: float
    phonetic_score: float

    @property
    def id(self) -> str:
        return self.entry["id"]

    @property
    def name(self) -> str:
        return self.entry["name"]

    @property
    def tier(self) -> str:
        return self.entry["tier"]


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize_name(name: str) -> str:
    """Accent-fold (unidecode), lowercase, strip punctuation, collapse whitespace."""
    folded = unidecode(name or "").lower()
    cleaned = _NON_ALNUM.sub(" ", folded)
    return " ".join(cleaned.split())


def tokens(name: str) -> list[str]:
    return normalize_name(name).split()


@lru_cache(maxsize=4096)
def _phonetic_codes(token: str) -> frozenset[str]:
    """Metaphone + NYSIIS + Soundex codes for a single normalized token."""
    codes: set[str] = set()
    for fn in (jellyfish.metaphone, jellyfish.nysiis, jellyfish.soundex):
        try:
            code = fn(token)
        except Exception:
            code = ""
        if code:
            codes.add(code)
    return frozenset(codes)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def fuzzy_score(query_norm: str, entry_norm: str) -> float:
    """Best of several rapidfuzz scorers, in [0, 1].

    WRatio handles general similarity; token_sort handles word-order swaps
    (Wei Zhang / Zhang Wei); token_set handles subset/extra-token cases.
    """
    if not query_norm or not entry_norm:
        return 0.0
    best = max(
        fuzz.WRatio(query_norm, entry_norm),
        fuzz.token_sort_ratio(query_norm, entry_norm),
        fuzz.token_set_ratio(query_norm, entry_norm),
    )
    return best / 100.0


def phonetic_score(query_tokens: list[str], entry_tokens: list[str]) -> float:
    """Fraction of query tokens that share a phonetic code with some entry token."""
    if not query_tokens:
        return 0.0
    entry_codes = [_phonetic_codes(t) for t in entry_tokens]
    matched = 0
    for qt in query_tokens:
        qc = _phonetic_codes(qt)
        if any(qc & ec for ec in entry_codes):
            matched += 1
    return matched / len(query_tokens)


def score_entry(query_norm: str, query_tokens: list[str], entry: dict) -> Candidate:
    entry_norm = normalize_name(entry["name"])
    f = fuzzy_score(query_norm, entry_norm)
    p = phonetic_score(query_tokens, tokens(entry["name"]))
    combined = FUZZY_WEIGHT * f + PHONETIC_WEIGHT * p
    return Candidate(entry=entry, combined_score=combined, fuzzy_score=f, phonetic_score=p)


# ---------------------------------------------------------------------------
# Corpus loading (cached)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=8)
def load_corpus(path: str | None = None) -> tuple[dict, ...]:
    corpus_path = Path(path) if path else DEFAULT_CORPUS_PATH
    data = json.loads(corpus_path.read_text(encoding="utf-8"))
    return tuple(data["entries"])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def find_candidates(
    name: str,
    corpus: list[dict] | tuple[dict, ...] | None = None,
    top_n: int = DEFAULT_TOP_N,
    score_cutoff: float = DEFAULT_SCORE_CUTOFF,
) -> list[Candidate]:
    """Return up to `top_n` plausible corpus candidates for `name`, best first.

    Deterministic and LLM-free. Runs in well under a millisecond per query for
    an 80-entry corpus.
    """
    entries = corpus if corpus is not None else load_corpus()
    query_norm = normalize_name(name)
    query_tokens = query_norm.split()
    if not query_norm:
        return []

    scored = [score_entry(query_norm, query_tokens, e) for e in entries]

    kept = [
        c for c in scored
        if c.combined_score >= score_cutoff
        or c.fuzzy_score >= STRONG_FUZZY
        or c.phonetic_score >= STRONG_PHONETIC
    ]
    # Sort by combined score, then fuzzy, then id for stable/deterministic order.
    kept.sort(key=lambda c: (-c.combined_score, -c.fuzzy_score, c.id))
    return kept[:top_n]


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Deterministic watchlist candidate filter.")
    parser.add_argument("name", help="counterparty name to screen")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    args = parser.parse_args()

    results = find_candidates(args.name, top_n=args.top_n)
    if not results:
        print(f"No candidates for {args.name!r}.")
        return
    print(f"Candidates for {args.name!r}:")
    for c in results:
        print(
            f"  {c.id:9s} {c.name:28s} {c.tier:22s} "
            f"combined={c.combined_score:.3f} fuzzy={c.fuzzy_score:.3f} phon={c.phonetic_score:.3f}"
        )


if __name__ == "__main__":
    _cli()
