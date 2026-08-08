"""
Step 1 — Synthetic corpus + frozen evaluation set generator.

Design goals
------------
* Deterministic: fixed seeds so the committed JSON files are reproducible.
* Curated where it matters: the "interesting" corpus entries and every eval
  case are hand-authored so the eval set exercises specific failure modes
  (name-only false positives, phonetic variants, transliteration variants,
  partial-data ambiguity, true positives, and true negatives).
* Faker only fills the bulk of the watchlist so the candidate filter in Step 2
  has realistic noise to sift through.

The eval set's ground truth is the `expected_tier` field. `expected_confidence_band`
is a secondary, softer check used later in evaluation.

Run once, then commit data/corpus.json and data/eval_set.json. Do NOT regenerate
on every run — the eval set must stay frozen so metrics are comparable over time.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import jellyfish
from faker import Faker
from unidecode import unidecode

CORPUS_VERSION = "1.0.0"
EVAL_VERSION = "1.0.0"
SEED = 42

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# ---------------------------------------------------------------------------
# Hand-crafted watchlist entries (the ones the eval set references directly)
# ---------------------------------------------------------------------------
# Tiers:
#   CONFIRMED_SANCTIONS  -> on an official sanctions list
#   ADVERSE_NEWS         -> negative media / open investigation, not sanctioned
#   NAME_SIMILARITY_ONLY -> a benign person whose name collides with watchlisted
#                           names; present to generate realistic false positives
CRAFTED_CORPUS = [
    # --- CONFIRMED_SANCTIONS -------------------------------------------------
    {"id": "SANC-001", "name": "Elena Volkov",          "dob": "1968-04-15", "nationality": "Russia",       "tier": "CONFIRMED_SANCTIONS"},
    {"id": "SANC-002", "name": "Mohammed Al-Rashid",    "dob": "1972-09-30", "nationality": "Syria",        "tier": "CONFIRMED_SANCTIONS"},
    {"id": "SANC-003", "name": "Dmitri Ivanov",         "dob": "1980-01-22", "nationality": "Russia",       "tier": "CONFIRMED_SANCTIONS"},
    {"id": "SANC-004", "name": "Viktor Petrov",         "dob": "1965-07-11", "nationality": "Belarus",      "tier": "CONFIRMED_SANCTIONS"},
    {"id": "SANC-005", "name": "Kim Jong-Su",           "dob": "1958-12-03", "nationality": "North Korea",  "tier": "CONFIRMED_SANCTIONS"},
    {"id": "SANC-006", "name": "Ali Hassan Nasrallah",  "dob": "1975-06-18", "nationality": "Lebanon",      "tier": "CONFIRMED_SANCTIONS"},
    {"id": "SANC-007", "name": "Ahmed Farouk",          "dob": "1983-02-27", "nationality": "Egypt",        "tier": "CONFIRMED_SANCTIONS"},
    {"id": "SANC-008", "name": "Ramzan Kadyrov",        "dob": "1976-10-05", "nationality": "Russia",       "tier": "CONFIRMED_SANCTIONS"},
    {"id": "SANC-009", "name": "Bashar Nadal",          "dob": "1970-08-21", "nationality": "Syria",        "tier": "CONFIRMED_SANCTIONS"},
    {"id": "SANC-010", "name": "Reza Motevalli",        "dob": "1978-03-14", "nationality": "Iran",         "tier": "CONFIRMED_SANCTIONS"},

    # --- ADVERSE_NEWS --------------------------------------------------------
    {"id": "NEWS-001", "name": "Jose Garcia",           "dob": "1979-11-05", "nationality": "Mexico",       "tier": "ADVERSE_NEWS"},
    {"id": "NEWS-002", "name": "Katherine Muller",      "dob": "1985-03-19", "nationality": "Germany",      "tier": "ADVERSE_NEWS"},
    {"id": "NEWS-003", "name": "Robert Chen",           "dob": "1970-08-14", "nationality": "Singapore",    "tier": "ADVERSE_NEWS"},
    {"id": "NEWS-004", "name": "Fatima Bourbon",        "dob": "1988-05-22", "nationality": "France",       "tier": "ADVERSE_NEWS"},
    {"id": "NEWS-005", "name": "Sergei Morozov",        "dob": "1974-10-09", "nationality": "Russia",       "tier": "ADVERSE_NEWS"},
    {"id": "NEWS-006", "name": "Carlos Mendoza",        "dob": "1981-01-30", "nationality": "Colombia",     "tier": "ADVERSE_NEWS"},
    {"id": "NEWS-007", "name": "Nadia Petrenko",        "dob": "1987-06-12", "nationality": "Ukraine",      "tier": "ADVERSE_NEWS"},
    {"id": "NEWS-008", "name": "Ibrahim Toure",         "dob": "1976-09-08", "nationality": "Mali",         "tier": "ADVERSE_NEWS"},
    {"id": "NEWS-009", "name": "Lakshmi Nair",          "dob": "1983-04-17", "nationality": "India",        "tier": "ADVERSE_NEWS"},
    {"id": "NEWS-010", "name": "Giovanni Russo",        "dob": "1969-12-01", "nationality": "Italy",        "tier": "ADVERSE_NEWS"},

    # --- NAME_SIMILARITY_ONLY (benign collision decoys) ----------------------
    {"id": "SIM-001",  "name": "John Smith",            "dob": "1975-03-12", "nationality": "Nigeria",      "tier": "NAME_SIMILARITY_ONLY"},
    {"id": "SIM-002",  "name": "Maria Rodriguez",       "dob": "1982-07-30", "nationality": "Spain",        "tier": "NAME_SIMILARITY_ONLY"},
    {"id": "SIM-003",  "name": "David Johnson",         "dob": "1969-04-25", "nationality": "United States","tier": "NAME_SIMILARITY_ONLY"},
    {"id": "SIM-004",  "name": "Wei Zhang",             "dob": "1990-06-08", "nationality": "China",        "tier": "NAME_SIMILARITY_ONLY"},
    {"id": "SIM-005",  "name": "Anna Kowalski",         "dob": "1986-09-17", "nationality": "Poland",       "tier": "NAME_SIMILARITY_ONLY"},
    {"id": "SIM-006",  "name": "Michael Brown",         "dob": "1978-02-14", "nationality": "United Kingdom","tier": "NAME_SIMILARITY_ONLY"},
    {"id": "SIM-007",  "name": "Sofia Rossi",           "dob": "1991-11-23", "nationality": "Italy",        "tier": "NAME_SIMILARITY_ONLY"},
]


# Reserved names we must not let Faker filler duplicate.
RESERVED_NAMES = {e["name"].lower() for e in CRAFTED_CORPUS}

# Names used only in NO_MATCH eval cases — keep them out of the corpus entirely.
NO_MATCH_NAMES = {
    "oluwaseun adeyemi", "lars andersen", "priya nair", "thomas mueller",
    "chen wei guo", "amelia clarke", "yuki tanaka", "diego fernandez",
}


def _name_tokens(name: str) -> list[str]:
    return "".join(c if c.isalnum() else " " for c in unidecode(name).lower()).split()


def _phon(token: str) -> set[str]:
    codes = set()
    for fn in (jellyfish.metaphone, jellyfish.nysiis, jellyfish.soundex):
        try:
            c = fn(token)
        except Exception:
            c = ""
        if c:
            codes.add(c)
    return codes


def _is_full_name_twin(name: str, reserved: list[str]) -> bool:
    """True if `name` is a first+last phonetic twin of any reserved name.

    Prevents Faker filler from accidentally colliding with a frozen eval name
    on BOTH given and family name (e.g. 'Thomas Miller' vs 'Thomas Mueller'),
    which would silently turn a true-negative eval case into a real match.
    Surname-only or given-only overlap is allowed on purpose — those are the
    benign collisions the corpus is meant to contain.
    """
    nt = _name_tokens(name)
    if len(nt) < 2:
        return False
    n_first, n_last = _phon(nt[0]), _phon(nt[-1])
    for other in reserved:
        ot = _name_tokens(other)
        if len(ot) < 2:
            continue
        if (n_first & _phon(ot[0])) and (n_last & _phon(ot[-1])):
            return True
    return False


def build_filler_corpus(target_total: int) -> list[dict]:
    """Faker-generated filler across the three tiers, deduped against reserved names."""
    fake = Faker()
    Faker.seed(SEED)
    random.seed(SEED)

    # Full reserved name list (crafted corpus + eval-only NO_MATCH names) so filler
    # never becomes a full-name twin of a frozen eval reference.
    reserved_full = [e["name"] for e in CRAFTED_CORPUS] + sorted(NO_MATCH_NAMES)

    tiers = ["CONFIRMED_SANCTIONS", "ADVERSE_NEWS", "NAME_SIMILARITY_ONLY"]
    nationalities = [
        "Russia", "China", "Iran", "Syria", "North Korea", "Venezuela", "Cuba",
        "Belarus", "Myanmar", "Sudan", "Nigeria", "United States", "Germany",
        "France", "Italy", "Spain", "India", "Brazil", "Mexico", "Ukraine",
        "Turkey", "Egypt", "Pakistan", "Indonesia", "Vietnam", "Poland",
    ]

    filler: list[dict] = []
    counters = {"CONFIRMED_SANCTIONS": 10, "ADVERSE_NEWS": 10, "NAME_SIMILARITY_ONLY": 7}
    prefix = {"CONFIRMED_SANCTIONS": "SANC", "ADVERSE_NEWS": "NEWS", "NAME_SIMILARITY_ONLY": "SIM"}

    used = set(RESERVED_NAMES) | set(NO_MATCH_NAMES)
    n_needed = target_total - len(CRAFTED_CORPUS)
    i = 0
    while len(filler) < n_needed:
        # Weight toward the similarity/decoy tier so false-positive pressure is realistic.
        tier = random.choices(tiers, weights=[0.3, 0.3, 0.4])[0]
        name = fake.name()
        key = name.lower()
        if key in used or "." in name or len(name.split()) < 2:
            i += 1
            continue
        if _is_full_name_twin(name, reserved_full):
            i += 1
            continue
        used.add(key)
        counters[tier] += 1
        entry_id = f"{prefix[tier]}-{counters[tier]:03d}"
        dob = fake.date_of_birth(minimum_age=25, maximum_age=75).isoformat()
        filler.append({
            "id": entry_id,
            "name": name,
            "dob": dob,
            "nationality": random.choice(nationalities),
            "tier": tier,
        })
        i += 1

    return filler


def build_corpus() -> dict:
    filler = build_filler_corpus(target_total=80)
    entries = CRAFTED_CORPUS + filler
    entries.sort(key=lambda e: e["id"])
    return {
        "version": CORPUS_VERSION,
        "description": "Synthetic sanctions / adverse-media watchlist corpus. Names, DOBs, "
                       "and nationalities are fictional and generated for testing only.",
        "seed": SEED,
        "count": len(entries),
        "tier_counts": {
            t: sum(1 for e in entries if e["tier"] == t)
            for t in ["CONFIRMED_SANCTIONS", "ADVERSE_NEWS", "NAME_SIMILARITY_ONLY"]
        },
        "entries": entries,
    }


# ---------------------------------------------------------------------------
# Frozen evaluation set — every case hand-authored with ground-truth labels
# ---------------------------------------------------------------------------
# Confidence bands (soft secondary check; expected_tier is the hard label):
#   high -> >= 0.85    mid -> 0.40-0.70    low -> <= 0.30    zero -> == 0.0
EVAL_CASES = [
    # === TRUE POSITIVES: name + DOB + nationality all match =================
    {"case_id": "TP-01", "counterparty_name": "Elena Volkov",         "dob": "1968-04-15", "nationality": "Russia",    "category": "true_positive_exact",   "expected_tier": "CONFIRMED_SANCTIONS", "expected_confidence_band": "high", "expected_confidence_approx": 0.95, "related_corpus_id": "SANC-001", "notes": "Worked example #2: exact match on all attributes."},
    {"case_id": "TP-02", "counterparty_name": "Mohammed Al-Rashid",   "dob": "1972-09-30", "nationality": "Syria",     "category": "true_positive_exact",   "expected_tier": "CONFIRMED_SANCTIONS", "expected_confidence_band": "high", "expected_confidence_approx": 0.94, "related_corpus_id": "SANC-002", "notes": "Exact confirmed-sanctions hit."},
    {"case_id": "TP-03", "counterparty_name": "Viktor Petrov",        "dob": "1965-07-11", "nationality": "Belarus",   "category": "true_positive_exact",   "expected_tier": "CONFIRMED_SANCTIONS", "expected_confidence_band": "high", "expected_confidence_approx": 0.93, "related_corpus_id": "SANC-004", "notes": "Exact confirmed-sanctions hit."},
    {"case_id": "TP-04", "counterparty_name": "Ali Hassan Nasrallah", "dob": "1975-06-18", "nationality": "Lebanon",   "category": "true_positive_exact",   "expected_tier": "CONFIRMED_SANCTIONS", "expected_confidence_band": "high", "expected_confidence_approx": 0.93, "related_corpus_id": "SANC-006", "notes": "Exact confirmed-sanctions hit."},
    {"case_id": "TP-05", "counterparty_name": "Jose Garcia",          "dob": "1979-11-05", "nationality": "Mexico",    "category": "true_positive_exact",   "expected_tier": "ADVERSE_NEWS",        "expected_confidence_band": "high", "expected_confidence_approx": 0.90, "related_corpus_id": "NEWS-001", "notes": "Exact adverse-news hit."},
    {"case_id": "TP-06", "counterparty_name": "Katherine Muller",     "dob": "1985-03-19", "nationality": "Germany",   "category": "true_positive_exact",   "expected_tier": "ADVERSE_NEWS",        "expected_confidence_band": "high", "expected_confidence_approx": 0.90, "related_corpus_id": "NEWS-002", "notes": "Exact adverse-news hit."},
    {"case_id": "TP-07", "counterparty_name": "Sergei Morozov",       "dob": "1974-10-09", "nationality": "Russia",    "category": "true_positive_exact",   "expected_tier": "ADVERSE_NEWS",        "expected_confidence_band": "high", "expected_confidence_approx": 0.90, "related_corpus_id": "NEWS-005", "notes": "Exact adverse-news hit."},

    # === FALSE POSITIVES: name matches, but DOB/nationality clearly diverge ==
    {"case_id": "FP-01", "counterparty_name": "John Smith",           "dob": "1990-11-02", "nationality": "Canada",    "category": "false_positive_name_only", "expected_tier": "NAME_SIMILARITY_ONLY", "expected_confidence_band": "low", "expected_confidence_approx": 0.15, "related_corpus_id": "SIM-001", "notes": "Worked example #1: DOB + nationality both diverge from corpus entry."},
    {"case_id": "FP-02", "counterparty_name": "Elena Volkov",         "dob": "1995-02-10", "nationality": "Ukraine",   "category": "false_positive_name_only", "expected_tier": "NAME_SIMILARITY_ONLY", "expected_confidence_band": "low", "expected_confidence_approx": 0.12, "related_corpus_id": "SANC-001", "notes": "Name matches a sanctioned entry but DOB (27yr gap) and nationality diverge -> coincidental collision."},
    {"case_id": "FP-03", "counterparty_name": "Mohammed Al-Rashid",   "dob": "2001-01-01", "nationality": "United Arab Emirates", "category": "false_positive_name_only", "expected_tier": "NAME_SIMILARITY_ONLY", "expected_confidence_band": "low", "expected_confidence_approx": 0.12, "related_corpus_id": "SANC-002", "notes": "Common name; DOB and nationality diverge from sanctioned entry."},
    {"case_id": "FP-04", "counterparty_name": "Dmitri Ivanov",        "dob": "1955-12-12", "nationality": "Kazakhstan","category": "false_positive_name_only", "expected_tier": "NAME_SIMILARITY_ONLY", "expected_confidence_band": "low", "expected_confidence_approx": 0.15, "related_corpus_id": "SANC-003", "notes": "Very common Russian name; attributes diverge."},
    {"case_id": "FP-05", "counterparty_name": "Robert Chen",          "dob": "1992-01-19", "nationality": "Malaysia",  "category": "false_positive_name_only", "expected_tier": "NAME_SIMILARITY_ONLY", "expected_confidence_band": "low", "expected_confidence_approx": 0.18, "related_corpus_id": "NEWS-003", "notes": "Name matches adverse-news entry but DOB (22yr gap) and nationality diverge."},
    {"case_id": "FP-06", "counterparty_name": "Maria Rodriguez",      "dob": "1999-05-05", "nationality": "Argentina", "category": "false_positive_name_only", "expected_tier": "NAME_SIMILARITY_ONLY", "expected_confidence_band": "low", "expected_confidence_approx": 0.20, "related_corpus_id": "SIM-002", "notes": "Extremely common name; matches similarity-only decoy with divergent attributes."},

    # === PHONETIC VARIANTS (Soundex/Metaphone territory) ====================
    {"case_id": "PH-01", "counterparty_name": "Catherine Mueller",    "dob": "1985-03-19", "nationality": "Germany",   "category": "phonetic_variant", "expected_tier": "ADVERSE_NEWS",        "expected_confidence_band": "high", "expected_confidence_approx": 0.88, "related_corpus_id": "NEWS-002", "notes": "Catherine/Katherine + Mueller/Muller phonetic; DOB + nationality match -> real hit."},
    {"case_id": "PH-02", "counterparty_name": "Katharine Mueller",    "dob": None,         "nationality": None,        "category": "phonetic_variant", "expected_tier": "ADVERSE_NEWS",        "expected_confidence_band": "mid",  "expected_confidence_approx": 0.55, "related_corpus_id": "NEWS-002", "notes": "Phonetic name match but NO auxiliary data -> name-only, capped <=0.6."},
    {"case_id": "PH-03", "counterparty_name": "Jon Smyth",            "dob": "1975-03-12", "nationality": "Nigeria",   "category": "phonetic_variant", "expected_tier": "NAME_SIMILARITY_ONLY","expected_confidence_band": "mid",  "expected_confidence_approx": 0.55, "related_corpus_id": "SIM-001", "notes": "Phonetic near-match to a similarity-only decoy; attributes match the decoy, not any sanctioned party."},

    # === TRANSLITERATION VARIANTS (accents / spelling systems) ==============
    {"case_id": "TR-01", "counterparty_name": "José García",          "dob": "1979-11-05", "nationality": "Mexico",    "category": "transliteration_variant", "expected_tier": "ADVERSE_NEWS",       "expected_confidence_band": "high", "expected_confidence_approx": 0.90, "related_corpus_id": "NEWS-001", "notes": "Accented form of corpus 'Jose Garcia'; all attributes match."},
    {"case_id": "TR-02", "counterparty_name": "Muhammad Al Rashid",   "dob": "1972-09-30", "nationality": "Syria",     "category": "transliteration_variant", "expected_tier": "CONFIRMED_SANCTIONS","expected_confidence_band": "high", "expected_confidence_approx": 0.90, "related_corpus_id": "SANC-002", "notes": "Muhammad/Mohammed + hyphen dropped; attributes match sanctioned entry."},
    {"case_id": "TR-03", "counterparty_name": "Dmitry Ivanov",        "dob": "1980-01-22", "nationality": "Russia",    "category": "transliteration_variant", "expected_tier": "CONFIRMED_SANCTIONS","expected_confidence_band": "high", "expected_confidence_approx": 0.91, "related_corpus_id": "SANC-003", "notes": "Dmitry/Dmitri transliteration; attributes match."},
    {"case_id": "TR-04", "counterparty_name": "Sergey Morozov",       "dob": "1974-10-09", "nationality": "Russia",    "category": "transliteration_variant", "expected_tier": "ADVERSE_NEWS",       "expected_confidence_band": "high", "expected_confidence_approx": 0.89, "related_corpus_id": "NEWS-005", "notes": "Sergey/Sergei transliteration; attributes match."},

    # === AMBIGUOUS: name matches, only partial auxiliary data ===============
    {"case_id": "AM-01", "counterparty_name": "Elena Volkov",         "dob": None,         "nationality": "Russia",    "category": "ambiguous_partial", "expected_tier": "CONFIRMED_SANCTIONS", "expected_confidence_band": "mid", "expected_confidence_approx": 0.65, "related_corpus_id": "SANC-001", "notes": "Name + nationality match, DOB missing -> partial confirmation, mid confidence."},
    {"case_id": "AM-02", "counterparty_name": "Mohammed Al-Rashid",   "dob": None,         "nationality": None,        "category": "ambiguous_partial", "expected_tier": "CONFIRMED_SANCTIONS", "expected_confidence_band": "mid", "expected_confidence_approx": 0.55, "related_corpus_id": "SANC-002", "notes": "Name-only match to sanctioned entry, no auxiliary data -> capped <=0.6."},
    {"case_id": "AM-03", "counterparty_name": "Jose Garcia",          "dob": None,         "nationality": "Mexico",    "category": "ambiguous_partial", "expected_tier": "ADVERSE_NEWS",        "expected_confidence_band": "mid", "expected_confidence_approx": 0.60, "related_corpus_id": "NEWS-001", "notes": "Name + nationality match, DOB missing -> mid confidence."},
    {"case_id": "AM-04", "counterparty_name": "Viktor Petrov",        "dob": "1965-07-11", "nationality": None,        "category": "ambiguous_partial", "expected_tier": "CONFIRMED_SANCTIONS", "expected_confidence_band": "mid", "expected_confidence_approx": 0.70, "related_corpus_id": "SANC-004", "notes": "Name + DOB match exactly, nationality missing -> strong partial, mid-high."},

    # === TRUE NEGATIVES: not on any list ====================================
    {"case_id": "NM-01", "counterparty_name": "Oluwaseun Adeyemi",    "dob": "1991-03-04", "nationality": "Nigeria",       "category": "no_match", "expected_tier": "NO_MATCH", "expected_confidence_band": "zero", "expected_confidence_approx": 0.0, "related_corpus_id": None, "notes": "Not present in corpus."},
    {"case_id": "NM-02", "counterparty_name": "Lars Andersen",        "dob": "1988-07-19", "nationality": "Denmark",       "category": "no_match", "expected_tier": "NO_MATCH", "expected_confidence_band": "zero", "expected_confidence_approx": 0.0, "related_corpus_id": None, "notes": "Not present in corpus."},
    {"case_id": "NM-03", "counterparty_name": "Priya Nair",           "dob": "1993-12-25", "nationality": "India",         "category": "no_match", "expected_tier": "NO_MATCH", "expected_confidence_band": "zero", "expected_confidence_approx": 0.0, "related_corpus_id": None, "notes": "Shares surname with NEWS-009 Lakshmi Nair but different given name -> should not match."},
    {"case_id": "NM-04", "counterparty_name": "Amelia Clarke",        "dob": "1990-02-11", "nationality": "Australia",     "category": "no_match", "expected_tier": "NO_MATCH", "expected_confidence_band": "zero", "expected_confidence_approx": 0.0, "related_corpus_id": None, "notes": "Not present in corpus."},
    {"case_id": "NM-05", "counterparty_name": "Yuki Tanaka",          "dob": "1986-08-30", "nationality": "Japan",         "category": "no_match", "expected_tier": "NO_MATCH", "expected_confidence_band": "zero", "expected_confidence_approx": 0.0, "related_corpus_id": None, "notes": "Not present in corpus."},
    {"case_id": "NM-06", "counterparty_name": "Diego Fernandez",      "dob": "1984-04-09", "nationality": "Chile",         "category": "no_match", "expected_tier": "NO_MATCH", "expected_confidence_band": "zero", "expected_confidence_approx": 0.0, "related_corpus_id": None, "notes": "Not present in corpus."},

    # === Additional true positives (broaden tier + name coverage) ============
    {"case_id": "TP-08", "counterparty_name": "Kim Jong-Su",          "dob": "1958-12-03", "nationality": "North Korea", "category": "true_positive_exact", "expected_tier": "CONFIRMED_SANCTIONS", "expected_confidence_band": "high", "expected_confidence_approx": 0.94, "related_corpus_id": "SANC-005", "notes": "Exact confirmed-sanctions hit."},
    {"case_id": "TP-09", "counterparty_name": "Ahmed Farouk",         "dob": "1983-02-27", "nationality": "Egypt",       "category": "true_positive_exact", "expected_tier": "CONFIRMED_SANCTIONS", "expected_confidence_band": "high", "expected_confidence_approx": 0.93, "related_corpus_id": "SANC-007", "notes": "Exact confirmed-sanctions hit."},
    {"case_id": "TP-10", "counterparty_name": "Reza Motevalli",       "dob": "1978-03-14", "nationality": "Iran",        "category": "true_positive_exact", "expected_tier": "CONFIRMED_SANCTIONS", "expected_confidence_band": "high", "expected_confidence_approx": 0.93, "related_corpus_id": "SANC-010", "notes": "Exact confirmed-sanctions hit."},
    {"case_id": "TP-11", "counterparty_name": "Carlos Mendoza",       "dob": "1981-01-30", "nationality": "Colombia",    "category": "true_positive_exact", "expected_tier": "ADVERSE_NEWS",        "expected_confidence_band": "high", "expected_confidence_approx": 0.90, "related_corpus_id": "NEWS-006", "notes": "Exact adverse-news hit."},
    {"case_id": "TP-12", "counterparty_name": "Giovanni Russo",       "dob": "1969-12-01", "nationality": "Italy",       "category": "true_positive_exact", "expected_tier": "ADVERSE_NEWS",        "expected_confidence_band": "high", "expected_confidence_approx": 0.89, "related_corpus_id": "NEWS-010", "notes": "Exact adverse-news hit."},
    {"case_id": "TP-13", "counterparty_name": "Fatima Bourbon",       "dob": "1988-05-22", "nationality": "France",      "category": "true_positive_exact", "expected_tier": "ADVERSE_NEWS",        "expected_confidence_band": "high", "expected_confidence_approx": 0.89, "related_corpus_id": "NEWS-004", "notes": "Exact adverse-news hit."},

    # === Additional false positives =========================================
    {"case_id": "FP-07", "counterparty_name": "David Johnson",        "dob": "1955-01-01", "nationality": "Ghana",       "category": "false_positive_name_only", "expected_tier": "NAME_SIMILARITY_ONLY", "expected_confidence_band": "low", "expected_confidence_approx": 0.20, "related_corpus_id": "SIM-003", "notes": "Very common name matching a decoy; attributes diverge."},
    {"case_id": "FP-08", "counterparty_name": "Viktor Petrov",        "dob": "2000-03-03", "nationality": "Bulgaria",    "category": "false_positive_name_only", "expected_tier": "NAME_SIMILARITY_ONLY", "expected_confidence_band": "low", "expected_confidence_approx": 0.14, "related_corpus_id": "SANC-004", "notes": "Name matches sanctioned entry; DOB (35yr gap) + nationality diverge."},
    {"case_id": "FP-09", "counterparty_name": "Jose Garcia",          "dob": "1950-06-06", "nationality": "Spain",       "category": "false_positive_name_only", "expected_tier": "NAME_SIMILARITY_ONLY", "expected_confidence_band": "low", "expected_confidence_approx": 0.18, "related_corpus_id": "NEWS-001", "notes": "Name matches adverse-news entry; DOB + nationality diverge."},
    {"case_id": "FP-10", "counterparty_name": "Wei Zhang",            "dob": "1965-05-05", "nationality": "Taiwan",      "category": "false_positive_name_only", "expected_tier": "NAME_SIMILARITY_ONLY", "expected_confidence_band": "low", "expected_confidence_approx": 0.22, "related_corpus_id": "SIM-004", "notes": "Common Chinese name matching decoy; attributes diverge."},

    # === Additional phonetic / transliteration ==============================
    {"case_id": "PH-04", "counterparty_name": "Sergei Morosov",       "dob": "1974-10-09", "nationality": "Russia",      "category": "phonetic_variant",        "expected_tier": "ADVERSE_NEWS",        "expected_confidence_band": "high", "expected_confidence_approx": 0.86, "related_corpus_id": "NEWS-005", "notes": "Morozov/Morosov phonetic (z/s); attributes match."},
    {"case_id": "PH-05", "counterparty_name": "Elana Volkova",        "dob": "1968-04-15", "nationality": "Russia",      "category": "phonetic_variant",        "expected_tier": "CONFIRMED_SANCTIONS", "expected_confidence_band": "high", "expected_confidence_approx": 0.86, "related_corpus_id": "SANC-001", "notes": "Elana/Elena + feminine surname form Volkova; attributes match sanctioned entry."},
    {"case_id": "TR-05", "counterparty_name": "Ali Hassan Nasralla",  "dob": "1975-06-18", "nationality": "Lebanon",     "category": "transliteration_variant", "expected_tier": "CONFIRMED_SANCTIONS", "expected_confidence_band": "high", "expected_confidence_approx": 0.88, "related_corpus_id": "SANC-006", "notes": "Nasrallah/Nasralla transliteration; attributes match."},
    {"case_id": "TR-06", "counterparty_name": "Mohamad Al-Rashid",    "dob": None,         "nationality": "Syria",       "category": "transliteration_variant", "expected_tier": "CONFIRMED_SANCTIONS", "expected_confidence_band": "mid",  "expected_confidence_approx": 0.60, "related_corpus_id": "SANC-002", "notes": "Mohamad/Mohammed transliteration + nationality match but DOB missing -> mid."},

    # === Additional ambiguous partial =======================================
    {"case_id": "AM-05", "counterparty_name": "Katherine Muller",     "dob": "1985-03-19", "nationality": None,          "category": "ambiguous_partial", "expected_tier": "ADVERSE_NEWS", "expected_confidence_band": "mid", "expected_confidence_approx": 0.70, "related_corpus_id": "NEWS-002", "notes": "Name + DOB match exactly, nationality missing -> strong partial."},
    {"case_id": "AM-06", "counterparty_name": "Dmitri Ivanov",        "dob": None,         "nationality": None,          "category": "ambiguous_partial", "expected_tier": "CONFIRMED_SANCTIONS", "expected_confidence_band": "mid", "expected_confidence_approx": 0.50, "related_corpus_id": "SANC-003", "notes": "Very common name, no auxiliary data -> name-only, low end of mid, capped <=0.6."},

    # === Additional true negatives ==========================================
    {"case_id": "NM-07", "counterparty_name": "Thomas Mueller",       "dob": "1994-09-13", "nationality": "Austria",     "category": "no_match", "expected_tier": "NO_MATCH", "expected_confidence_band": "zero", "expected_confidence_approx": 0.0, "related_corpus_id": None, "notes": "Shares surname sound with NEWS-002 but different given name; not a listed party."},
    {"case_id": "NM-08", "counterparty_name": "Chen Wei Guo",         "dob": "1987-07-07", "nationality": "China",       "category": "no_match", "expected_tier": "NO_MATCH", "expected_confidence_band": "zero", "expected_confidence_approx": 0.0, "related_corpus_id": None, "notes": "Not present in corpus."},
]


def build_eval_set() -> dict:
    categories = sorted({c["category"] for c in EVAL_CASES})
    return {
        "version": EVAL_VERSION,
        "description": "Frozen, hand-labeled evaluation set. Ground truth is `expected_tier`. "
                       "Do not regenerate — edits should be deliberate and version-bumped.",
        "count": len(EVAL_CASES),
        "category_counts": {c: sum(1 for x in EVAL_CASES if x["category"] == c) for c in categories},
        "tier_counts": {
            t: sum(1 for x in EVAL_CASES if x["expected_tier"] == t)
            for t in ["CONFIRMED_SANCTIONS", "ADVERSE_NEWS", "NAME_SIMILARITY_ONLY", "NO_MATCH"]
        },
        "cases": EVAL_CASES,
    }


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    corpus = build_corpus()
    eval_set = build_eval_set()

    (DATA_DIR / "corpus.json").write_text(json.dumps(corpus, indent=2), encoding="utf-8")
    (DATA_DIR / "eval_set.json").write_text(json.dumps(eval_set, indent=2), encoding="utf-8")

    print(f"Corpus written: {corpus['count']} entries -> {corpus['tier_counts']}")
    print(f"Eval set written: {eval_set['count']} cases -> {eval_set['category_counts']}")


if __name__ == "__main__":
    main()
