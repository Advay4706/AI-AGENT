# Sanctions & Adverse-Media Name-Match Disambiguation Engine

Screens transaction counterparty names against a tiered watchlist corpus and decides
whether a name match is a **real hit** or a **coincidental collision** — the same
problem real AML screening tools (ComplyAdvantage, Refinitiv World-Check) solve.

## Architecture (two-stage, cheap-before-expensive)

1. **Deterministic candidate filter** (Step 2) — `rapidfuzz` (edit distance) +
   `jellyfish` (phonetic) + `unidecode` (normalization) narrows 60–100 corpus
   entries to a handful of plausible candidates in milliseconds, no LLM call.
2. **LLM disambiguation agent** (Step 3) — Claude reasons over the narrowed
   candidates plus any available DOB/nationality, returning a schema-enforced
   `ScreeningResult` with a calibrated confidence.

## Output schema

Every screening result validates against `ScreeningResult`:

| field | meaning |
|---|---|
| `counterparty_name` | the screened name |
| `match_tier` | `CONFIRMED_SANCTIONS` \| `ADVERSE_NEWS` \| `NAME_SIMILARITY_ONLY` \| `NO_MATCH` |
| `confidence` | 0.0–1.0, calibrated (see anchors below) |
| `matched_entry` | which corpus entry matched, if any |
| `reasoning` | why this tier/confidence, citing specific comparison points |

### Confidence calibration anchors

- **0.90+** — name matches AND at least one other identifying attribute (DOB, nationality) also matches.
- **0.5–0.7** — name matches closely, but only partial or missing auxiliary data to confirm.
- **below 0.3** — name matches (exactly or fuzzily) but at least one identifying attribute clearly diverges.
- A bare name-only string match with no auxiliary data available never exceeds **0.6**.

## Project status

- [x] **Step 1** — Synthetic corpus + frozen labeled eval set (`data/corpus.json`, `data/eval_set.json`)
- [ ] Step 2 — Deterministic candidate filter
- [ ] Step 3 — LLM disambiguation agent
- [ ] Step 4 — FastAPI endpoint + Gradio demo
- [ ] Step 5 — Evaluation harness, tests, Docker, final metrics

## Data

- `data/corpus.json` — 80-entry watchlist (28 CONFIRMED_SANCTIONS / 24 ADVERSE_NEWS / 28 NAME_SIMILARITY_ONLY), each with name + DOB + nationality.
- `data/eval_set.json` — 48 hand-labeled counterparty test cases spanning exact true positives, name-only false positives, phonetic variants, transliteration variants, ambiguous partial-data cases, and true negatives.

Both are **committed and frozen** so evaluation is reproducible. Regenerate deliberately with:

```bash
python src/generate_data.py
```

> All names, dates of birth, and nationalities are **synthetic** and generated for testing only.
