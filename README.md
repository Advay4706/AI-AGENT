# Sanctions & Adverse-Media Name-Match Disambiguation Engine

Screens transaction counterparty names against a tiered watchlist corpus and decides
whether a name match is a **real hit** or a **coincidental collision** — the same
problem real AML screening tools (ComplyAdvantage, Refinitiv World-Check) solve.

## Architecture — two-stage, cheap-before-expensive

```
counterparty (name + optional DOB/nationality)
        │
        ▼
┌─────────────────────────────┐   deterministic, ~0.1 ms, no LLM
│ Stage 1: candidate filter   │   rapidfuzz (edit distance)
│  (src/candidate_filter.py)  │ + jellyfish (Metaphone/NYSIIS/Soundex)
│  80 entries → top-N (≤5)    │ + unidecode (accent/translit normalization)
└──────────────┬──────────────┘
        │ a handful of plausible candidates (or none)
        ▼
┌─────────────────────────────┐   one Claude call, only when needed
│ Stage 2: LLM disambiguation │   compares DOB/nationality vs each candidate,
│  (src/disambiguator.py)     │   applies calibration anchors, returns a
│  → ScreeningResult          │   schema-enforced verdict via tool-calling
└──────────────┬──────────────┘
        ▼
     ScreeningResult { match_tier, confidence, matched_entry, reasoning }
```

**Why two stages.** Fuzzy + phonetic + transliteration matching over the whole
corpus is cheap and deterministic, so Stage 1 runs on every screen in well under
a millisecond and throws away the ~75 entries that obviously don't matter. The
expensive, latency-heavy LLM reasoning in Stage 2 then only ever looks at a short
candidate list — and is skipped entirely when Stage 1 finds nothing (returns
`NO_MATCH` with no API call). This is the standard AML pattern: a broad, recall-
oriented filter feeds a precise, judgment-heavy disambiguator, so you pay for
intelligence only where a real decision has to be made.

## Output schema

Every screening result validates against `ScreeningResult` (`src/schema.py`):

| field | meaning |
|---|---|
| `counterparty_name` | the screened name |
| `match_tier` | `CONFIRMED_SANCTIONS` \| `ADVERSE_NEWS` \| `NAME_SIMILARITY_ONLY` \| `NO_MATCH` |
| `confidence` | 0.0–1.0, calibrated (anchors below) |
| `matched_entry` | which corpus entry matched, if any |
| `reasoning` | why this tier/confidence, citing specific comparison points |

The schema is enforced by **forced tool-calling** (a `strict` tool the model must
call), then re-checked with deterministic guardrails, so the calibration rules
hold even if the model drifts.

### Confidence calibration anchors

Embedded verbatim in the disambiguation prompt:

- **0.90+** — name matches AND at least one other identifying attribute (DOB, nationality) also matches.
- **0.5–0.7** — name matches closely, but only partial or missing auxiliary data to confirm.
- **below 0.3** — name matches (exactly or fuzzily) but at least one identifying attribute clearly diverges.
- A bare name-only string match with no auxiliary data available never exceeds **0.6**.

## Data

- `data/corpus.json` — 80-entry watchlist (CONFIRMED_SANCTIONS / ADVERSE_NEWS / NAME_SIMILARITY_ONLY), each with name + DOB + nationality.
- `data/eval_set.json` — 48 hand-labeled counterparty test cases: exact true positives, name-only false positives, phonetic variants, transliteration variants, ambiguous partial-data cases, and true negatives.

Both are **committed and frozen** so evaluation is reproducible. Regenerate deliberately with `python src/generate_data.py`. All names, DOBs, and nationalities are **synthetic**.

## Evaluation

`python -m src.evaluate` runs the full pipeline over the frozen eval set, computes
per-tier precision/recall/F1 with scikit-learn, does the same for a naive exact-
string baseline, and reports the false-positive reduction of the pipeline over that
baseline.

Two backends:
- **live** (`python -m src.evaluate`) — the real Claude agent; needs `ANTHROPIC_API_KEY`.
- **mock** (`python -m src.evaluate --mock`) — a deterministic, calibration-faithful
  rule-based stand-in that shares the entire pipeline code path except the LLM call.
  Proves the harness end-to-end with no key.

### Results — rule-based stand-in (`--mock`, 48 cases)

> These are **stand-in** numbers used to validate the harness, **not** the live LLM.
> The stand-in scores highly because the frozen eval set uses clear-cut DOB/nationality
> divergences that a rule engine handles by construction; the LLM's value is on
> messier, real-world inputs and nuanced confidence calibration. Regenerate live
> numbers with a key (see below).

| tier | pipeline P / R / F1 | baseline P / R / F1 |
|---|---|---|
| CONFIRMED_SANCTIONS | 1.00 / 1.00 / 1.00 | 0.73 / 0.69 / 0.71 |
| ADVERSE_NEWS | 1.00 / 1.00 / 1.00 | 0.80 / 0.62 / 0.70 |
| NAME_SIMILARITY_ONLY | 1.00 / 1.00 / 1.00 | 1.00 / 0.36 / 0.53 |
| NO_MATCH | 1.00 / 1.00 / 1.00 | 0.42 / 1.00 / 0.59 |
| **macro avg** | **1.00 / 1.00 / 1.00** | **0.74 / 0.67 / 0.63** |
| **accuracy** | **1.00** | **0.65** |

**False-positive reduction** (non-hit cases escalated to a hit tier): baseline **6**
→ pipeline **0** = **100% reduction**. The baseline's failures are the two failure
modes the pipeline is built to fix: it escalates same-name-different-person cases to
sanctions hits (false positives), and it misses phonetic/transliteration variants
(false negatives).

Reports are written to `data/eval_report_mock.json` / `data/eval_report_live.json`.

To generate the **live** numbers, set your key and run:

```bash
python -m src.evaluate
```

## Running

The LLM stage needs an `ANTHROPIC_API_KEY` (see `.env.example`). Names with no
plausible candidate short-circuit to `NO_MATCH` and work without a key.

FastAPI service (interactive docs at `/docs`):

```bash
uvicorn src.api:app --reload
```

```bash
curl -X POST http://127.0.0.1:8000/screen -H "Content-Type: application/json" -d "{\"name\":\"Elena Volkov\",\"dob\":\"1968-04-15\",\"nationality\":\"Russia\"}"
```

Gradio demo:

```bash
python -m src.gradio_app
```

### Docker (FastAPI service)

```bash
docker build -t aml-screen .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=sk-ant-... aml-screen
```

The image installs only the service runtime (`requirements-api.txt`) and bundles the frozen corpus.

## Tests

100 tests, all runnable with **no key and no network** (the LLM client is mocked):

```bash
pytest -q
```

## Layout

```
src/
  generate_data.py    Step 1  frozen corpus + eval set generator
  candidate_filter.py Step 2  deterministic fuzzy/phonetic filter
  schema.py           Step 3  ScreeningResult contract
  disambiguator.py    Step 3  LLM agent (tool-calling) + full pipeline screen()
  api.py              Step 4  FastAPI POST /screen
  gradio_app.py       Step 4  Gradio demo
  baseline.py         Step 5  naive exact-string baseline
  evaluate.py         Step 5  metrics harness (sklearn) + rule-based stand-in
tests/                        100 tests across Steps 2–5
Dockerfile / requirements-api.txt
data/corpus.json / data/eval_set.json
```

## Project status

- [x] **Step 1** — Synthetic corpus + frozen labeled eval set
- [x] **Step 2** — Deterministic candidate filter
- [x] **Step 3** — LLM disambiguation agent
- [x] **Step 4** — FastAPI endpoint + Gradio demo
- [x] **Step 5** — Evaluation harness, tests, Docker, metrics
