"""
Step 5 — Evaluation harness.

Runs the full two-stage pipeline against the frozen Step-1 eval set, computes
per-tier precision/recall/F1 via scikit-learn, does the same for the naive
exact-string baseline, and reports the false-positive reduction of the pipeline
over that baseline.

Two disambiguation backends:
* `disambiguate`        — the real Claude agent (Step 3). Needs ANTHROPIC_API_KEY.
* `rule_based_disambiguate` — a deterministic stand-in that mirrors the
  calibration logic (compare DOB/nationality to the top candidate). Used to prove
  the harness end-to-end with no key. Select it with `--mock`.

Both backends share the exact same code path up to the decision: Step-2 filter,
the `Candidate` list, the `ScreeningResult` schema, and the guardrails.

Usage:
    python -m src.evaluate --mock     # rule-based stand-in, no key
    python -m src.evaluate            # live Claude agent (needs key)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Optional, Sequence

from sklearn.metrics import precision_recall_fscore_support

from src.baseline import baseline_predict
from src.candidate_filter import Candidate, find_candidates
from src.disambiguator import _apply_guardrails, disambiguate
from src.schema import ScreeningResult

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TIERS = ["CONFIRMED_SANCTIONS", "ADVERSE_NEWS", "NAME_SIMILARITY_ONLY", "NO_MATCH"]
# Tiers that represent a real escalation (a "hit"). Everything else is "clear".
HIT_TIERS = {"CONFIRMED_SANCTIONS", "ADVERSE_NEWS"}
STRONG_NAME_SCORE = 0.72

Disambiguator = Callable[..., ScreeningResult]


def load_eval_cases() -> list[dict]:
    data = json.loads((DATA_DIR / "eval_set.json").read_text(encoding="utf-8"))
    return data["cases"]


# ---------------------------------------------------------------------------
# Rule-based disambiguation stand-in (deterministic, no LLM)
# ---------------------------------------------------------------------------
def _eq(a: Optional[str], b: Optional[str]) -> bool:
    return bool(a) and bool(b) and a.strip().casefold() == b.strip().casefold()


def rule_based_disambiguate(
    name: str,
    candidates: Sequence[Candidate],
    dob: Optional[str] = None,
    nationality: Optional[str] = None,
) -> ScreeningResult:
    """A calibration-faithful, LLM-free disambiguator (drop-in for `disambiguate`)."""
    if not candidates:
        return ScreeningResult(
            counterparty_name=name, match_tier="NO_MATCH", confidence=0.0,
            matched_entry=None, reasoning="No candidates cleared the filter.",
        )

    top = candidates[0]
    e = top.entry
    tier_c = e["tier"]
    strong_name = top.combined_score >= STRONG_NAME_SCORE
    dob_match = _eq(dob, e["dob"]) if dob else False
    nat_match = _eq(nationality, e["nationality"])
    dob_div = bool(dob) and not _eq(dob, e["dob"])
    nat_div = bool(nationality) and not _eq(nationality, e["nationality"])
    has_aux = bool(dob) or bool(nationality)

    if not strong_name:
        # Weak name overlap (e.g. shared surname only). Only a real match if both
        # auxiliary attributes corroborate; otherwise it's not this person.
        if dob_match and nat_match:
            tier, conf, reason = tier_c, 0.60, "Weak name overlap but DOB and nationality both corroborate."
        else:
            return ScreeningResult(
                counterparty_name=name, match_tier="NO_MATCH", confidence=0.0,
                matched_entry=None,
                reasoning="Name overlap is weak and auxiliary attributes do not corroborate a match.",
            )
    elif dob_div or nat_div:
        tier, conf, reason = "NAME_SIMILARITY_ONLY", 0.15, "Name matches but DOB and/or nationality clearly diverge."
    elif not has_aux:
        tier, conf, reason = tier_c, 0.55, "Name matches closely but no auxiliary data to confirm."
    elif dob_match and nat_match:
        tier, conf, reason = tier_c, 0.95, "Name plus DOB and nationality all match."
    else:
        tier, conf, reason = tier_c, 0.65, "Name matches and one auxiliary attribute corroborates; the other is missing."

    result = ScreeningResult(
        counterparty_name=name, match_tier=tier, confidence=conf,
        matched_entry=e["id"], reasoning=reason,
    )
    return _apply_guardrails(result, dob, nationality)


# ---------------------------------------------------------------------------
# Prediction runs
# ---------------------------------------------------------------------------
def pipeline_predict(case: dict, disambiguator: Disambiguator, top_n: int = 5) -> str:
    candidates = find_candidates(case["counterparty_name"], top_n=top_n)
    result = disambiguator(
        case["counterparty_name"], candidates,
        dob=case["dob"], nationality=case["nationality"],
    )
    return result.match_tier


def collect_predictions(
    cases: list[dict], disambiguator: Disambiguator, top_n: int = 5
) -> tuple[list[str], list[str], list[str]]:
    y_true = [c["expected_tier"] for c in cases]
    y_pipeline = [pipeline_predict(c, disambiguator, top_n=top_n) for c in cases]
    y_baseline = [baseline_predict(c["counterparty_name"])[0] for c in cases]
    return y_true, y_pipeline, y_baseline


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=TIERS, average=None, zero_division=0
    )
    per_tier = {
        tier: {
            "precision": round(float(p[i]), 3),
            "recall": round(float(r[i]), 3),
            "f1": round(float(f1[i]), 3),
            "support": int(support[i]),
        }
        for i, tier in enumerate(TIERS)
    }
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=TIERS, average="macro", zero_division=0
    )
    accuracy = sum(int(a == b) for a, b in zip(y_true, y_pred)) / len(y_true)
    return {
        "per_tier": per_tier,
        "macro": {
            "precision": round(float(macro_p), 3),
            "recall": round(float(macro_r), 3),
            "f1": round(float(macro_f1), 3),
        },
        "accuracy": round(accuracy, 3),
    }


def count_false_positives(y_true: list[str], y_pred: list[str]) -> int:
    """A false positive = a non-hit case escalated to a hit tier."""
    return sum(
        1 for t, pr in zip(y_true, y_pred)
        if t not in HIT_TIERS and pr in HIT_TIERS
    )


def false_positive_reduction(y_true, y_baseline, y_pipeline) -> dict:
    base_fp = count_false_positives(y_true, y_baseline)
    pipe_fp = count_false_positives(y_true, y_pipeline)
    reduction_pct = round((base_fp - pipe_fp) / base_fp * 100, 1) if base_fp else 0.0
    return {
        "baseline_false_positives": base_fp,
        "pipeline_false_positives": pipe_fp,
        "reduction_pct": reduction_pct,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _format_metrics_table(title: str, metrics: dict) -> str:
    lines = [f"{title}", "  tier                    prec   recall   f1    support"]
    for tier, m in metrics["per_tier"].items():
        lines.append(
            f"  {tier:22s} {m['precision']:.3f}  {m['recall']:.3f}  {m['f1']:.3f}    {m['support']}"
        )
    mac = metrics["macro"]
    lines.append(f"  {'MACRO AVG':22s} {mac['precision']:.3f}  {mac['recall']:.3f}  {mac['f1']:.3f}")
    lines.append(f"  accuracy: {metrics['accuracy']:.3f}")
    return "\n".join(lines)


def run_evaluation(mock: bool = False, top_n: int = 5, write: bool = True) -> dict:
    disambiguator: Disambiguator = rule_based_disambiguate if mock else disambiguate
    backend = "rule_based_stand_in" if mock else "claude_llm"

    cases = load_eval_cases()
    y_true, y_pipeline, y_baseline = collect_predictions(cases, disambiguator, top_n=top_n)

    pipeline_metrics = compute_metrics(y_true, y_pipeline)
    baseline_metrics = compute_metrics(y_true, y_baseline)
    fp = false_positive_reduction(y_true, y_baseline, y_pipeline)

    report = {
        "backend": backend,
        "eval_case_count": len(cases),
        "pipeline": pipeline_metrics,
        "baseline": baseline_metrics,
        "false_positive_reduction": fp,
    }

    print(f"\n=== Evaluation ({backend}, {len(cases)} cases) ===\n")
    print(_format_metrics_table("PIPELINE (Step 2 filter + disambiguation)", pipeline_metrics))
    print()
    print(_format_metrics_table("BASELINE (naive exact-string match)", baseline_metrics))
    print()
    print("FALSE-POSITIVE REDUCTION (non-hit cases escalated to a hit tier):")
    print(f"  baseline false positives: {fp['baseline_false_positives']}")
    print(f"  pipeline false positives: {fp['pipeline_false_positives']}")
    print(f"  reduction: {fp['reduction_pct']}%")

    if write:
        suffix = "mock" if mock else "live"
        out = DATA_DIR / f"eval_report_{suffix}.json"
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nReport written: {out}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the screening evaluation harness.")
    parser.add_argument(
        "--mock", action="store_true",
        help="Use the rule-based stand-in disambiguator (no API key required).",
    )
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--no-write", action="store_true", help="Do not write the JSON report.")
    args = parser.parse_args()
    run_evaluation(mock=args.mock, top_n=args.top_n, write=not args.no_write)


if __name__ == "__main__":
    main()
