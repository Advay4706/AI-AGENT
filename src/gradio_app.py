"""
Step 4 — Minimal Gradio demo.

Wraps the same `screen()` pipeline as the FastAPI service in a quick visual UI.
Run:  python -m src.gradio_app   (requires ANTHROPIC_API_KEY for real matches)
"""

from __future__ import annotations

import gradio as gr

from src.disambiguator import screen

EXAMPLES = [
    ["Elena Volkov", "1968-04-15", "Russia"],       # true positive
    ["John Smith", "1990-11-02", "Canada"],          # name-only false positive
    ["Catherine Mueller", "1985-03-19", "Germany"],  # phonetic variant
    ["Muhammad Al Rashid", "1972-09-30", "Syria"],   # transliteration variant
    ["Oluwaseun Adeyemi", "1991-03-04", "Nigeria"],  # true negative
]


def run_screen(name: str, dob: str, nationality: str) -> dict:
    """Adapt the Gradio inputs to screen() and return a JSON-friendly dict."""
    name = (name or "").strip()
    if not name:
        return {"error": "Please enter a counterparty name."}
    try:
        result = screen(name, dob=dob or None, nationality=nationality or None)
        return result.model_dump()
    except Exception as exc:  # missing key / upstream error — surface it, don't crash the UI
        return {"error": f"{type(exc).__name__}: {exc}"}


def build_demo() -> "gr.Blocks":
    return gr.Interface(
        fn=run_screen,
        inputs=[
            gr.Textbox(label="Counterparty name", placeholder="e.g. Elena Volkov"),
            gr.Textbox(label="Date of birth (optional)", placeholder="YYYY-MM-DD"),
            gr.Textbox(label="Nationality (optional)", placeholder="e.g. Russia"),
        ],
        outputs=gr.JSON(label="ScreeningResult"),
        examples=EXAMPLES,
        title="Sanctions & Adverse-Media Name-Match Disambiguation",
        description="Screen a counterparty against the tiered watchlist. Returns the "
        "match tier, calibrated confidence, matched entry, and reasoning.",
        flagging_mode="never",
    )


demo = build_demo()

if __name__ == "__main__":
    demo.launch()
