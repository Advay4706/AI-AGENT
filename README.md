# Sanctions & Adverse-Media Name-Match Disambiguation Engine

A two-stage AML screening system that decides whether a counterparty name
match is a real sanctions/adverse-media hit or a coincidental collision.

- **Stage 1** — deterministic fuzzy + phonetic candidate filter (no LLM)
- **Stage 2** — Claude-based disambiguation with calibrated confidence

Standalone implementation of the Adverse Media / Sanctions screening agent
(Stage 2) from a larger multi-agent AML pipeline.

## Status
🚧 Under construction — building step by step.
