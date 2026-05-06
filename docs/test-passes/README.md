# Test passes

Empirical snapshots from prompt / model / hardware variations on the simplified
architecture (`src/simplified.py`). Each file captures verbatim outputs across
the 5 test queries plus per-query observations. These are working artefacts —
they feed the report's appendix and the analytical narrative in
`../evaluation-findings.md`.

## Naming convention

`<prompt-version>-<model>-<hardware-or-context>-<descriptor>.md`

Examples:
- `v3-qwen-1.5b-local-baseline.md` — V3 prompt, Qwen 1.5B, local CPU, baseline run
- `v4-qwen-1.5b-prompt-hygiene.md` — V4 prompt iteration tested at the same model/hardware

## Index

| Date | File | Configuration | Key finding |
|---|---|---|---|
| 2026-05-05 | `v3-qwen-1.5b-local-baseline.md` | V3 prompt + Qwen 2.5-1.5B-Instruct on CPU + BGE-large retrieval | Q5 functionally correct (gap finding right with mid-paragraph factual error). Q1 truncated. Q2/Q3 "FRIA leak" in Section 3. Q4 retrieval drift to deep-fakes. **6 documented failure patterns.** Deployed default. |
| 2026-05-05 | `v4-qwen-1.5b-prompt-hygiene.md` | V4 prompt (V3 with topic-specific examples removed) + Qwen 2.5-1.5B-Instruct on CPU + BGE-large | FRIA leak **persists** despite removing FRIA mentions from system prompt — leak is training-data association, not prompt-design issue. Q5 regressed (clean but wrong conclusion). **Recommended: roll back to V3.** |

## What goes in here vs in `evaluation-findings.md`

- **Test pass docs** (this folder) — verbose, dated, raw. Verbatim outputs, retrieval scores, per-query observations. Reference material; the report's appendix pulls from these.
- **`evaluation-findings.md`** — curated analytical narrative across architectures and prompt iterations. The report's Critical Analysis dimension pulls from this.

When a finding is novel and important, it goes in `evaluation-findings.md` as a Stage. When it's a routine variant test (different model, different prompt iteration, different hardware), it goes here as its own file.

## Future test passes likely

Predicted as the project progresses:

- `v3-qwen-7b-colab.md` — same prompt, bigger model, GPU
- `v3-gemma-2b-colab.md` — Gemma comparison
- `v3-qwen-1.5b-bge-base.md` — embedding-model variant (cheaper BGE, if tested)
- `query-expansion-attempt.md` — if we implement Category 1c mitigation
- `query-decomposition-attempt.md` — if we implement Category 2 mitigation

Each one is small, focused, dated, and named clearly. They accumulate as
comparison points without bloating the main `docs/` folder.
