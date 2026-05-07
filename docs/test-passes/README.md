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
| 2026-05-06 | `v4-qwen-7b-colab.md` | V4 prompt + Qwen 2.5-7B-Instruct on Colab T4 + BGE-large | **FRIA leak suppressed on Q2/Q3/Q4** (training-data prior is small-model-scale specific, not prompt-design). **Q5 clean and demo-quality** — explicitly distinguishes FRIA from DPIA, the conceptual conflation 1.5B couldn't manage. Q1 still single-obligation (architectural limit). Q4 retrieval-driven wrong-audience anchoring persists (upstream of LLM). |
| 2026-05-06 | `v4-qwen-3b-colab.md` | V4 prompt + Qwen 2.5-3B-Instruct on Colab T4 + BGE-large | **FRIA leak threshold sharpens to between 1.5B and 3B** (every leaking query at 1.5B is fully clean at 3B). **Q5 demo-quality achievable at 3B** — same FRIA-vs-DPIA distinction as 7B in ~5× less compute. **But 3B introduces new quality regressions** that 7B avoids: Q3 Section 3 broken (repeats law text), Q2 internal inconsistency between sections. Different failure modes have different scale thresholds. |
| 2026-05-06 | `v4-gemma-3-4b-colab.md` | Gemma 3-4B-it on Colab T4 + BGE-large; **two prompt configurations** (V4 long-system + Gemma-adapted short-system + rules-in-user) | **Cross-family FRIA-leak suppression confirmed:** Gemma 4B clean across Q2/Q3/Q4 with both prompts. **Prompt-format adaptation has real but mixed effects:** Q1 multi-facet engagement and Q3 sub-clause analysis substantially improved with Gemma-adapted prompt; Q2 and Q4 retrieval anchoring worsened. **Q5 conceptual failure persists across both prompts** — Gemma 4B accepts Novara's self-classification despite Annex III §4 mandating high-risk; failure is model-capability-bound, not prompt-bound. **Three orthogonal failure-mode axes now established:** scale (FRIA leak), prompt-format-per-family (multi-step instruction following), model-family-capability (Q5 conceptual reasoning). |
| 2026-05-07 | `v4-qwen-3b-ranking-strategies-comparison.md` | Qwen 3B + V4 + BGE-large + bge-reranker-base; **three ranking strategies compared** (BGE-only, rerank_only, RRF) | **Pure cross-encoder reranking is the empirical winner:** uniquely fixes Q4 wrong-audience anchoring (Article 50 para-1 transparency-to-natural-persons surfaced over Article 13 para-1 transparency-to-deployers), without regressions elsewhere. **RRF rejected:** dilutes the cross-encoder's targeted demotions, re-introduces the Q4 wrong-audience choice, and triggers a FRIA leak on Q3 by promoting DPIA-adjacent chunks. **Friendly grounding labels validated:** 4 of 5 queries had per-side confidence labels matching substantive truth. **FRIA leak refinement:** the leak isn't strictly a 1.5B-only issue — it can be context-triggered at 3B when retrieval surfaces FRIA-adjacent DPIA chunks prominently. Default switched from `rrf` to `rerank_only`. |

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
