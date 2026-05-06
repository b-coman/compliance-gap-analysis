# Test pass — V4 prompt on Qwen 3B (Colab GPU)

> Same V4 prompt and BGE retrieval as the V4 1.5B local and V4 7B Colab
> test passes; intermediate model size between 1.5B (current local default,
> exhibits FRIA leak) and 7B (suppresses leak, demo-quality Q5). Tests
> whether the FRIA-leak suppression threshold sits between 1.5B and 3B,
> or between 3B and 7B.
>
> Run date: 2026-05-06.

---

## Hypothesis going in

V4 7B (`v4-qwen-7b-colab.md`) suppressed the FRIA leak fully and produced
demo-quality Q5 output (FRIA-vs-DPIA distinction). V4 1.5B leaked
verbatim. We did not know whether the leak suppression was gradual
(attenuated at 3B, fully gone at 7B) or threshold-like (off until some
size, then fully on).

Two predicted outcomes possible:
- **Gradual:** 3B leaks but less severely; 7B clean. Threshold somewhere
  in the upper half of the 3B–7B range.
- **Threshold:** 3B is clean already (or 3B still leaks fully). Threshold
  is between 1.5B and 3B (or between 3B and 7B). Sharper claim either way.

---

## Configuration captured

| Component | Value |
|---|---|
| Architecture | `src/simplified.py` (single retrieve + single LLM call) |
| Embedding model | `BAAI/bge-large-en-v1.5` |
| Embedding device | CUDA |
| Retrieval top-k | 5 (regulation), 5 (DEP + DEP_EXTRAS) |
| Operational corpus (ICO) | Excluded at query time |
| LLM | `Qwen/Qwen2.5-3B-Instruct` |
| LLM device | CUDA (full GPU residence, no offloading) |
| LLM dtype | float16 |
| Generation: do_sample | False (deterministic) |
| Generation: repetition_penalty | 1.05 |
| Generation: max_new_tokens | 550 |
| Prompt | V4 (current default in `src/simplified.py`) |
| Hardware | Colab Tesla T4 (16 GB GPU, ~13 GB CPU RAM) |

**Latency profile:** Q1 took 139.7s (probable cold-start — first query of
the session, GPU kernels JIT-compiling on T4). Q2-Q5 each completed in
10.9–15.0s. Total run: ~3 minutes including BGE corpus embedding (one-time
~36s) and model download (~30s). **3B fp16 fits T4 fully — no CPU
offloading**, no AWQ runtime needed.

---

## V4 1.5B vs V4 3B vs V4 7B across all 5 queries

| Query | V4 1.5B local | V4 3B Colab (this run) | V4 7B Colab |
|---|---|---|---|
| Q1 multi-facet | FRIA leak; output truncated | Single-obligation (Article 14 para-5); **no leak** | Single-obligation (Article 26 para-9); no leak |
| Q2 red-teaming | FRIA leak verbatim in Section 3 | **No leak.** Internal inconsistency: Section 2 cites §3.4 Red-Teaming policy; Section 3 says *"policy does not explicitly mention red-teaming"* | No leak. Different reading error: *"policy does not mention testing prior to deployment"* (contradicts §3.4's "Gate G2") |
| Q3 Article 22 | FRIA leak (fabricated *"FRIA required by UK GDPR Article 22"*) | **No leak.** Section 3 broken — repeats Article 22 text from Section 1 verbatim instead of stating a gap | No leak. Substantively correct: identifies the three missing sub-clauses (human intervention, contest, point of view) |
| Q4 transparency | FRIA leak persisted in Section 3 | **No leak.** Wrong-audience anchoring (Article 13 para-1, transparency-to-deployers) | No leak. Same wrong-audience anchoring |
| Q5 FRIA target | Wrong conclusion (accepted Novara's "Standard AI Feature" self-classification) | **Clean — distinguishes FRIA from DPIA explicitly:** *"focuses on data protection impact assessments rather than fundamental rights impact assessments"* | Clean — same FRIA-vs-DPIA distinction |

---

## Verbatim outputs (per-query)

### Q1 — multi-facet

**Latency:** 139.7s (cold-start; subsequent queries 10–15s).

**LLM output:**
```
### What the law requires

[regulation/eu-ai-act-2024-1689#article-14-para-5] For high-risk AI
systems referred to in point 1(a) of Annex III, the measures referred to
in paragraph 3 of this Article shall be such as to ensure that, in
addition, no action or decision is taken by the deployer on the basis
of the identification resulting from the system unless that
identification has been separately verified and confirmed by at least
two natural persons.

### What the policy says

[deployer-extras/novara-talentlens-dpia#5-1-mitigations-applied-per-risk]
Novara TalentLens applies bias auditing methodology at each version
release to ensure demographic parity ≥ 0.85, which mitigates the risk
of biased outcomes.

### Gap

[regulation/eu-ai-act-2024-1689#article-14-para-5] The policy does not
address ensuring that no action or decision is taken by the deployer on
the basis of the identification resulting from the system unless that
identification has been separately verified and confirmed by at least
two natural persons.
```

**Observations:**
- Single-obligation output (Article 14 human oversight, biometric two-person verification clause). Same architectural single-call limit as V4 1.5B and V4 7B.
- No FRIA leak — cleanly suppressed at 3B.
- Section 2 retrieval picks up a DPIA bias-auditing chunk that's only loosely related (demographic parity ≥ 0.85) — the model uses what's available rather than declining to engage. Note the mismatch is *retrieval-driven*: Section 1 (Article 14 oversight) and Section 2 (bias auditing) are different topics; the LLM presents them as if they're the same matter.
- Compare V3 1.5B Q1 which also surfaced Article 14 (1 of 4 correctly) — the 3B model converges on the same single article.

---

### Q2 — red-teaming

**Latency:** 13.2s.

**LLM output:**
```
### What the law requires
[regulation/eu-ai-act-2024-1689#article-26-para-1] Deployers of
high-risk AI systems shall take appropriate technical and organizational
measures to ensure they use such systems in accordance with the
instructions for use accompanying the systems, pursuant to paragraphs
3 and 6.

### What the policy says
[deployer/novara-ai-policy-v3.1#section-3-4] Novara AI Policy v3.1 §3.4
Red-Teaming and Safety Testing mandates that every AI System undergo a
structured red-team exercise before Gate G2, with specific requirements
for high-risk AI features.

### Gap
[regulation/eu-ai-act-2024-1689#article-26-para-1] The policy does not
explicitly mention red-teaming or any other form of testing required by
the EU AI Act Article 26 (para-1).
```

**Observations:**
- **No FRIA leak.** Compare V4 1.5B which produced *"policy does not address performing a fundamental rights impact assessment"* on this same query.
- **Internal inconsistency.** Section 2 explicitly says §3.4 mandates red-teaming before Gate G2, with requirements for high-risk AI features. Section 3 then claims *"the policy does not explicitly mention red-teaming"* — directly contradicting Section 2.
- The 3B model picks up surface signals (Article 26 → "instructions for use") but doesn't carry the thread across sections. 7B made a different reading error here; 1.5B leaked. **3B has its own failure profile.**

---

### Q3 — GDPR Article 22 sub-clauses

**Latency:** 15.0s.

**LLM output:**
```
### What the law requires

[regulation/uk-gdpr-art-22#whole] The data subject shall have the right
not to be subject to a decision based solely on automated processing,
including profiling, which produces legal effects concerning him or her
or similarly significantly affects him or her.

### What the policy says

[deployer/novara-ai-policy-v3.1#section-4-3] Novara AI's standard data
subject rights procedures (NAI-POL-0009) apply to all AI-processed
personal data. In addition, the following AI-specific obligations apply:
Right to Explanation Where an AI System makes or significantly influences
a decision affecting a data subject, the subject may request a
plain-language explanation of the factors that influenced the output.
Product teams must implement measures to ensure compliance with this right.

### Gap

[regulation/uk-gdpr-art-22#whole] The data subject shall have the right
not to be subject to a decision based solely on automated processing,
including profiling, which produces legal effects concerning him or her
or similarly significantly affects him or her.
```

**Observations:**
- **No FRIA leak.** Compare V4 1.5B which fabricated *"FRIA required by UK GDPR Article 22"*.
- **Section 3 is broken.** Instead of stating the gap, the model copy-pasted the Article 22 text from Section 1 into Section 3. Pure repetition; no analytical content.
- This is a **quality regression vs 7B**: 7B correctly identified the three missing sub-clauses (human intervention, contest, point of view). 3B can't do that level of sub-clause analysis.
- Speculation: the model recognises a gap exists but lacks the reasoning depth to articulate which specific sub-clauses are missing, so it falls back to repetition. A multi-step reasoning capability that emerges at larger scale.

---

### Q4 — transparency (ambiguous framing)

**Latency:** 12.9s.

**LLM output:**
```
### What the law requires

[regulation/eu-ai-act-2024-1689#article-13-para-1] High-risk AI systems
shall be designed and developed in such a way as to ensure that their
operation is sufficiently transparent to enable deployers to interpret
a system's output and use it appropriately.

### What the policy says

[deployer-extras/novara-talentlens-dpia#2-1-what-talentlens-does]
TalentLens is an AI-powered CV-screening and candidate-ranking service.
It provides structured assessments of skill alignment, identified
strengths, identified gaps, and a relative ranking when multiple
candidates are submitted, all of which are delivered to the Customer's
recruiter for human review.

### Gap

[regulation/eu-ai-act-2024-1689#article-13-para-1] The policy does not
address ensuring that the operation of TalentLens is sufficiently
transparent to enable deployers to interpret its output and use it
appropriately.
```

**Observations:**
- **No FRIA leak.**
- **Same wrong-audience anchoring as 7B.** Article 13 para-1 is about transparency *to deployers* (the receiving organisation). The query asks about transparency *to candidates* (data subjects). BGE retrieval ranks Article 13 para-1 high on the word "transparency"; the LLM uses what's surfaced.
- Confirms the **retrieval-side failure persists at every model size tested.** Q4 cannot be fixed by scaling the LLM; the mitigation is upstream (query expansion / disambiguation, Category 1 mitigation in `evaluation-findings.md`).

---

### Q5 — FRIA target (the headline)

**Latency:** 10.9s.

**LLM output:**
```
### What the law requires
[regulation/eu-ai-act-2024-1689#article-27-para-1] Deployers of high-risk
AI systems, including those intended to be used in the areas listed in
point 2 of Annex III, must perform a fundamental rights impact
assessment prior to deploying such systems.

### What the policy says
[deployer-extras/novara-talentlens-dpia#preamble] The policy document
for Novara TalentLens focuses on data protection impact assessments
rather than fundamental rights impact assessments.

### Gap
[regulation/eu-ai-act-2024-1689#article-27-para-1] The policy does not
address performing a fundamental rights impact assessment, which is
required by EU AI Act Article 27.
```

**Observations:**
- **Clean, demo-quality output.** Section 1 cites Article 27 para-1 with a complete obligation sentence. Section 2 explicitly distinguishes FRIA from DPIA (the conceptual conflation 1.5B couldn't make). Section 3 states the gap clearly.
- **Same level of clarity as V4 7B.** The FRIA-vs-DPIA distinction does not require 7B; 3B suffices. **Significant for resource-budget claims — the demo Q5 output works at 3B, not just at 7B.**
- Slight phrasing differences from 7B (3B says *"focuses on... rather than..."*; 7B says *"...which is a different type of assessment"*). Substance equivalent; both are defensible.

---

## What this tells us — three findings that sharpen Stage 8

### 1. The FRIA leak threshold sits between 1.5B and 3B

Every query that leaked at V4 1.5B (Q2, Q3, Q4) is *fully* clean at 3B. The training-data prior is suppressed at 3B already; we don't need 7B to fix this specific failure mode. **Empirically tighter than the previous "between 1.5B and 7B" bound.**

This narrows the report's claim from *"the leak is a small-model problem fixed by scaling to 7B"* to *"the leak is a 1.5B-specific issue fixed by 3B; below the threshold the model produces the FRIA association regardless of prompt design."* The threshold is sharper than gradual.

### 2. Q5 demo-quality is achievable at 3B

The most striking finding from this run. Qwen 3B distinguishes FRIA from DPIA cleanly — same conceptual quality as 7B, in 5× less compute (3B vs 7B + offloading) and ~30× less inference latency (10s vs 300s on T4).

This means the demo-day Q5 output does not require running the 7B model on Colab during the demo. **A 3B model on a colleague's laptop GPU could produce the same headline gap finding.** Important if Daria wants the demo to run end-to-end on hardware she controls during the viva.

### 3. 3B introduces *new* failure modes that 7B doesn't have

- **Q3 Section 3 broken** — instead of stating the gap, 3B copy-pasted the law text from Section 1 verbatim into Section 3. 7B correctly produced sub-clause analysis here.
- **Q2 internal inconsistency** — Section 2 says §3.4 mandates red-teaming before Gate G2; Section 3 immediately contradicts it. 7B made a different reading error but kept Section 2 and Section 3 logically aligned.

These are *quality* regressions at 3B that 7B doesn't show. The 3B model picks up surface signals but doesn't always carry consistent reasoning across sections. Some multi-step reasoning capability emerges only at larger scale.

**Net pattern:** the FRIA leak is one failure mode that suppresses early (3B). Other failure modes (sub-clause analysis, section-consistency) need larger models. Different failure modes have different scale thresholds — the report's narrative is richer than a single "go bigger" claim.

---

## Status after this pass

- The simplified path now has **three Colab data points** plus the local 1.5B baseline + V3/V4 prompt iterations: a 4-point empirical comparison at the model-scale level.
- **Recommended demo Q5 output:** either 3B or 7B output works. 3B is cheaper to reproduce; 7B is slightly more polished phrasing.
- **Recommended next test pass:** Gemma 2-2B (alternative family at ~similar size to Qwen 3B). If Gemma 2B *also* suppresses the leak, training-corpus-family is irrelevant — the threshold is purely parameter-count. If Gemma 2B *leaks* (despite being similar size), suppression is Qwen-specific. Either result is publishable.
- **Documented limitation:** Q1 cold-start latency anomaly (139.7s on first query, 10–15s on subsequent queries). Likely T4-side JIT compilation; not a model property.
