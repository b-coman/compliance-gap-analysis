# Simplified Architecture — Baseline Run

> Empirical baseline: all 5 hand-written test queries run through the simplified architecture (`src/simplified.py`) under the deployed default configuration. Captured for comparison when we test better models, prompt iterations, or hardware changes.
>
> Distinct from `docs/evaluation-findings.md`, which is the analytical journey across chain and simplified architectures. This document is a **reference snapshot** for the simplified path, not a narrative.
>
> Run date: 2026-05-05. Total elapsed: ~3.6 minutes for all 5 queries (one-time BGE corpus embedding cost amortised; Qwen lazy-loaded on first call).

---

## Configuration captured

| Component | Value |
|---|---|
| Architecture | `src/simplified.py` (single retrieve + single LLM call) |
| Embedding model | `BAAI/bge-large-en-v1.5` |
| Embedding query prefix | `"Represent this sentence for searching relevant passages: "` |
| Embedding device | MPS (Apple Silicon) |
| Embedding cache | `embeddings_bge/embeddings.npy` |
| Retrieval top-k | 5 (regulation), 5 (DEP + DEP_EXTRAS) |
| Operational corpus (ICO) | Excluded at query time |
| LLM | `Qwen/Qwen2.5-1.5B-Instruct` |
| LLM device | CPU |
| LLM dtype | float32 |
| Generation: do_sample | False (deterministic) |
| Generation: repetition_penalty | 1.05 |
| Generation: max_new_tokens | 550 |
| Prompt | V3 refined (see `simplified.py` SYSTEM_PROMPT + `_user_message`) |
| LLM cache | Cleared before run (cold) |

---

## Per-query baseline results

### Q1 — multi-facet (Articles 13/14/22/26)

**Query:**
> TalentLens compliance under EU AI Act Annex III §4 — am I covered on Article 13 deployer instructions, Article 14 human oversight, Article 26 logs and worker information, and the related Article 22 GDPR automated-decisions duties? Where are the gaps?

**Top-5 regulation retrieval:**
```
0.740  EU AI Act Article 26 (para-9)
0.724  EU AI Act Article 22 (para-3)        ← AI Act Art 22 (authorised reps), not GDPR Art 22
0.719  EU AI Act Article 43 (para-3)
0.718  EU AI Act Article 14 (para-5)        ← correct (human oversight, biometric)
0.718  EU AI Act Article 78 (para-3)
```
**Article 13 missed entirely. AI Act Art 22 retrieved instead of GDPR Art 22.**

**Top-5 deployer retrieval:**
```
0.744  Novara TalentLens Model Card — 7.2 Customer responsibilities
0.735  Novara TalentLens DPIA — Preamble
0.729  Novara TalentLens DPIA — 5.1 Mitigations applied per risk
0.727  Novara TalentLens Model Intake Assessment — 7. Risk Classification
0.720  Novara 2025 AI Governance Report — 5.4 Other developments
```

**LLM output (verbatim, truncated):**
```
### What the law requires

1. **Article 13 deployer instructions**: Deployers of high-risk AI systems must use the
   information provided under Article 13 of the EU AI Act to comply with their obligation
   to carry out a data protection impact assessment (DPIA)...
2. **Article 14 human oversight**: For high-risk AI systems referred to in point 1(a) of
   Annex III, the measures referred to in paragraph 3 of this Article shall be such as to
   ensure that, in addition, no action or decision is taken by the deployer on the basis
   of the identification resulting from the system unless that identification has been
   separately verified and confirmed by at least two natural persons.
3. **Article 26 logs and worker information**: The authorised representative shall provide
   a copy of the mandate to the market surveillance authorities upon request...
4. **Article 22 GDPR automated-decisions duties**: The authorised representative shall
   perform the tasks specified in the mandate received from the provider...

### What the policy says

1. **Novara TalentLens Model Card — 7.2 Customer responsibilities**: Customer organisations
   deploying TalentLens are responsible for configuring HITL review processes...
2. **Novara TalentLens DPIA — Preamble**: This DPIA is Confidential — Internal Use Only...
3. **Novara TalentLens DPIA — 5.1                              ← TRUNCATED HERE
```

**Observations:**
- **Output truncated** — hit `max_new_tokens=550` limit before reaching the Gap section. Multi-facet queries need a larger budget.
- Section 1 quality is mixed across the four provisions:
  - Article 13: quoted Article 26 para-9 text (about DPIA, wrong topic). Model substituted "Article 13" in the heading but the quoted obligation is from Article 26.
  - Article 14: **correct** quote from Article 14 para-5 about biometric two-person verification.
  - Article 26: quoted authorised-representative text (from AI Act Art 22). Wrong topic.
  - Article 22 GDPR: same authorised-representative text. Fundamentally wrong — that's AI Act Art 22, not GDPR Art 22.
- 1 of 4 provisions correctly handled (Article 14). Same multi-facet failure mode as previous Q1 runs (`evaluation-findings.md` Stage 6).
- Generation time: 75.9s — significantly longer than other queries due to multi-section format.

**Runtime:** 85.3s (incl. Qwen first-load).

---

### Q2 — red-teaming (single-facet, adequate expected)

**Query:**
> Does our policy address the red-teaming requirements before deploying a high-risk AI system to production?

**Top-5 regulation retrieval:**
```
0.743  EU AI Act Article 26 (para-1)        ← wrong topic (instructions for use)
0.743  EU AI Act Article 26 (para-7)        ← wrong topic
0.730  EU AI Act Article 14 (para-4)        ← wrong topic
0.729  EU AI Act Article 26 (para-8)        ← wrong topic
0.719  EU AI Act Article 9 (para-8)         ← correct (risk management) but rank #5
```
**Article 9 (where red-teaming requirements live) surfaced at rank #5; Article 26 dominates.**

**Top-5 deployer retrieval:**
```
0.810  Novara AI Policy v3.1 §3.4 Red-Teaming and Safety Testing  ← strong correct hit
0.673  Novara 2025 AI Governance Report — 6.1 Training and certification
0.657  Novara 2025 AI Governance Report — 2.1 Risk register status
0.656  Novara AI Policy v3.1 §3.5 Human-in-the-Loop Requirements
0.649  Novara AI Policy v3.1 §5.3 Incident Response
```

**LLM output (verbatim):**
```
### What the law requires

[regulation/eu-ai-act-2024-1689#article-26-para-1] EU AI Act Article 26 (para-1)

Deployers of high-risk AI systems shall take appropriate technical and organisational
measures to ensure they use such systems in accordance with the instructions for use
accompanying the systems, pursuant to paragraphs 3 and 6.

### What the policy says

[deployer/novara-ai-policy-v3.1#section-3-4] Novara AI Policy v3.1 §3.4 Red-Teaming and Safety Testing

Every AI System must undergo a structured red-team exercise before Gate G2. Red-teaming
must be conducted by a team that did not build the model under review. For High-Risk AI
Features, at least one red-teamer must be external to Novara AI...

### Gap

[deployer/novara-ai-policy-v3.1#section-3-4] Novara AI Policy v3.1 §3.4 Red-Teaming and Safety Testing

The policy does not address performing a fundamental rights impact assessment.
```

**Observations:**
- **Section 3 contains a notable contamination: the model wrote "fundamental rights impact assessment" in the Gap section despite Q2 asking about red-teaming.** This phrase is mentioned as an example in the SYSTEM_PROMPT. The small model treated the example as content to include rather than as instruction. This is the same kind of leak we saw with the bracketed placeholder in earlier prompts — small-model prompt fragility. We refer to this below as **the "FRIA leak"**.
- Section 1 quotes Article 26 (instructions for use) — wrong topic for red-teaming. Article 9 was retrieved at rank #5; the model anchored on the higher-ranked Article 26.
- Section 2 correctly quotes §3.4 Red-Teaming.
- Output is internally contradictory: Section 2 shows policy addresses red-teaming; Section 3 says policy doesn't address FRIA (a different topic entirely).

**Runtime:** 47.1s.

---

### Q3 — Article 22 sub-clauses (single-facet, partial expected)

**Query:**
> How do we meet GDPR Article 22 requirements on solely automated decisions affecting candidates — explicit consent, right to obtain human intervention, right to contest the decision, and right to express their point of view?

**Top-5 regulation retrieval:**
```
0.754  UK GDPR Article 22                    ← correct top hit
0.698  EU AI Act Article 86
0.674  EU AI Act Article 22 (para-3)         ← AI Act Art 22 (authorised reps), unrelated
0.673  UK GDPR Article 35
0.672  EU AI Act Article 7 (para-2)
```

**Top-5 deployer retrieval:**
```
0.658  Novara AI Policy v3.1 §4.3 Data Subject Rights in AI Contexts  ← correct top hit
0.658  Novara TalentLens DPIA — 3.2 Lawful basis
0.656  Novara TalentLens DPIA — 6.2 External consultation
0.654  Novara TalentLens DPIA — 4. Risks Identified
0.638  Novara TalentLens DPIA — 7. Sign-off
```

**LLM output (verbatim):**
```
### What the law requires

UK GDPR Article 22
Art. 22 GDPR – Automated individual decision-making, including profiling - General Data Protection Regulation (GDPR)

The data subject shall have the right not to be subject to a decision based solely on
automated processing, including profiling, which produces legal effects concerning him
or her or similarly significantly affects him or her. Paragraph 1 shall not apply if
the decision: is necessary for entering into, performing a contract, or fulfilling a
pre-contractual obligation in respect of the data subject.

### What the policy says

Novara AI Policy v3.1 §4.3 Data Subject Rights in AI Contexts
Novara AI's standard data subject rights procedures (NAI-POL-0009) apply to all
AI-processed personal data. In addition, the following AI-specific obligations apply:
Right to Explanation Where an AI System makes or significantly influences a decision
affecting a data subject, the subject may request a plain-language explanation of the
factors that influenced the output. Product teams must implement this obligation.

### Gap

The policy does not address performing a fundamental rights impact assessment required
by UK GDPR Article 22.
```

**Observations:**
- Section 1 quotes UK GDPR Article 22(1) and (2) accurately.
- Section 2 quotes Novara §4.3 with Right to Explanation — relevant but only one of the four sub-clauses Q3 asked about (consent, intervention, contest, point of view).
- **FRIA leak again in Section 3.** "The policy does not address performing a fundamental rights impact assessment required by UK GDPR Article 22." This is a fabricated claim — UK GDPR Article 22 does not require a fundamental rights impact assessment. The model conflated FRIA (AI Act Art 27) with the rights-impact framing in the system prompt.
- The expected partial finding (HITL exists; right-to-contest absent; right-to-express-view absent) was not surfaced. The model produced one wrong claim instead of decomposing the four sub-clauses.

**Runtime:** 34.9s.

---

### Q4 — transparency (ambiguous framing)

**Query:**
> Are we doing enough on transparency for candidates assessed by TalentLens?

**Top-5 regulation retrieval:**
```
0.574  EU AI Act Article 50 (para-4)        ← deep-fakes provision (wrong subtopic)
0.568  EU AI Act Article 13 (para-1)        ← provider→deployer transparency
0.560  EU AI Act Article 112 (para-11)
0.549  EU AI Act Article 50 (para-1)
0.547  EU AI Act Article 10 (para-5)
```
**Top hit is Article 50 para-4 (about deep-fakes disclosure) — not the relevant transparency obligation for CV screening.**

**Top-5 deployer retrieval:**
```
0.752  Novara TalentLens DPIA — 2.1 What TalentLens does
0.739  Novara TalentLens DPIA — 3.1 Why processing is necessary
0.735  Novara TalentLens Model Card — 2.1 Primary use case
0.729  Novara TalentLens Transparency Notice — Changes to this Notice
0.726  Novara TalentLens Model Card — 2.2 Out-of-scope uses
```

**LLM output (verbatim):**
```
### What the law requires

4. Deployers of an AI system that generates or manipulates image, audio or video content
constituting a deep fake, shall disclose that the content has been artificially generated
or manipulated. This obligation shall not apply where the use is authorised by law to
detect, prevent, investigate or prosecute criminal offence...

### What the policy says

TalentLens is an AI-powered CV-screening and candidate-ranking service. Enterprise HR
teams (Customers) submit candidate CVs and a job description; the system returns
structured assessments of skill alignment, identified strengths, identified gaps, and
a relative ranking when multiple candidates are submitted. Outputs are delivered to
the Customer's recruiter for human review.

### Gap

The law requires deploying entities to disclose that AI-generated content is artificial,
while the policy states that TalentLens does not disclose this information. This
discrepancy highlights a gap in transparency regarding the nature of the AI-generated
content used by TalentLens.
```

**Observations:**
- **Significant topic drift.** Section 1 quotes the deep-fakes disclosure obligation from Article 50(4). TalentLens is a CV-screening tool — it doesn't generate "image, audio or video content constituting a deep fake." The quoted obligation isn't applicable.
- Section 2 describes what TalentLens does but doesn't engage with transparency-to-candidates specifically.
- Section 3 applies the deep-fake disclosure obligation to TalentLens incorrectly. Wrong gap finding.
- The expected partial finding (transparency notice exists; depth varies; Article 13/14 GDPR coverage incomplete) was not surfaced.
- Q4's ambiguous framing ("transparency enough?") allowed retrieval to grab the wrong "transparency" obligation. **Query expansion or query disambiguation would help here** (Category 1 mitigation in `evaluation-findings.md`).

**Runtime:** 28.2s.

---

### Q5 — FRIA (single-facet, silence expected — the canary)

**Query:**
> Have we performed a Fundamental Rights Impact Assessment under EU AI Act Article 27 for TalentLens as a deployer of an Annex III high-risk system?

**Top-5 regulation retrieval:**
```
0.787  EU AI Act Article 27 (para-1)        ← correct top hit
0.741  EU AI Act Article 26 (para-9)
0.731  EU AI Act Article 27 (para-2)        ← also retrieved
0.727  EU AI Act Article 86
0.719  EU AI Act Article 7 (para-3)
```

**Top-5 deployer retrieval:**
```
0.759  Novara TalentLens DPIA — Preamble
0.737  Novara TalentLens Model Intake Assessment — 7. Risk Classification
0.732  Novara TalentLens DPIA — 5.1 Mitigations applied per risk
0.721  Novara TalentLens DPIA — 6.2 External consultation
0.706  Novara TalentLens DPIA — 4. Risks Identified
```

**LLM output (verbatim):**
```
### What the law requires

[regulation/eu-ai-act-2024-1689#article-27-para-1]

### What the policy says

Novara TalentLens DPIA — Preamble states that the DPIA was conducted for TalentLens as
a deployer of an Annex III high-risk system.

### Gap

The law requires a Fundamental Rights Impact Assessment (Article 27 para-1) before
deploying a high-risk AI system, while the policy only mentions a Data Protection Impact
Assessment (DPIA) for TalentLens. The DPIA is more comprehensive and covers fundamental
rights, whereas the policy focuses on data protection aspects. Therefore, there is a gap
in the policy regarding the fundamental rights impact assessment required by the EU AI Act.
```

**Observations:**
- **Gap finding is correct.** Section 3 correctly identifies the FRIA gap.
- Section 1 is sparse — only the citation, no obligation sentence. Likely the model interpreted "begin with the chunk_id in square brackets" too literally.
- Section 2 cites the DPIA Preamble — substantive but inverts the actual situation: the DPIA's existence is true, but the implication that it covers FRIA is wrong.
- Section 3 contains a factual error mid-paragraph: *"The DPIA is more comprehensive and covers fundamental rights"* — this is incorrect (DPIA = data protection; FRIA = fundamental rights, distinct legal framework). But the sentence preceding and following arrive at the correct gap conclusion. **Same DPIA/FRIA conflation we documented in V2 prompt iteration (`evaluation-findings.md` Stage 4).**
- Functionally correct on the headline (gap), imperfect on supporting reasoning.

**Runtime:** 27.9s.

---

## Cross-query summary

| Query | Retrieval REG (top hit on right topic?) | Retrieval DEP | Section 1 | Section 2 | Section 3 (Gap) | Overall |
|---|---|---|---|---|---|---|
| Q1 multi-facet | Mixed (Art 14 ✓; Art 13 missing; AI Act Art 22 ≠ GDPR Art 22) | Adequate | 1 of 4 sub-questions correct | Truncated mid-output | **Did not run (output truncated)** | Truncated, mostly off-topic |
| Q2 red-teaming | Wrong-article (Art 26 dominates; Art 9 at #5) | Strong (§3.4 at 0.810) | Wrong topic (Art 26 instructions) | Correct (red-teaming policy) | **FRIA leak: wrong gap** | Internally inconsistent |
| Q3 Article 22 sub-clauses | Correct (UK GDPR 22 top hit) | Adequate (§4.3 top hit) | Correct quote | Partial (1 of 4 sub-clauses) | **FRIA leak: fabricated claim** | Sections 1-2 OK, Section 3 wrong |
| Q4 transparency (ambiguous) | Off-topic (Art 50(4) deep-fakes top) | Adequate (DPIA, Model Card) | Wrong topic (deep fakes) | Adequate description | **Wrong gap (deep fakes for CV screening)** | Significantly wrong |
| Q5 FRIA (silence) | **Correct (Art 27 paras top)** | Adequate (DPIA Preamble) | Sparse (citation only) | Substantive | **Correct gap with mid-paragraph factual error** | Functionally correct |

**Headline finding accuracy: 1 of 5 correct (Q5).** Q2 and Q3 produce internally inconsistent or factually wrong gap statements due to a system-prompt-induced "FRIA leak." Q1 truncates before reaching the Gap section. Q4 retrieves the wrong transparency obligation entirely.

---

## Patterns observed at this baseline

These are the recurring failure modes specific to Qwen 1.5B + V3 prompt at this scale:

**Pattern 1 — "FRIA leak"** (Q2, Q3): the system prompt's example sentence (*"if the law says 'fundamental rights impact assessment,' write that exact phrase"*) leaks into output as content even when the query is unrelated to FRIA. Small model treats the example as a thing to mention rather than as a rule about formatting. This is a prompt-design fragility specific to small-model scale.

**Pattern 2 — Multi-facet truncation** (Q1): the 550-token output budget is insufficient to cover 4 provisions at one section each plus policy sections plus gap. Output cut off before Gap section.

**Pattern 3 — Wrong-article anchoring** (Q2 specifically): when retrieval surfaces multiple articles and the wrong-topic article ranks higher (Article 26 above Article 9 for red-teaming), the LLM anchors on the higher-ranked article in Section 1.

**Pattern 4 — Concept conflation** (Q3 GDPR Art 22 ↔ FRIA; Q5 DPIA ↔ FRIA): small model conflates legally-distinct but semantically-adjacent assessments / rights frameworks. Documented in `evaluation-findings.md` Category 3.

**Pattern 5 — Topic drift on ambiguous queries** (Q4): a vague query ("transparency enough?") matches whichever transparency provision has highest cosine similarity, regardless of applicability. CV screening doesn't deal with deep fakes; the system can't distinguish.

**Pattern 6 — Section 1 substance variance** (Q5 sparse citation; Q2/Q4 quoting wrong-topic obligations; Q3 quoting correct text): the prompt's Section 1 instruction produces inconsistent outcomes. Small model handles "begin with chunk_id; then state obligation" inconsistently across queries.

---

## Baseline characteristics for future comparison

When a different configuration is tested (bigger model, GPU, different prompt), compare against this baseline along these axes:

1. **Retrieval-side metrics** (model-agnostic for given embedding):
   - Did top-5 REG surface all law articles named in the query?
   - Did the deployer-side retrieval surface document(s) actually relevant?

2. **LLM output substance**:
   - Did the gap-finding section identify the correct gap on Q5? (Currently: yes, with imperfections)
   - Did the FRIA leak appear on Q2/Q3? (Currently: yes)
   - Did Q1 complete without truncation? (Currently: no)
   - Did Q4 retrieve the right transparency obligation? (Currently: no — deep fakes instead)

3. **LLM output format**:
   - Three Markdown sections present?
   - Citations present in [square brackets]?
   - No prompt instruction text leaked into output?

4. **Operational metrics**:
   - Per-query runtime (cold)
   - Output truncation events

5. **Demoability**:
   - Can the output be shown to a non-technical audience without narration?
   - Where does the marker need explanation?

---

## What this baseline is NOT

- Not a gold-set evaluation against `intentional-gaps.md`. Expected outcomes are documented there; this baseline is a snapshot of system *behaviour*, not a programmatic *score*.
- Not a measurement of upper-bound capability. Qwen 1.5B is the lower end of capable local models.
- Not a final state. Intended specifically as the floor against which future model/prompt improvements are measured.

## Logged for: future comparison points

When we run on Colab with a bigger model (e.g., Qwen 2.5-7B-Instruct or Gemma 2-9B), we expect (predictions, not measurements):

- Pattern 1 (FRIA leak) likely reduced — bigger models follow prompt-structure rules more reliably
- Pattern 2 (truncation) addressable by raising max_new_tokens; cost is more compute
- Pattern 3 (wrong-article anchoring) addressed by query expansion or per-sub-question retrieval, not model size — predicted similar
- Pattern 4 (concept conflation) likely reduced — bigger models distinguish related concepts better
- Pattern 5 (topic drift on ambiguous) addressed by query disambiguation, not model size — predicted similar
- Pattern 6 (Section 1 variance) likely reduced — better prompt-following

If Patterns 1, 4, 6 reduce significantly with a bigger model, that's evidence the current baseline failures are model-capability-bound. If they persist, that's evidence the failures are deeper architectural / prompt issues.
