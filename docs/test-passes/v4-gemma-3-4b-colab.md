# Test pass — Gemma 3-4B-it on Colab GPU (two prompt configurations)

> Cross-family comparison point against the Qwen 1.5B / 3B / 7B sequence
> documented in `v3-qwen-1.5b-local-baseline.md`, `v4-qwen-1.5b-prompt-hygiene.md`,
> `v4-qwen-3b-colab.md`, and `v4-qwen-7b-colab.md`. Tests two distinct
> hypotheses on Gemma 3-4B: (a) does the FRIA-leak suppression observed
> at 3B+ in Qwen extend across families, and (b) does prompt-format
> adaptation (per Google's published Gemma 3 guidance) close the
> remaining substance gap on Q5.
>
> Run dates: 2026-05-06 (UTC). Documented 2026-05-07.

---

## Hypotheses going in

**H1: Cross-family FRIA-leak suppression.** From `v4-qwen-3b-colab.md`:
the FRIA-leak threshold sits between 1.5B and 3B for Qwen. We don't
know whether this is parameter-count-driven (any 3B+ instruction-tuned
model is clean) or family-specific (Qwen's training corpus or
instruction-tuning suppresses the leak; other families might not).
Gemma 3-4B is the natural test: similar parameter range to Qwen 3B,
different model family.

**H2: Prompt-format adaptation closes the residual quality gap.** Our V4
SYSTEM_PROMPT is 690 chars / 5 numbered rules + role description.
Google's official Gemma 3 launch documentation
(https://huggingface.co/blog/gemma3) explicitly recommends *"very short
system prompts followed by user prompts"* — their example system prompt
is a single sentence (~70 chars). If Gemma 3-4B was under-weighting our
long V4 system prompt and we re-package the rules into the user turn
(per Google's recommendation), the residual quality issues observed on
the V4-long-system Gemma run should improve.

---

## Configuration captured

Two runs on the same model, same retrieval, same queries, only the
prompt structure varied:

### Run A — V4 long-system prompt (same as Qwen runs)

| Component | Value |
|---|---|
| LLM | `google/gemma-3-4b-it` |
| LLM dtype | bfloat16 (per Google's recommendation) |
| Loader | `Gemma3ForConditionalGeneration` + `AutoProcessor` (multimodal class, text-only inference) |
| System prompt | V4 long: 690 chars, 5 rules + role |
| User message | `_user_message`: question + chunks + output template |
| Hardware | Colab Tesla T4 (16 GB GPU) |
| Embedding model | `BAAI/bge-large-en-v1.5` |
| ICO operational corpus | Excluded at query time |

### Run B — Gemma-adapted prompt (short system + rules-in-user)

| Component | Value |
|---|---|
| LLM | `google/gemma-3-4b-it` *(unchanged)* |
| All other config | *(unchanged from Run A)* |
| System prompt | `GEMMA3_SYSTEM_PROMPT`: 68 chars, role only |
| User message | `_user_message_gemma3`: 5 rules prepended, then question + chunks + template |

The two runs produce different cache keys (because system prompts differ), so they live as separate cache entries; no contamination between runs.

---

## Result — both runs across all 5 queries

| Query | Run A (V4 long-system) | Run B (Gemma-adapted) | Δ A→B |
|---|---|---|---|
| Q1 multi-facet | Single-obligation (Art 26 para-9 about DPIA) | Multi-section engagement: Art 26 para-9 + Art 14 + bias auditing; Section 3 picks Art 14 (two-person verification) as the gap | **Substantially better** — first model output across the project to engage multiple obligations rather than collapse to one |
| Q2 red-teaming | Cited Art 26 para-8 (testing requirement); Section 3 framing identified Gate G2 vs prior-deployment as the interpretive gap | Cited Art 26 para-8 (registration obligations — wrong topic); Section 3 says policy doesn't address pre-deployment red-teaming despite Section 2 quoting §3.4 mandating exactly that | **Worse** — Section 1 article less relevant; Section 2/3 internally contradictory |
| Q3 Article 22 sub-clauses | Cited Art 86; Section 3 framed as "policy lacks specific details" (vague) | Cited Art 86; Section 3 **explicitly identifies missing sub-clauses**: human intervention, contest, point of view | **Substantially better** — sub-clause analysis now comparable to Qwen 7B |
| Q4 transparency | Wrong-audience: Art 13 para-1 (transparency-to-deployers, not candidates) | Off-topic: Art 50 para-4 (deep-fakes — TalentLens generates no deep-fakes) | **Worse** — different wrong anchor |
| Q5 FRIA target | Accepted Novara's "Standard AI Feature" classification; soft gap framing | Still accepts self-classification; slightly stronger gap framing ("despite the legal requirement") | **Marginally better, core failure persists** |

**Net:** mixed. Two queries (Q1, Q3) substantially improved on the prompt change; two queries (Q2, Q4) regressed; Q5 marginally improved but the core conceptual failure persists.

---

## Verbatim outputs (per-query, both runs side-by-side)

### Q1 — multi-facet

**Latency:** Run A 193.5s · Run B 391.4s.

**Run A (V4 long-system):**
```
### What the law requires
[regulation/eu-ai-act-2024-1689#article-26-para-9] requires deployers of
high-risk AI systems to use the information provided under Article 13 to
comply with their obligation to carry out a data protection impact
assessment.

### What the policy says
The policy states that customer organizations are responsible for
conducting customer-side data protection compliance assessments and
honoring data subject rights requests, but does not detail how they will
fulfill the requirement to conduct a data protection impact assessment
under Article 35 of Regulation (EU) 2016/679 or Article 27 of Directive
(EU) 2016/680 as mandated by [regulation/eu-ai-act-2024-1689#article-26-para-9].

### Gap
The policy lacks specific details on how to implement the data protection
impact assessment required by Article 26 of the EU AI Act, particularly
concerning the steps needed to ensure compliance with Article 35 or
Article 27.
```

**Run B (Gemma-adapted):**
```
### What the law requires
[regulation/eu-ai-act-2024-1689#article-26-para-9] states that deployers
of high-risk AI systems shall use the information provided under Article 13
to comply with their data protection impact assessment obligations.

### What the policy says
The policy outlines customer responsibilities including providing
TalentLens output to candidates after recruiter review and conducting
customer-side data protection compliance assessments
[deployer-extras/novara-talentlens-model-card#7-2-customer-responsibilities],
but does not explicitly detail the process for verifying and confirming
identification as required by Article 14
[regulation/eu-ai-act-2024-1689#article-14-para-5]. It also lacks specific
details regarding the implementation of bias auditing methodologies or
demographic parity requirements
[deployer-extras/novara-talentlens-dpia#5-1-mitigations-applied-per-risk].

### Gap
The policy fails to address the requirement to separately verify and
confirm identification as mandated by Article 14 of the EU AI Act,
specifically the need for at least two natural persons to perform this
verification [regulation/eu-ai-act-2024-1689#article-14-para-5].
```

**Observation:** Run B engages with multiple obligations across Sections 2 and 3 (Art 26-9, Art 14, bias auditing) — an approximation of multi-facet handling that no other model+prompt combination has produced in this project. The single-call architecture's structural single-obligation limit is *softened* by Gemma's natural verbosity + the rules being placed in the user turn (closer to the actual content being analysed). Run A defaulted to one obligation.

---

### Q2 — red-teaming

**Latency:** Run A 24.9s · Run B 27.9s.

**Run A:**
```
### What the law requires
[regulation/eu-ai-act-2024-1689#article-26-para-8] EU AI Act requires
deployers of high-risk AI systems to perform testing "throughout the
development process, and, in any event, prior to their being placed on
the market or put into service."

### What the policy says
Novara AI's policy [deployer/novara-ai-policy-v3.1#section-3-4] mandates
a red-teaming exercise before Gate G2, including prompt injection and
harm generation testing, and specifies that high-risk features require
external red-teamers.

### Gap
The policy does not explicitly state that red-teaming must occur *prior*
to deployment, only before a specific gate, potentially creating a gap
in compliance with the EU AI Act's requirement for testing "prior to
their being placed on the market or put into service"
[regulation/eu-ai-act-2024-1689#article-26-para-8].
```

**Run B:**
```
### What the law requires
[regulation/eu-ai-act-2024-1689#article-26-para-8] EU AI Act Article 26
(para-8) states that "Deployers of high-risk AI systems that are public
authorities, or Union institutions, bodies, offices or agencies shall
comply with the registration obligations referred to in Article 49."

### What the policy says
The policy explicitly mandates a structured red-team exercise before
Gate G2 for all AI systems, including High-Risk AI Features, and
specifies that at least one red-teamer must be external to Novara AI
for High-Risk AI Features [deployer/novara-ai-policy-v3.1#section-3-4].

### Gap
The policy does not address the requirement for pre-deployment
red-teaming of high-risk AI systems as mandated by the EU AI Act
[regulation/eu-ai-act-2024-1689#article-26-para-7], specifically the
requirement to perform testing "prior to their being placed on the
market or put into service" [regulation/eu-ai-act-2024-1689#article-9-para-8].
```

**Observation:** Run A picked the right paragraph of Art 26 (testing language); Run B picked the actual Art 26 para-8 (registration — different topic) and only references the testing language indirectly in Section 3. Run B's Section 2/3 are also internally contradictory: Section 2 quotes §3.4 mandating pre-deployment red-teaming, Section 3 says the policy doesn't address it. Run A was more internally coherent.

---

### Q3 — GDPR Article 22 sub-clauses

**Latency:** Run A 21.6s · Run B 26.0s.

**Run A:**
```
### What the law requires
[regulation/eu-ai-act-2024-1689#article-86] EU AI Act Article 86 requires
affected persons subject to a decision taken by a deployer on the basis
of a high-risk AI system to have the right to explanation.

### What the policy says
The policy states that where an AI system makes or significantly influences
a decision affecting a data subject, the subject may request a
plain-language explanation of the factors that influenced the output
[deployer/novara-ai-policy-v3.1#section-4-3]. However, the policy does
not explicitly detail how this right to explanation will be implemented
or the process for providing such explanations.

### Gap
The policy lacks specific details regarding the implementation of the
right to explanation mandated by EU AI Act Article 86, particularly
concerning the practical steps required by product teams to fulfill this
obligation.
```

**Run B:**
```
### What the law requires
[regulation/eu-ai-act-2024-1689#article-86] EU AI Act Article 86 states
that "any affected person subject to a decision which is taken by the
deployer on the basis of the output from a high-risk AI system listed in
Annex III, with the exception of systems listed under point 2 thereof,
and which produces legal effects or similarly significantly affects that
person in a way that they consider…"

### What the policy says
Novara AI's policy [deployer/novara-ai-policy-v3.1#section-4-3] outlines
a "Right to Explanation" where a data subject may request a plain-language
explanation of the factors influencing an AI system's output.

### Gap
The policy [deployer/novara-ai-policy-v3.1#section-4-3] addresses the
"Right to Explanation," but the policy does not explicitly detail how to
implement this right, nor does it cover the other required elements of
GDPR Article 22, such as the right to obtain human intervention, the
right to contest the decision, or the right to express their point of
view.
```

**Observation:** Run B explicitly enumerates the three missing sub-clauses (human intervention, contest, point of view) — comparable to Qwen 7B's level of substance on this query. Run A produced a vague "lacks specific details" framing without naming the missing rights. Both runs anchored on AI Act Art 86 rather than GDPR Art 22 directly (a partial topic switch, defensible since both deal with the right-to-explanation domain).

---

### Q4 — transparency

**Latency:** Run A 24.6s · Run B 24.9s.

**Run A:**
```
### What the law requires
[regulation/eu-ai-act-2024-1689#article-13-para-1] High-risk AI systems
shall be designed and developed in such a way as to ensure that their
operation is sufficiently transparent to enable deployers to interpret
a system's output and use it appropriately.

### What the policy says
The policy states that TalentLens is designed to assist HR teams in
screening and ranking candidates, but it does not explicitly detail how
the system's outputs will be sufficiently transparent for interpretability
and appropriate use, nor does it mention any process for ensuring this
transparency.

### Gap
There is a gap between the law's requirement for sufficient transparency
regarding AI system outputs and the policy, which lacks specific details
on how TalentLens will achieve this transparency, particularly concerning
how recruiters can interpret the structured assessments provided.
```

**Run B:**
```
### What the law requires
[regulation/eu-ai-act-2024-1689#article-50-para-4] EU AI Act Article 50
(para-4) states that deployers of deepfake content must disclose that the
content has been artificially generated or manipulated.

### What the policy says
The Novara TalentLens policy does not explicitly require disclosure that
the candidate assessments are artificially generated or manipulated,
although it does state that the system delivers outputs for "human review"
[deployer-extras/novara-talentlens-dpia#2-1-what-talentlens-does]. The
Transparency Notice only details changes to the notice itself
[deployer-extras/novara-talentlens-transparency-notice#changes-to-this-notice].

### Gap
There is a gap between the EU AI Act's requirement for disclosure of
manipulated content and the Novara TalentLens policy, which lacks a
specific provision mandating disclosure regarding the AI-generated nature
of the candidate assessments.
```

**Observation:** Run A had the *wrong-audience* failure (Art 13 transparency-to-deployers cited for a query about transparency-to-candidates). Run B has a *different* wrong anchor — Art 50 para-4 (deep-fakes), which doesn't apply to CV screening at all. Both are retrieval-driven failures but the new prompt structure surfaced a less-relevant chunk to anchor on. Q4 illustrates that prompt-format choice can shift which retrieved chunk the model attends to first, with no guarantee the new attention is more accurate.

---

### Q5 — FRIA target

**Latency:** Run A 28.4s · Run B 28.8s.

**Run A:**
```
### What the law requires
[regulation/eu-ai-act-2024-1689#article-27-para-1] EU AI Act Article 27
(para-1) requires deployers of high-risk AI systems, specifically those
governed by public law or providing public services, and deployers of
high-risk AI systems intended to be used in the area listed in point 2
of Annex III, to conduct a fundamental rights impact assessment prior to
deployment.

### What the policy says
The policy states that TalentLens is classified as a "Standard AI Feature"
and does not require a fundamental rights impact assessment
[deployer-extras/novara-talentlens-model-intake-assessment#7-risk-classification].
The DPIA documentation confirms that no consultation has occurred with
data subjects or the supervisory authority
[deployer-extras/novara-talentlens-dpia#6-2-external-consultation].

### Gap
The policy does not address whether TalentLens, classified as a high-risk
AI system due to its function as a decision-support system, meets the
criteria for triggering the requirement to perform a fundamental rights
impact assessment under [regulation/eu-ai-act-2024-1689#article-27-para-1].
```

**Run B:**
```
### What the law requires
[regulation/eu-ai-act-2024-1689#article-27-para-1] EU AI Act Article 27
(para-1) requires deployers of high-risk AI systems, specifically those
governed by public law or providing public services, and deployers of
high-risk AI systems intended to be used in the area listed in point 2
of Annex III, to perform a fundamental rights impact assessment prior to
deployment.

### What the policy says
The policy states that TalentLens is classified as a "Standard AI Feature"
and does not require a fundamental rights impact assessment under Article
27 of the EU AI Act
[deployer-extras/novara-talentlens-model-intake-assessment#7-risk-classification].
The DPIA [deployer-extras/novara-talentlens-dpia#4-risks-identified]
identifies potential risks related to bias and data breaches, but does
not explicitly state whether a fundamental rights impact assessment was
conducted.

### Gap
The policy does not confirm whether a fundamental rights impact assessment
was performed for TalentLens as a deployer of an Annex III high-risk
system, despite the legal requirement outlined in
[regulation/eu-ai-act-2024-1689#article-27-para-1].
```

**Observation:** This is the diagnostic. Both runs exhibit the same core failure — accepting Novara's "Standard AI Feature" classification at face value, despite Annex III §4 mandating high-risk classification for employment-purpose CV-screening AI regardless of internal labelling. Run B's Section 3 is *marginally* sharper ("despite the legal requirement"), but neither produces the FRIA-vs-DPIA conceptual distinction that Qwen 3B and 7B handled cleanly with the same V4 prompt. **Prompt-format adaptation does not close the substantive Q5 gap. The failure mode is model-capability-bound, not prompt-bound.**

---

## What both runs together tell us

### H1 (cross-family FRIA-leak suppression): supported by both runs
Both Run A and Run B show no FRIA-leak insertion across Q2/Q3/Q4. Combined with the Qwen 3B / 7B clean runs, this gives **three non-leaking samples in the 3–4B range across two model families**, against one leaking sample at Qwen 1.5B. The training-data-prior leak that drives the FRIA insertion in 1.5B-class models is suppressed at 3B+ regardless of family. **Stronger empirical claim than Qwen-only data could support.**

### H2 (prompt-format adaptation closes the substance gap): partially supported, partially refuted
- **Improved on multi-step engagement.** Q1's multi-section reasoning and Q3's sub-clause enumeration are clearly better in Run B. Both required following multi-step instructions; Gemma's instruction-tuning preference for rules-in-user-turn produced more compliance with those instructions.
- **Did not improve and sometimes worsened retrieval anchoring.** Q2 and Q4 picked less-relevant chunks in Run B. The mechanism isn't obvious — possibly the longer user message changes which retrieved chunks the model attends to first.
- **Did not close the Q5 substance gap.** The conceptual failure (accepting deployer self-classification) survives both prompt designs. This isolates *prompt-format* (a design lever) from *model-capability* (not a design lever) — and demonstrates that for some failure modes, only the latter matters.

### Net empirical claim — three separable axes
Stage 8 originally separated *prompt-design* from *model-scale* as orthogonal factors. The Gemma runs add a third orthogonal axis: *prompt-format-per-family*. Cross-family deployment requires per-family prompt design (Run A ≠ Run B), but per-family prompt design has limits (Run B still fails on Q5).

**Three orthogonal failure-mode classes mapped to three orthogonal mitigation levers:**

| Failure mode | Lever |
|---|---|
| FRIA leak (training-data prior at 1.5B) | Model scale (≥ 3B suppresses across families) |
| Multi-step instruction following (Q3 sub-clauses) | Prompt format (rules-in-user works for Gemma; rules-in-system works for Qwen) |
| Conceptual reasoning (Q5 FRIA-vs-DPIA distinction; deployer self-classification override) | Model family + capability (Qwen 3B/7B handle; Gemma 4B does not, despite being larger) |
| Retrieval-side failures (Q4 wrong-audience / wrong-topic) | Upstream: query expansion / disambiguation. Not addressable by either prompt or scale. |
| Architectural single-obligation (Q1) | Upstream: light decomposition before `analyse()`. Architectural, not prompt or model. |

This is a richer error-category model than the project entered with — five orthogonal axes, each with a different mitigation path. Strong material for the report's Critical Analysis dimension.

---

## Status after this pass

- **Empirical comparison points now total five:** Qwen 1.5B (V3), Qwen 1.5B (V4), Qwen 3B (V4), Qwen 7B (V4), Gemma 3-4B (V4 long-system + Gemma-adapted). Cross-family + cross-prompt-format coverage.
- **Production demo path unchanged:** local Qwen 1.5B with V4 long-system prompt remains the demo default. Q5 demo-quality output achievable at Qwen 3B (Colab GPU) per `v4-qwen-3b-colab.md`.
- **For the report's "scaling and automation considerations":** documented as a `decisions.md` entry — per-family prompt design as a real production deployment cost.

## What this run does NOT establish

- That Gemma 4B is "worse" than Qwen 3B. Gemma 4B is *better* on multi-step instruction-following (Q1 multi-facet engagement, Q3 sub-clause enumeration); Qwen 3B/7B are better on the specific conceptual reasoning task at Q5. Different models have different strengths at similar scale.
- That the Gemma-adapted prompt is the universally better choice. It improved Q1 and Q3 but regressed Q2 and Q4. A production deployment supporting both prompt formats (via dispatch on model id, as our `_get_prompts` does) would let users pick per-query, but the project did not test which format is *typically* better — only that they differ.
- That Gemma 3-12B or 27B (which we did not run) would close the Q5 gap. The substance failure could be a 4B-scale issue within Gemma's family, or a Gemma-specific instruction-tuning issue at any scale. Untested.
