# Test pass — V4 prompt hygiene attempt (Qwen 1.5B local)

> Iteration on V3 (`v3-qwen-1.5b-local-baseline.md`) testing whether removing
> topic-specific examples from the system prompt eliminates the "FRIA leak"
> failure mode observed on Q2 / Q3 / Q4. **Result: it did not.**
>
> Run date: 2026-05-05.

---

## Hypothesis going in

The V3 baseline showed an unexpected failure mode: the phrase *"fundamental rights impact assessment"* appeared in the Gap section of Q2 (red-teaming), Q3 (GDPR Article 22), and Q4 (transparency) — none of which are about FRIA. We called this the "FRIA leak."

The hypothesis was that the leak was caused by topic-specific examples in the V3 SYSTEM_PROMPT that mentioned FRIA explicitly:

```
Rule 2: "...If the law says 'fundamental rights impact assessment,' write
        that exact phrase."
Rule 4: "...For example, a fundamental rights impact assessment (Art 27 of
        the EU AI Act) is NOT the same as a data protection impact
        assessment (Art 35 of GDPR)..."
```

Best-practice prompt hygiene for small models (Qwen / Gemma at <3B parameter scale) recommends:
- No topic-specific examples in instructions (small models treat them as content)
- Strictly procedural rules
- Examples — if needed — go in chat-template `user`/`assistant` turns, not embedded in system prompts

References: web.dev "Practical prompt engineering for smaller LLMs"; OWASP LLM07 system prompt leakage.

**Predicted outcome:** removing the FRIA mentions from system prompt would eliminate the leak in Q2/Q3/Q4 outputs. Q1 truncation, Q4 retrieval drift, and Q5 sparse Section 1 unaffected (different failure modes).

## What changed (V3 → V4)

The V4 SYSTEM_PROMPT is V3 with the topic-specific examples replaced by abstract instructions:

```
RULES:
1. Cite chunk IDs verbatim in [square brackets].
2. Quote key legal phrases from the law passages verbatim. Do not paraphrase.
3. Do not invent obligations or policy provisions not present in the
   passages provided.
4. Stay on the specific topic in the question. Different legal frameworks
   have different obligations even when topics seem similar; do not
   conflate them.
5. If the policy passages do not address the obligation in the question,
   say so directly. Do not stretch unrelated content to fit.
```

No mention of FRIA, DPIA, Article 27, or Article 35 anywhere in the prompt.

User-message template, retrieval configuration, model, and decoding parameters all unchanged from V3.

## Result — V3 vs V4 across all 5 queries

| Query | V3 outcome | V4 outcome | Δ |
|---|---|---|---|
| Q1 multi-facet | 1 of 4 sub-sections correct (Article 14); output truncated before Gap section | Same — 1 of 4 correct, output truncated | **No change** |
| Q2 red-teaming | Section 3: *"The policy does not address performing a fundamental rights impact assessment"* — FRIA leak | Section 3: *"The policy does not address performing a fundamental rights impact assessment"* — **FRIA leak persists, verbatim** | **No change** |
| Q3 Article 22 sub-clauses | Section 3: *"...fundamental rights impact assessment required by UK GDPR Article 22"* — FRIA leak (fabricated claim) | Section 3: *"...fundamental rights impact assessment required by UK GDPR Article 22"* — **FRIA leak persists, verbatim** | **No change** |
| Q4 transparency | Section 1: deep-fakes obligation (Art 50 para-4); Section 3: deep-fakes gap | Section 1: AI-system interaction transparency (Art 50 para-1, slightly more relevant); Section 3: contains *"a fundamental rights impact assessment"* — leak persists | Section 1 mildly improved, leak persists |
| Q5 FRIA (silence target) | Section 1 sparse; Section 2 brief; Section 3: factual error mid-paragraph but **correct gap conclusion** ("there is a gap in the policy regarding the fundamental rights impact assessment required by the EU AI Act") | Section 1 sparse; Section 2 substantively engages with policy; Section 3: clean reasoning but **wrong conclusion** — *"the obligation to conduct a fundamental rights impact assessment does not apply to TalentLens"* (accepts Novara's self-classification as Standard AI Feature; conflates internal classification with EU AI Act mandatory classification) | **Regression on the headline gap finding** |

## Verbatim outputs (per-query)

For reproducibility, the full V4 outputs and retrieval scores are in `/tmp/baseline_5q_v4.log`. Key excerpts:

### Q2 V4 (FRIA leak persists)
```
### Gap

[deployer/novara-ai-policy-v3.1#section-3-4] Novara AI Policy v3.1 §3.4 Red-Teaming and Safety Testing

The policy does not address performing a fundamental rights impact assessment.
```
*(Identical to V3 Section 3 wording.)*

### Q3 V4 (FRIA leak persists)
```
### Gap

The policy does not address performing a fundamental rights impact assessment
required by UK GDPR Article 22.
```
*(Identical to V3 Section 3 wording. UK GDPR Article 22 does not require a FRIA;
this is a fabricated claim — same factual error as V3.)*

### Q5 V4 (regression — wrong conclusion)
```
### Gap

Novara TalentLens has not performed a Fundamental Rights Impact Assessment (FRIA)
under EU AI Act Article 27 for TalentLens as a deployer of an Annex III high-risk
system. The TalentLens system is classified as a Standard AI Feature, which is
not considered a high-risk AI system under the EU AI Act. Therefore, the
obligation to conduct a fundamental rights impact assessment does not apply to
TalentLens.
```

The model accepts Novara's internal "Standard AI Feature" classification at
face value. Per EU AI Act Annex III §4, employment-purpose CV-screening AI is
mandatorily high-risk regardless of internal labeling — the C1 contradiction
documented in `intentional-gaps.md`. V4 fails to surface this because Section 3
takes the policy's self-classification as authoritative. Compare V3 Section 3,
which preserved the gap conclusion despite a mid-paragraph factual error.

## What this tells us — the actual cause of the FRIA leak

V4 disproved the hypothesis that the FRIA leak originated from system-prompt examples. With those examples removed, the leak persists verbatim in Q2 and Q3 Section 3 outputs. Three more likely causes:

1. **Training-data associations.** Qwen 1.5B was trained on EU AI Act discussions where *deployer + high-risk + obligation* co-occurs strongly with *fundamental rights impact assessment*. The model generates the phrase as a probabilistic association, not from instruction. Prompt-hygiene changes don't override model priors.

2. **Retrieved-chunk content.** Article 26 para-9 — appearing in Q2/Q3/Q5 retrieval — references *"Article 27 of Directive (EU) 2016/680"* (a different Article 27 about data processing audits). The model may pattern-match "Article 27" to FRIA from training-data associations.

3. **Cross-query priming via single shared model state.** Less likely since each query is a fresh chat-template invocation, but worth noting if it persists in further tests.

The dominant cause is almost certainly (1). At Qwen 1.5B scale, a prompt-hygiene change cannot fully suppress training-data biases.

## What this tells us about the simplified architecture's capability ceiling on Qwen 1.5B

V3 → V4 testing confirms the failure modes documented in `evaluation-findings.md` Category 3 ("Citation/text binding hallucination" — small-model concept conflation). The proposed mitigations there — larger generative model, hybrid LLM + rule-based verification, numbered-chunk indirection — remain the right paths.

What V4 *does not* support: the claim that "better prompt design" can fix the substance issues at this model scale. We tried; it didn't. Either we accept the ceiling (Path A: demo Q5, document limits) or move to a bigger model (Colab GPU, Qwen 7B / Gemma 9B).

## Recommendation

**Roll back SYSTEM_PROMPT to V3.** V4 didn't help on the failure modes it targeted (FRIA leak in Q2/Q3/Q4) and produced a regression on Q5 (the only query that previously worked) by switching from "messy but correct gap finding" to "clean but wrong conclusion."

**Skip the few-shot worked example (the "#2" iteration originally proposed).** Less confident now that few-shot will help, given training-data prior is the dominant cause. The token cost (~300 extra tokens per query) is real; the predicted improvement is uncertain.

**For the report:** V4 is a useful negative finding. *"We tested whether better prompt hygiene (removing topic-specific examples from the system prompt) would fix the FRIA leak observed on Q2/Q3/Q4. It did not. The leak survives prompt changes, indicating the cause is training-data association at small-model scale rather than prompt design. This strengthens our architectural claim that small-model capability is the dominant constraint, not prompt engineering, and supports the recommendation to test on a larger model (Colab GPU, 7B+ parameters) before further prompt iteration."*

That's evidence-based critical analysis. The rubric explicitly rewards "results discussed and interpreted, and insight demonstrated."

## Status after this pass

- Deployed `src/simplified.py` SYSTEM_PROMPT: V4 (currently)
- **Recommended action:** revert to V3 (better Q5 output)
- Next test pass to consider: same prompt (V3) on Colab GPU with Qwen 2.5-7B-Instruct or Gemma 2-9B
