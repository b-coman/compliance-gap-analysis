# Test pass — V4 prompt on Qwen 7B (Colab GPU)

> Same V4 prompt and BGE retrieval as the local-1.5B test passes; model
> swapped from Qwen 2.5-1.5B-Instruct to Qwen 2.5-7B-Instruct on a Colab
> Tesla T4. Tests the model-scale hypothesis from
> `v4-qwen-1.5b-prompt-hygiene.md`: does the FRIA leak persist at 7B+
> parameters, or is it a small-model training-data prior?
>
> Run date: 2026-05-06.

---

## Hypothesis going in

V4 (the V3 prompt with topic-specific examples removed) did not eliminate
the "FRIA leak" on Qwen 1.5B. The leak appeared verbatim on Q2, Q3, and
Q4 even though FRIA was nowhere in the prompt. The most likely cause
(documented in `v4-qwen-1.5b-prompt-hygiene.md`) was *training-data
association at small-model scale*: Qwen 1.5B reaches for "fundamental
rights impact assessment" as a high-probability completion in
deployer/high-risk contexts, regardless of instruction.

Predicted outcome at 7B+ scale:
- FRIA leak suppressed on Q2, Q3, Q4 (model-scale fix)
- Q5 substantially improved (less concept conflation)
- Q1 multi-facet still single-output (architectural limit, not scale)
- Q4 retrieval-driven failures still present (upstream of LLM)

---

## Configuration captured

| Component | Value |
|---|---|
| Architecture | `src/simplified.py` (single retrieve + single LLM call) |
| Embedding model | `BAAI/bge-large-en-v1.5` |
| Embedding device | CUDA |
| Retrieval top-k | 5 (regulation), 5 (DEP + DEP_EXTRAS) |
| Operational corpus (ICO) | Excluded at query time |
| LLM | `Qwen/Qwen2.5-7B-Instruct` |
| LLM device | CUDA + CPU offloading via `device_map="auto"` |
| LLM dtype | float16 |
| Generation: do_sample | False (deterministic) |
| Generation: repetition_penalty | 1.05 |
| Generation: max_new_tokens | 550 |
| Prompt | V4 (current default in `src/simplified.py`) |
| Hardware | Colab Tesla T4 (16 GB GPU, ~13 GB CPU RAM) |

**Latency anomaly worth recording:** T4's 16 GB GPU is *just* short of what
fp16 7B requires for full residence (model ~14 GB + activations). Accelerate's
`device_map="auto"` handled this gracefully — kernel survived — but offloaded
some layers to CPU, causing per-token GPU↔CPU transfer overhead. Result: Q1
took 696.8s (~11.6 min) instead of the expected 10–20s. Future Colab runs
should either use Qwen 3B (full GPU residence on T4) or 4-bit quantisation
via `bitsandbytes`. Latency is environment-bound, not model-bound; the
output substance is unaffected.

---

## V4-1.5B vs V4-7B across all 5 queries

| Query | V4 1.5B local | V4 7B Colab | Δ |
|---|---|---|---|
| Q1 multi-facet | 1 of 4 sub-sections, output truncated | 1 of 4, no truncation, single-section format | Architectural limit unchanged; truncation gone |
| Q2 red-teaming | FRIA leak (Section 3): *"policy does not address performing a fundamental rights impact assessment"* | **No FRIA leak.** Section 3 cites Article 9 para-8 testing requirement | **Leak suppressed** |
| Q3 Article 22 | FRIA leak (Section 3): *"fundamental rights impact assessment required by UK GDPR Article 22"* (fabricated) | **No FRIA leak.** Substantively correct: identifies missing sub-clauses (human intervention, contest, point of view) | **Leak suppressed; output substantively right** |
| Q4 transparency | FRIA leak persisted in Section 3 | **No FRIA leak.** Cites Article 13 para-1 (wrong-audience: transparency *to deployers*, not candidates) | Leak suppressed; new failure mode (retrieval-driven wrong-audience) |
| Q5 FRIA target | Wrong conclusion (accepted Novara's "Standard AI Feature" self-classification) | **Clean and correct.** Cites Article 27 para-1, distinguishes FRIA from DPIA explicitly | **Headline gap finding fixed** |

---

## Verbatim outputs (per-query)

### Q1 — multi-facet

**Latency:** 696.8s (CPU offloading, see anomaly note above).

**LLM output:**
```
### What the law requires
[regulation/eu-ai-act-2024-1689#article-26-para-9] Deployers of high-risk
AI systems shall use the information provided under Article 13 of this
Regulation to comply with their obligation to carry out a data protection
impact assessment under Article 35 of Regulation (EU) 2016/679 or
Article 27 of Directive (EU) 2016/680.

### What the policy says
[deployer-extras/novara-talentlens-dpia#preamble] The policy mentions
conducting a data protection compliance assessment, but it does not
explicitly state that this assessment is a data protection impact
assessment as required by the EU AI Act.

### Gap
[regulation/eu-ai-act-2024-1689#article-26-para-9] The policy does not
address performing a data protection impact assessment as required by
the EU AI Act.
```

**Observations:**
- Single-obligation output (the model picked Article 26 para-9 from the multi-facet question and built around it). The query asked about Articles 13, 14, 22 GDPR, and 26; only Article 26 surfaces.
- This is the **same multi-facet limitation as 1.5B**, but cleaner — no truncation, no off-topic claims about Articles the model didn't cover. The single-call architecture cannot decompose; that's a Phase-1 design trade-off, not a model-scale issue.
- Note the legal nuance: Article 26 para-9 references *Article 27 of Directive (EU) 2016/680* (a different Article 27 about data-processing audits, not the AI Act FRIA Article 27). The model quoted accurately but a careless reader could conflate the two — worth flagging in the report.

---

### Q2 — red-teaming

**Latency:** 192.2s.

**LLM output:**
```
### What the law requires
[regulation/eu-ai-act-2024-1689#article-9-para-8] The testing of high-risk
AI systems shall be performed, as appropriate, at any time throughout the
development process, and, in any event, prior to their being placed on
the market or put into service.

### What the policy says
[deployer/novara-ai-policy-v3.1#section-3-4] Every AI System must undergo
a structured red-team exercise before Gate G2. However, the policy does
not explicitly mention conducting tests or evaluations prior to deployment.

### Gap
[regulation/eu-ai-act-2024-1689#article-9-para-8] The policy does not
address performing testing of high-risk AI systems prior to their being
placed on the market or put into service.
```

**Observations:**
- **No FRIA leak.** Compare V4 1.5B Section 3, which said *"The policy does not address performing a fundamental rights impact assessment"* on this same query. The 7B model stays on the red-teaming topic.
- However, the gap finding is still wrong, in a *new* way. The policy explicitly mandates red-teaming "before Gate G2" (i.e., before deployment), which is exactly what Article 9 para-8 requires. The model reads §3.4 correctly in Section 2 but then writes "the policy does not explicitly mention conducting tests or evaluations prior to deployment" — directly contradicting §3.4's own language about Gate G2.
- This is a **reading/reasoning error**, not a training-data leak. It's a different failure category than V4-1.5B made.
- Section 1 retrieval improved over V3-1.5B baseline: Article 9 para-8 (correct) is cited rather than Article 26 (wrong topic).

---

### Q3 — GDPR Article 22 sub-clauses

**Latency:** 195.2s.

**LLM output:**
```
### What the law requires
[regulation/uk-gdpr-art-22#whole] The data subject shall have the right
not to be subject to a decision based solely on automated processing,
including profiling, which produces legal effects concerning him or her
or similarly significantly affects him or her.

### What the policy says
[deployer/novara-ai-policy-v3.1#section-4-3] Novara AI provides a right
to explanation where an AI System makes or significantly influences a
decision affecting a data subject. However, the policy does not explicitly
mention the right to obtain human intervention, the right to contest the
decision, or the right to express their point of view.

### Gap
[regulation/uk-gdpr-art-22#whole] The policy does not address the rights
to obtain human intervention, contest the decision, or express their point
of view, which are required by GDPR Article 22.
```

**Observations:**
- **No FRIA leak.** Compare V4 1.5B Section 3, which fabricated *"fundamental rights impact assessment required by UK GDPR Article 22"* (a meaningless claim — UK GDPR Article 22 has no FRIA requirement). The 7B model stays on Article 22's actual sub-clauses.
- **Substantively correct gap finding.** The query asked about four sub-clauses (consent, human intervention, contest, point of view); the model identifies the three that the policy genuinely lacks (intervention, contest, point of view) and notes the one that the policy partially addresses (right to explanation under §4.3).
- This is the **clearest 7B win on substance**. The 1.5B model could not produce this sub-clause analysis even with the V4 prompt.

---

### Q4 — transparency (ambiguous framing)

**Latency:** 234.4s.

**LLM output:**
```
### What the law requires
[regulation/eu-ai-act-2024-1689#article-13-para-1] High-risk AI systems
shall be designed and developed in such a way as to ensure that their
operation is sufficiently transparent to enable deployers to interpret
a system's output and use it appropriately.

### What the policy says
The policy does not address ensuring that the operation of the AI system
is sufficiently transparent to enable deployers to interpret its output
and use it appropriately.
[deployer-extras/novara-talentlens-model-card#2-1-primary-use-case] and
[deployer-extras/novara-talentlens-model-card#2-2-out-of-scope-uses]
focus on the primary use cases and out-of-scope uses but do not explicitly
mention transparency requirements.

### Gap
[regulation/eu-ai-act-2024-1689#article-13-para-1] The policy does not
address ensuring that the operation of the AI system is sufficiently
transparent to enable deployers to interpret its output and use it
appropriately.
```

**Observations:**
- **No FRIA leak.** Compare V4 1.5B which still produced one. Suppressed at 7B.
- **But: wrong-audience anchoring.** The query asked about transparency *for candidates* (data subjects). Article 13 para-1 of the EU AI Act is about transparency *to deployers* (the receiving organisation) — different audience entirely. The relevant provisions are Article 50 (transparency to natural persons), Article 13/14 GDPR (information at collection), and Article 22(3) GDPR (logic of automated decisions).
- This is **not a model-size issue**. BGE retrieval ranks Article 13 para-1 high because the chunk contains the word "transparent" multiple times. The LLM uses what retrieval surfaces. A 70B model would make the same error given the same retrieval input.
- This confirms the **Category 1 mitigation** (query expansion / disambiguation) documented in `evaluation-findings.md` — retrieval ranking is upstream of LLM capability.

---

### Q5 — FRIA target (the headline)

**Latency:** 170.5s.

**LLM output:**
```
### What the law requires
[regulation/eu-ai-act-2024-1689#article-27-para-1] Deployers must perform
a fundamental rights impact assessment prior to deploying a high-risk AI
system.

### What the policy says
The policy does not address performing a fundamental rights impact
assessment. [deployer-extras/novara-talentlens-dpia#preamble] focuses on
a Data Protection Impact Assessment (DPIA) for TalentLens, which is a
different type of assessment.

### Gap
[regulation/eu-ai-act-2024-1689#article-27-para-1] The policy does not
address the requirement for a fundamental rights impact assessment, which
is distinct from the DPIA mentioned in the policy.
```

**Observations:**
- **Clean, demo-quality output.** This is the strongest gap finding produced across any prompt-or-model combination tested so far.
- Section 1 cites the correct article (27 para-1) and states the obligation in a complete sentence. Compare V3 1.5B which produced only the citation without an obligation sentence; V4 1.5B which produced a full but factually wrong conclusion.
- Section 2 explicitly distinguishes FRIA from DPIA. **The 1.5B models could not make this distinction.** V3 conflated them (*"The DPIA is more comprehensive and covers fundamental rights"* — false); V4 accepted Novara's internal "Standard AI Feature" self-classification (also false).
- Section 3 is the headline finding the project's intentional silence target was designed to surface. It is correct, defensible, and reads naturally to a non-technical compliance audience.

---

## What this tells us — three findings

### 1. The FRIA leak is a small-model training-data prior, suppressed at 7B+

V3 → V4 → 7B is now a clean experimental sequence:
- V3 1.5B: leak present on Q2, Q3, Q4 (system prompt mentioned FRIA explicitly)
- V4 1.5B: leak persisted verbatim (system prompt no longer mentioned FRIA)
- V4 7B: leak gone on Q2, Q3, Q4 (same prompt, larger model)

This is empirically the strongest evidence the project has produced for the
training-data-prior-vs-prompt-design distinction. The cost difference between
1.5B and 7B is the only variable changed; the leak vanishes. **For the report's
Critical Analysis dimension, this is the cleanest negative-then-positive
empirical narrative we have.**

### 2. Model scale does *not* fix retrieval-side failures (Q4)

Q4's wrong-audience anchoring (Article 13 para-1 transparency-to-deployers
returned for a query about transparency-to-candidates) persists at 7B because
the failure is in BGE's ranking, not in the LLM's reasoning. Documented as
Category 1 mitigation territory in `evaluation-findings.md` (query expansion,
query disambiguation). **Confirms that retrieval improvements and LLM scale
are orthogonal levers.**

### 3. Model scale does *not* fix architectural single-obligation limits (Q1)

Q1 multi-facet on V4 7B produces a single-obligation output (Article 26
para-9), same shape as V4 1.5B, just without truncation. This is a Phase-1
trade-off: the simplified architecture sacrifices multi-facet decomposition
for simplicity. A 70B model running the same architecture would produce the
same shape. **Fixing Q1 requires either re-introducing decomposition (back
toward the chain architecture) or running the simplified path multiple times
with derived sub-questions. Both are mitigations beyond model-scale.**

---

## What this tells us about the next iteration

The three findings give a clean menu of next moves, each addressing a
different limitation:

- **Demo on Q5.** Use V4-7B output. Clean, defensible, distinguishes FRIA
  from DPIA, reads naturally. Headline gap-finding for the demo.
- **Q4 retrieval mitigation.** Implement Category 1 mitigation (query
  expansion to add adjacent legal-framework terms before retrieval).
  Should rescue Q4 by surfacing Article 50 / Article 22(3) / GDPR 13-14
  alongside the AI Act Article 13.
- **Q1 multi-facet mitigation (optional).** Implement light decomposition
  on the simplified path: split obviously multi-faceted queries, run
  `analyse()` per sub-question, concatenate. Crosses into chain territory
  but keeps each call a single-shot.
- **Q2 reading error.** New failure mode at 7B. Worth a follow-up test pass
  with a slightly stronger prompt (perhaps re-introducing a "if the policy
  literally says X, do not write 'policy does not address X'" rule). Defer
  until after Q1 / Q4 work.

---

## Status after this pass

- Production demo path: `src/simplified.py` with V4 prompt + BGE retrieval,
  Qwen 1.5B locally for daily iteration, Qwen 7B on Colab for the demo
  output capture.
- Recommended demo Q5 output: this pass's verbatim Section 3 above.
- Recommended report appendix table: V3-1.5B / V4-1.5B / V4-7B side-by-side
  on Q5 (showing the cleanest model-scale narrative).
- Latency anomaly to acknowledge in report: T4 + 7B fp16 + CPU offloading
  produces multi-minute generation per query. For a faster Colab run, swap
  to Qwen 3B or use 4-bit quantisation via `bitsandbytes`.
