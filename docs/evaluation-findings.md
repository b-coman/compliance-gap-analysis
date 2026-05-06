# Evaluation Findings

> Process, observations, and adaptations from the build-evaluation phase. Source material for the written report's Critical Analysis and Error Analysis sections, and for the presentation's Outcomes & Lessons Learned and Defence dimensions.
>
> Anchored in actual runs and verifiable outputs. No claims here that aren't grounded in observed evidence.

---

## What this document is

A working record of the testing sessions that examined whether the implemented system produces meaningful, accurate compliance gap analyses on the five hand-written test queries. The methodology was: run queries through the implemented architecture, manually compare outputs against the expectations in `docs/intentional-gaps.md`, identify systematic error categories, propose mitigations.

Findings here translate directly into report and presentation content, with explicit pointers to which marking dimensions they support.

---

## The journey, summarised

### Stage 1 — initial system evaluation (Llama 70B chain on Groq)

**What we did:** ran all five test queries through the production chain, end-to-end, against the live Groq Llama 70B backend. Configuration: `build_chain(use_routing=False)`, default τ=0.35, four-bucket retrieval per spec.

**What we observed:**

| Query | Status | Outcome |
|---|---|---|
| Q2 (red-teaming, single-facet, adequate expected) | ✅ Completed (144s) | 6 adequate, 14 partial — chain extracted obligations from Article 2 (Scope) instead of Articles 9 / 15 (where red-teaming actually sits in the EU AI Act) |
| Q5 (FRIA, silence target) | ✅ Completed (165s) | 19 partial, 1 adequate, 0 silent — chain extracted Article 26 obligations (registration, cooperation, deployer record-keeping) instead of Article 27 (Fundamental Rights Impact Assessment) |
| Q4 (ambiguous) | ❌ Token-budget rate limit | Daily 100,000-token Groq cap exhausted after Q2 + Q5 |
| Q3 (Article 22 sub-clauses) | ❌ Token-budget rate limit | Same |
| Q1 (multi-facet) | ❌ Token-budget rate limit | Same |

**What this told us:**

1. The chain ran end-to-end mechanically (no crashes, JSON parsed, schema validation passed). 189 unit tests pass.
2. The chain's outputs were **task-misaligned** on the headline silence target (Q5): the system was asked about FRIA (Article 27) and instead produced a register about registration and cooperation duties (Article 26).
3. The chain's outputs were **task-misaligned** on the calibration anchor (Q2): the system was asked about red-teaming and produced a register citing the AI Act's Scope article (Article 2).
4. **Free-tier rate limits make iterative development on this architecture infeasible.** Each query consumes ~50,000 tokens (10 LLM calls × ~5,000 tokens average). Daily budget supports two queries.

### Stage 2 — root-cause diagnosis

**What we did:** examined the retrieval layer in isolation (no LLM cost) to understand why the chain produced task-misaligned obligations.

**What we observed:**

For Q5 ("Have we performed a Fundamental Rights Impact Assessment under EU AI Act Article 27..."):
- Top-1 regulation retrieval hit: **Article 27 (para-1) at 0.704** — correct
- But the next four retrieved chunks were Article 26 paragraphs (deployer obligations cluster) at 0.62–0.67
- The chain's obligation-extraction step (CHN-03) extracted obligations from *all* retrieved chunks, not just the top hit, so Article 26 obligations dominated the output

For Q2 ("Does our policy address the red-teaming requirements before deploying a high-risk AI system..."):
- Top-5 regulation hits: **all from Article 26 and Article 2** (vocabulary-related), none from Articles 9 (risk management system) or 15 (accuracy, robustness, cybersecurity) where red-teaming requirements actually live
- The EU AI Act does not use the phrase "red-teaming"; it uses "risk management system" and "appropriate level of accuracy, robustness and cybersecurity." The embedding model (`multi-qa-MiniLM-L6-cos-v1`) does not bridge this vocabulary gap

For Q1 (multi-facet, covering Articles 13, 14, 22, 26):
- Top-8 regulation retrieval missed Articles 13, 14, and 22 entirely — only Article 26 paragraphs and a few neighbouring articles surfaced
- A single-vector dense-retrieval query cannot cleanly cover four distinct topics

**What this told us:** retrieval quality, not LLM quality, is the dominant determinant of system accuracy. The LLM is downstream of retrieval; it can only be as good as its inputs. When retrieval surfaces the right chunks, the LLM produces correct output (verifiable on Q5 once isolated). When retrieval surfaces wrong or incomplete chunks, the LLM either hallucinates to fill gaps (Q1) or reaches confident-wrong conclusions (Q2).

### Stage 3 — architectural simplification test

**What we did:** built a parallel single-call simplified architecture to test whether the chain's complexity was contributing to the failures, or whether retrieval-driven errors propagate regardless of architecture. Configuration: top-5 retrieval over regulation + deployer-side combined, single LLM call asking for a 3-section gap analysis. Backend: local Qwen 2.5-1.5B-Instruct (free, runs on CPU; capability proxy for Daria's intended Gemma 2-2B-it).

**What we observed:**

| Query | Outcome on simplified architecture |
|---|---|
| Q5 | Correctly identified the FRIA gap; clean structure once prompt was iterated to V3 |
| Q1 | Hallucinated 3 of 4 law-requirement sections; retrieval-driven failure (Articles 13, 14, 22 not retrieved → model fabricated plausible-sounding text under correct article citations) |
| Q2 | Retrieval pulled Article 26 instead of Articles 9 / 15; model produced an internally-contradictory output (correctly quoted the policy's red-teaming section but then concluded "the policy does not address" red-teaming because it was comparing against wrong-article law text) |

**What this told us:** simplification removed some failure modes (chain-step drift, token-budget exhaustion) but exposed retrieval-driven failures more starkly. The retrieval problem is architecture-independent — the simpler architecture is more honest about it because it doesn't have intermediate steps to mask the issue.

### Stage 4 — prompt engineering iterations on Q5

**What we did:** iterated the prompt on Q5 (the canonical success case) to test whether output quality issues could be addressed by prompt engineering alone. Three versions:

- **V1**: minimal prompt, raw-text completion mode
- **V2**: chat template, repetition_penalty=1.2, tighter instructions
- **V3**: chat template, repetition_penalty=1.05, explicit verbatim-quote instruction, anti-confusion guard distinguishing FRIA from DPIA

**What we observed:**

| Version | Format | Substance | Specific issue |
|---|---|---|---|
| V1 | Messy (output repeated 3×) | Mostly accurate | Repetition; prompt instructions echoed in output |
| V2 | Clean | Slightly off | Substituted "DPIA" for "FRIA" in summary (concept conflation between two related-but-distinct assessments) |
| V3 | Cleanest | Gap correct, law section wrong | Cited `[article-27-para-1]` but quoted text from Article 26 (citation/text binding hallucination) |

**What this told us:** prompt engineering significantly improves format but does not reliably fix substance issues at the small-model scale. Each iteration fixed one issue but exposed another. We hit the model's capability ceiling on tasks that require simultaneous citation-binding and verbatim quoting from retrieved chunks.

### Stage 5 — FLEX-3 escalation simulation (BGE-large vs MiniLM, retrieval-only)

**Why we tested this:** Stage 2's diagnosis attributed Q2's failure to the embedding layer specifically — `multi-qa-MiniLM-L6-cos-v1` (the course tutorial's default, ~22M parameters, trained on Q&A pairs) does not bridge the vocabulary gap between practitioner terms ("red-teaming") and the law's formal terminology ("risk management system"). Daria's strategic spec named `BAAI/bge-large-en-v1.5` as the FLEX-3 escalation target precisely for this failure mode. We invoked FLEX-3 in simulation to test the prediction empirically without committing code changes.

**Why BGE-large specifically (rationale):** the spec named it explicitly as the planned escalation target (FLEX-3 path). BGE was designed for retrieval tasks (BAAI General Embedding family) rather than Q&A; ~335M parameters vs MiniLM's ~22M; trained on more diverse text including legal/professional content; substantially stronger on the MTEB retrieval benchmark; free, open weights, runs locally — compatible with both the brief's free-tier compute constraint and the project's no-paid-tools rule. Testing the spec's named escalation is more defendable than picking some other embedding model arbitrarily.

**What we did:** built a parallel BGE retriever in memory, embedding the same 1140 chunks. Used BGE's recommended query-instruction prefix (`"Represent this sentence for searching relevant passages: "`) on queries. Did not modify any source code. Compared top-5 retrieval results across MiniLM and BGE for Q1, Q2, Q5; ran τ histogram on 10 hand-crafted obligations against the deployer corpus.

**What we observed (retrieval comparison):**

| Query | What changed under BGE |
|---|---|
| Q1 (multi-facet) | BGE pulled **Article 14 (para-5)** into top-5 — MiniLM missed it entirely. Article 14 is human oversight, one of Q1's four explicit facets. Q1's other facets still partially missed; multi-vector limit not fundamentally solved. |
| Q2 (red-teaming) | BGE pulled **Article 9 (para-8)** into top-5 — MiniLM missed it entirely. Article 9 is the risk management system article where red-teaming actually lives in the EU AI Act. The vocabulary-mismatch fix worked. Deployer-side: Novara §3.4 Red-Teaming jumped from 0.651 (MiniLM) to **0.810** (BGE) — sharper signal. |
| Q5 (FRIA) | BGE pulled **both Article 27 paragraphs** into top-5 (para-1 at 0.787, para-2 at 0.731). MiniLM only got para-1. Better Article 27 recall. |

**What we observed (τ histogram across 10 sample obligations):**

```
                MiniLM    BGE-large
MEDIAN max-sim    0.528    0.671
MEAN              0.519    0.660
MIN               0.425    0.571
MAX               0.617    0.779
```

Two distinct findings emerge from these numbers:

1. **BGE produces systematically higher similarity scores** (~0.14 higher median). Under the decisions.md §4 rule (`if median ≥ 0.35, recommended τ = median + 0.05`), the BGE-corpus τ should be ~0.72.

2. **MiniLM's median (0.528) is itself far above the current default τ = 0.35.** The minimum across 10 sample obligations was 0.425 — *every* sample obligation already scores above the silence threshold. This means with current MiniLM + τ=0.35 configuration, **almost nothing trips silence detection** regardless of corpus content. Per the spec's own decisions.md §4 rule, MiniLM-corpus τ should have been raised to ~0.58 at the freeze gate. The freeze-gate τ spot-check that would have surfaced this evidently was not run with representative obligations during the build.

**What this told us:** FLEX-3 (BGE escalation) empirically helps with the Category 1b vocabulary-mismatch failure mode, validated by Q2 surfacing Article 9 and Q1 surfacing Article 14 — both chunks that MiniLM missed entirely. It does not fundamentally solve Category 2 (multi-facet retrieval); BGE's Q1 results are better but still don't include all four target articles. Separately, the τ histogram revealed that the current τ=0.35 default was likely too low for the actual corpus distribution under either embedding model — a methodology finding about the freeze-gate calibration step that adds to the project's critical-analysis material.

**Important scope caveat:** Stage 5 tested *retrieval only*. The retrieval improvement is a necessary precondition for downstream improvement; whether the LLM converts that into better output is a separate empirical question, addressed in Stage 6.

### Stage 6 — end-to-end BGE test on Q5 (closes the Stage 5 scope question)

**Why we ran this:** Stage 5 left an explicit open question — does retrieval-layer improvement translate to LLM-output improvement? We ran the simplified architecture (Qwen 1.5B + V3 prompt) end-to-end with BGE retrieval substituted for MiniLM, on Q5 specifically. Same prompt, same model, same query. Only the retrieval layer changed. This isolates the effect of retrieval quality on system output.

**What we observed (side-by-side on Q5):**

V3 + MiniLM (previous run):
```
### What the law requires
[regulation/eu-ai-act-2024-1689#article-27-para-1] EU AI Act Article 27 (para-1)
Deployers of high-risk AI systems that are public authorities, or Union
institutions, bodies, offices or agencies shall comply with the registration
obligations referred to in Article 49.    ← WRONG TEXT (this is Article 26 content)

### What the policy says
The policy does not address [Fundamental Rights Impact Assessment].

### Gap
The policy does not require a Fundamental Rights Impact Assessment under
EU AI Act Article 27 for TalentLens as a deployer of an Annex III high-risk system.
```

V3 + BGE (this run):
```
### What the law requires
[EU AI Act Article 27 (para-1)]
Deployers of high-risk AI systems must perform a fundamental rights impact
assessment before deploying such systems.    ← CORRECT TEXT, matches Article 27

### What the policy says
The policy states that Novara TalentLens is classified as a Standard AI Feature
and does not require a fundamental rights impact assessment.

### Gap
The policy is silent on the requirement for a fundamental rights impact
assessment under EU AI Act Article 27.
```

**What this told us:**

1. **Citation hallucination resolved on Q5.** The Category 3 specimen (V3 cited Article 27 but quoted Article 26 text) does not reproduce when the retrieval surfaces both Article 27 paragraphs cleanly. The model now has the correct source text adjacent to the citation it produces.

2. **Policy section becomes substantive rather than absent.** MiniLM's "does not address" was correct but uninformative. BGE-fed run engages with the *specific* policy claim — that TalentLens is classified as a "Standard AI Feature" and therefore (per the policy's own logic) does not require a FRIA. This gives the compliance officer the *substance* of the policy's position, not just its absence. Mechanically, this happened because BGE's deployer-side top hits shifted from Governance Report excerpts (MiniLM: Executive Summary, Regulatory readiness) to **DPIA chunks** (BGE: Preamble, Risk Classification, Mitigations) — the documents where TalentLens's classification is actually documented.

3. **Gap statement is essentially equivalent.** Both runs correctly identify Q5's silence target. So the improvement is not in the headline finding but in supporting accuracy.

**Q5 result — what this validates:** the Stage 5 prediction held empirically on Q5 — better retrieval did translate to better LLM output. Specifically, Category 3 (citation/text binding hallucination) attaches to the LLM layer in principle but is *triggered* by retrieval surfacing partial or wrong-topic chunks. With cleaner retrieval inputs, the small model's binding behavior degrades less.

### Q2 result (red-teaming, end-to-end with BGE) — marginal improvement, output still wrong

**Retrieval:** BGE pulled **Article 9 (para-8)** into top-5 at rank #5, where MiniLM had missed it entirely (Article 9 is where red-teaming requirements actually live in the EU AI Act). Article 26 paragraphs still dominate the top of the list (paras 1, 7, 8 at scores 0.74–0.73), beating Article 9 at 0.72. Deployer side: §3.4 Red-Teaming hit at 0.810 (very strong, matches MiniLM behavior).

**LLM output:** The model still produced an internally-contradictory output. Section 1 quoted Article 26 (para-1) — instructions for use, **wrong topic for red-teaming**. Section 2 correctly quoted Novara §3.4 Red-Teaming (covering external red-teamers, prompt injection testing, etc.). Section 3 concluded *"the policy fails to adequately address this specific obligation,"* internally contradicting Section 2 which clearly demonstrates the policy *does* address red-teaming.

**What this told us about Q2:** BGE made Article 9 *visible* in retrieval but couldn't make it *dominant*. Article 26's deployer-flavored vocabulary still anchored the LLM's law section. The end-to-end output's internal contradiction persists. The Category 1b vocabulary-bridge improvement is necessary but not sufficient when Article 26's lexical overlap with the query terms is stronger than Article 9's.

### Q1 result (multi-facet, end-to-end with BGE) — substantially still wrong

**Retrieval:** BGE pulled **Article 14 (para-5)** into top-8 (correctly addressing one of Q1's four facets — human oversight). Article 22 (para-3) of the AI Act surfaced too, but this is the *AI Act's* Article 22 (authorised representatives), not the *GDPR's* Article 22 (automated decisions) that Q1 explicitly asked about. **Article 13 was missed entirely** despite being explicitly named in the query.

**LLM output (across the four provisions Q1 asked about):**

| Section | Model behavior | Reality |
|---|---|---|
| Article 13 (deployer instructions) | Quoted Article 26 para-9 text about DPIAs | Wrong article — no Article 13 chunk in top-8 retrieval; model picked Article 26 chunk that mentions "Article 13" inside its body |
| Article 14 (human oversight) | Quoted Article 14 para-5 text correctly | Correct quote — but the *policy section* fabricated text Novara's policy doesn't actually contain |
| Article 26 (logs / worker info) | Quoted Article 26 para-9 (DPIA-related text) — same content as Article 13 section | Wrong paragraph — Q1 asked about Article 26(6) logs and Article 26(7) worker info |
| Article 22 GDPR | Quoted EU AI Act Article 22 about authorised representatives | Fundamentally wrong topic — the AI Act and GDPR have different articles numbered 22; the model couldn't distinguish |

3 of 4 sections are still substantively wrong. The improvement vs. MiniLM is that Article 14 made it into retrieval, but the broader multi-facet failure persists at output level.

**What this told us about Q1:** Stage 5's prediction held — BGE alone does not solve Category 2 (multi-facet retrieval). Even with a stronger embedding model, a single query vector cannot cover four distinct topics simultaneously, and the retrieval gaps cause the LLM to hallucinate plausible-but-wrong text under correct-looking citations. **The mitigation that addresses Q1's failure is query decomposition (Category 2's identified mitigation), not embedding-model upgrade.**

### Stage 6 cross-query synthesis

| Query | Retrieval improvement (Stage 5) | End-to-end output (Stage 6) |
|---|---|---|
| Q5 (single-facet, vocab aligned) | Better — both Article 27 paragraphs surfaced | **Fixed** — citation hallucination resolved, policy section substantive |
| Q2 (single-facet, vocab mismatch) | Marginal — Article 9 surfaced at rank #5 but Article 26 still dominates | **Still wrong** — internal contradiction persists; Section 1 anchors on Article 26 instead of Article 9 |
| Q1 (multi-facet) | Marginal — Article 14 surfaced; Article 13 still missing | **Still wrong** — 3 of 4 sections hallucinate or quote wrong-article text |

**The pattern is now empirical, not predicted:** BGE is necessary but not sufficient. It addresses Category 1b (embedding-layer vocabulary mismatch) for queries where the retrieval can find the right chunks given a richer model (Q5). It does *not* address:

- **Multi-facet retrieval failure (Category 2)** — Q1 confirms this end-to-end. The mitigation is query decomposition, not embedding upgrade.
- **Wrong-article competition (a sub-pattern of Category 1)** — Q2 confirms this end-to-end. When the wrong-topic article (Article 26) has stronger lexical overlap with the query than the right-topic article (Article 9), even BGE's retrieval ranks the wrong-topic article higher. The mitigation is either query expansion or chunk-text article-number prefixing.

**A complete mitigation stack** for the failure modes documented across all three queries would need:

1. BGE-large embedding (Category 1b — vocabulary bridge) — Q5 validated end-to-end
2. Query decomposition before retrieval (Category 2 — multi-facet) — Q1 indicates this is required, not optional
3. Chunk-text article-number prefixing OR query expansion (Category 1a + 1c — lexical/practitioner gap) — Q2 indicates one of these is needed alongside BGE

No single mitigation is sufficient. The empirical finding is that the three categories are independent failure modes requiring distinct fixes.

### Stage 7 — diagnostic: chain + BGE on Q5 (does retrieval upgrade rescue the chain?)

**Why we ran this:** Stage 1's chain failure on Q5 (chain extracted Article 26 obligations instead of Article 27) was attributed to retrieval. Stage 5 tested the retrieval upgrade in isolation; Stage 6 tested it end-to-end through the simplified architecture. But we never tested it through the chain itself. If chain + BGE produces correct Q5 output, the chain's architectural complexity is salvageable; if it doesn't, the chain has deeper issues than retrieval.

**What we did:** built a parallel BGE retriever wrapper compatible with the chain's interface; passed it to `ComplianceGapChain` along with the existing `GroqLlama70B` adapter. No repo modifications. Ran Q5 with `verbose=True` to capture intermediate state across all chain steps. Cost: ~50K tokens, ~146 seconds wall time.

**What we observed (verbose chain trace on Q5):**

The decomposition (CHN-01) produced 4 sub-questions:
1. *"Is TalentLens considered a deployer of an AI system under the EU AI Act?"* — classification, not FRIA
2. *"Does the AI system deployed by TalentLens fall under Annex III as a high-risk system?"* — classification, not FRIA
3. *"What are the specific requirements for a Fundamental Rights Impact Assessment under EU AI Act Article 27?"* — FRIA ✓
4. *"Has a Fundamental Rights Impact Assessment been conducted for the TalentLens AI system in accordance with Article 27?"* — FRIA ✓

**Sub-questions 3 and 4 worked correctly with BGE retrieval.** Top regulation chunks for sub-Q3: Annex VIII para-26 (0.778), **Article 27 (para-1)** (0.771), **Article 27 (para-2)** (0.742), Article 26 (para-9) (0.739), **Article 27 (para-4)** (0.727). The chain's CHN-03 step then extracted *correct FRIA obligations* — *"the deployer shall conduct a fundamental rights impact assessment prior to deployment", "the deployer shall complement the data protection impact assessment with a fundamental rights impact assessment"*, etc. **This is a real improvement vs. the original Stage 1 chain run, which extracted Article 26 registration obligations even on Q5.**

**Sub-questions 1 and 2 went off-topic.** Their retrieval pulled Article 26, Article 3, Article 49, Article 6 (para-3), Article 80 — classification and registration territory. CHN-03 extracted obligations from those chunks: *"deployers shall comply with the registration obligations referred to in Article 49"*, *"the provider shall classify the AI system as high-risk where it materially influences..."*, etc. **10 of the 20 final register rows are about registration/classification, not about FRIA.**

**Final register: 20 rows, all classified `partial`. Zero silent. Zero adequate.** Confidence: 15× low, 5× high. Regulatory provisions cited (top 3): 10× *"EU AI Act Annex VIII (para-26)"*, 5× *"EU AI Act Article 26 (para-8)"*, 5× *"EU AI Act Article 6 (para-3)"*. None of the 20 rows is labelled with *"EU AI Act Article 27"* despite many of the obligations being about Article 27.

**What this told us — three architectural issues that retrieval upgrade does NOT fix:**

1. **Decomposition drift (architectural issue at CHN-01).** The decompose step produced 2 of 4 sub-questions that aren't about the user's original FRIA question — they're about classification preconditions. Correct strict reading of Q5 doesn't require asking "is TalentLens a deployer?" before asking about FRIA; the question explicitly states this. The chain has no mechanism to constrain decomposition to the user's original intent. This generated 10 off-topic rows in the final register regardless of retrieval quality.

2. **Silence detection empirically broken at τ=0.35.** Zero of 20 rows classified as silent — including on the FRIA obligations where the policy genuinely says nothing about Article 27. This confirms Stage 5's finding that τ=0.35 is far below the actual deployer-corpus max-sim distribution (BGE median 0.671). The chain's distinctive intellectual claim — *"silence is a retrieval property, not an LLM judgment"* — is not actually working as designed because the threshold is wrong. Per `decisions.md §4`, τ should have been recalibrated to ~0.72 for BGE; this calibration step is documented in the spec but was never executed empirically.

3. **`regulatory_provision` labeling is misleading.** Half the rows say *"EU AI Act Annex VIII (para-26)"* — the top retrieved chunk for sub-questions 3 and 4. But the actual *obligations* in those rows are about Article 27 FRIA, not Annex VIII. The chain's `_derive_regulatory_provision` helper picks the first retrieved chunk's article rather than tracing each obligation back to its actual source paragraph. **The output's most important user-facing field is wrong half the time.**

**Comparing chain + BGE (Stage 7) against simplified + BGE (Stage 6) on Q5:**

| Aspect | Simplified + BGE (Stage 6) | Chain + BGE (Stage 7) |
|---|---|---|
| Output structure | One 3-section text response | 20 structured register rows |
| Gap finding correctness | ✓ Correctly identifies FRIA gap | ✗ Zero silent rows; FRIA never flagged as missing |
| Cited provision | Article 27 (correct) | Annex VIII para-26 / Article 26 / Article 6 (none cite Article 27 even when obligation is about Article 27) |
| On-topic content | All sections about FRIA | 10/20 rows about classification/registration (off-topic) |
| Token cost | ~3K tokens | ~50K tokens (~17×) |
| Runtime (cold) | ~21 seconds | ~146 seconds (~7×) |
| Reads usefully to a compliance officer | Yes — direct answer | No — must filter 20 rows to find the 0 actually-silent FRIA findings |

**What this changes for the project's path decision:**

Path C (chain + 8B local model on Colab to avoid Groq cost limits) is no longer attractive. Even with cleaner retrieval and sufficient model capability, the chain produces unusable output on Q5 because of three independent architectural issues:

- Decomposition produces off-topic sub-questions
- Silence detection is empirically miscalibrated
- Provenance labeling is unreliable

Each would require substantial fixing — query-intent constraint at CHN-01, τ recalibration at CHN-04, and per-obligation provenance tracing at CHN-05. None of these is a model-swap fix.

**Path A (simplified architecture) is empirically the right call.** Not because the chain is impossible to make work, but because the chain's architectural complexity introduces failure modes that the simpler approach avoids entirely — and fixing them costs more engineering time than rebuilding around the simpler architecture.

---

### Stage 8 — model-scale comparison on the simplified path (V4 7B Colab vs V4 1.5B local)

The V4 prompt-hygiene experiment (`docs/test-passes/v4-qwen-1.5b-prompt-hygiene.md`) produced a negative finding: removing the topic-specific FRIA examples from the V3 system prompt did *not* suppress the "FRIA leak" on Q2, Q3, and Q4 at Qwen 1.5B scale. The output text continued to insert "fundamental rights impact assessment" into Section 3 of unrelated queries, verbatim. We hypothesised the cause was a *training-data prior* — at 1.5B parameters, the model treats *deployer + high-risk + obligation* as strongly co-occurring with *FRIA* and reaches for that phrase regardless of instruction.

Stage 8 tests this hypothesis directly by holding the V4 prompt and BGE retrieval constant and changing only the model size. Run on Colab Tesla T4 GPU with `Qwen/Qwen2.5-7B-Instruct` (fp16, `device_map="auto"` to handle the model's footprint relative to T4's 16 GB). Full verbatim outputs in `docs/test-passes/v4-qwen-7b-colab.md`.

**Headline result: the FRIA leak is suppressed at 7B+.**

| Query | V4 1.5B local | V4 7B Colab |
|---|---|---|
| Q2 red-teaming | FRIA leak (Section 3): *"policy does not address performing a fundamental rights impact assessment"* | **No leak.** Cites Article 9 para-8 testing requirement |
| Q3 Article 22 | FRIA leak (Section 3): *"fundamental rights impact assessment required by UK GDPR Article 22"* (fabricated) | **No leak.** Substantively correct: identifies missing sub-clauses (human intervention, contest, point of view) |
| Q4 transparency | FRIA leak persisted in Section 3 | **No leak.** New failure mode (wrong-audience: cites Article 13 para-1 transparency-to-deployers for a query about transparency-to-candidates) |
| Q5 FRIA target | Wrong conclusion (accepted Novara's "Standard AI Feature" self-classification) | **Clean and correct.** Cites Article 27 para-1, distinguishes FRIA from DPIA explicitly |

This is now a clean three-step empirical sequence: V3 1.5B (leak present, prompt mentions FRIA) → V4 1.5B (leak persists, prompt no longer mentions FRIA) → V4 7B (leak gone, same prompt as V4 1.5B). The only variable across the second and third steps is model size; the leak vanishes. **The 1.5B-vs-7B substitution is the cleanest empirical demonstration the project has of model-scale capability differences on a substantive failure mode.**

**Three orthogonal findings consolidate from Stage 8:**

1. **Training-data prior at small-model scale is real and substantial.** The FRIA leak in V3 1.5B and V4 1.5B was a probabilistic association between *deployer + high-risk + obligation* prompts and *fundamental rights impact assessment* outputs. At 7B the association is suppressed. This is not a prompt-design problem; we tried prompt design (V3 → V4) and it did nothing. It is a *parameter-count problem*, addressable only by scaling.

2. **Retrieval failures are upstream of LLM size.** Q4 wrong-audience anchoring persists at 7B because BGE-large ranks Article 13 para-1 highest on lexical overlap with the query word "transparency". A 70B model running the same retrieval input would make the same error. This validates Category 1 mitigation territory (query expansion / disambiguation) as orthogonal to LLM-scale work.

3. **Architectural single-obligation limits are not fixable by scale.** Q1 multi-facet on V4 7B still produces single-obligation output, the same shape as V4 1.5B (just without truncation). The simplified architecture is by construction single-call, single-output. Scaling the LLM does not introduce decomposition. Mitigations are architectural (light decomposition before `analyse()`), not model-side.

**Anomaly worth recording:** Q1 latency was 696.8s on Colab T4. T4's 16 GB GPU is ~2 GB short of full residence for fp16 7B + activations. `device_map="auto"` handled this by offloading some layers to CPU, surviving the OOM but generating slowly via per-token GPU↔CPU transfers. Future Colab runs should use Qwen 3B (full GPU residence) or 4-bit quantisation. *The anomaly is environment-bound, not model-bound; output substance is unaffected.*

**Q5 is now demo-quality.** The 7B Section 3 reads: *"The policy does not address the requirement for a fundamental rights impact assessment, which is distinct from the DPIA mentioned in the policy."* Clean, accurate, defensible, FRIA-vs-DPIA explicit. This is the strongest gap-finding output any prompt-or-model combination has produced; recommended as the demo-day Q5 output.

**What Stage 8 does NOT establish:**
- That 7B is the right *production* default. The local 1.5B path remains the demo-reliability default (no Colab dependency, no API keys, no daily limits). 7B is the *evaluation* default for empirical comparison points.
- That every failure mode is now categorised. Q2 produced a *new* failure mode at 7B (a reading error: model says "policy does not mention testing prior to deployment" while citing a §3.4 chunk that explicitly addresses Gate G2 pre-deployment red-teaming). Worth a follow-up test pass.
- That CPU offloading is acceptable. It is a workaround for this specific session, not a recommended pattern. Documented for the next reader so they don't conclude *"7B is unusable"*.

**For the report's Critical Analysis dimension,** Stage 8 contributes:
- The cleanest empirical narrative arc the project has on prompt-design-vs-model-scale: V3 → V4 (negative finding) → 7B (positive finding).
- An evidence-based claim that the three error categories in this document attach to *three different layers* of the system, with three different mitigation paths — not one homogeneous "improve the LLM" answer.
- A defensible position on the deployment trade-off: small-model demo for reliability, larger-model evaluation for substance, with the empirical grounds for both choices documented.

---

## The retrieval-and-generation pipeline as five layers

Before walking through error categories, it's useful to disambiguate where in the system each failure mode actually attaches. The RAG pipeline has five distinct layers:

```
1. Source documents (the law itself, the policy itself)
        ↓
2. Pre-processing (PDF → text; furniture stripped)
        ↓
3. Chunking (text → indexed chunks with metadata)
        ↓
4. Embedding (chunks → vectors; query → vector)
        ↓
5. Retrieval algorithm (cosine top-k)
        ↓
LLM (generation over retrieved chunks)
```

When we say "retrieval failed," the proximate cause can sit at any of layers 2–5. The retriever code itself (layer 5 implementation) is correct — verified by reading `src/retrieval.py` against the week-7 tutorial pattern. The failures we observed all attach to one of the upstream layers.

This layered view matters for the report's Critical Analysis dimension because it forces precision in attributing each finding to its actual root cause, rather than waving at "retrieval problems" generically.

---

## Three error categories (Error Analysis Log)

The brief requires "three categories of error types" with "prompt engineering, architectural mitigations or deployment mitigations" for each. Our testing produced exactly three, with empirical specimens. Each category is annotated with the layer where its root cause sits.

### Category 1 — Retrieval vocabulary mismatch (layers 3 + 4 + query side)

**Specimen:** Q2 ("red-teaming requirements") retrieved Article 26 (instructions for use) and Article 2 (Scope) instead of Articles 9 (risk management) and 15 (robustness, cybersecurity), where red-teaming requirements actually appear.

**Root cause:** the EU AI Act uses formal regulatory terminology ("risk management system", "appropriate level of accuracy, robustness and cybersecurity") that doesn't lexically overlap with practitioner terminology ("red-teaming"). Verifiable: `grep -i red-teaming` against the AI Act `.txt` returns zero matches.

This category decomposes into three sub-causes, each at a different layer of the pipeline. Each has its own mitigation. Treating them together as "retrieval problems" obscures the diagnosis; treating them separately points to specific, layered fixes.

**Sub-cause 1a — Chunk text lacks article-level lexical context (layer 3, chunking).**
Chunks are paragraph-level (e.g., `Article 27 (para-1)`) and the article identifier lives in metadata only — it does not appear in the chunk text body. The Article 27 chunk text starts with `"1. Prior to deploying a high-risk AI system referred to in Article 6(2)..."` rather than `"Article 27. Fundamental rights impact assessment for high-risk AI systems. 1. Prior to deploying..."`. A query containing "Article 27" or "fundamental rights impact assessment" therefore does not gain the lexical-overlap boost it would otherwise have.

*Mitigation (chunking-layer):* prefix each chunk with its article number and title during ingestion. Roughly 10 lines of additional code in `src/ingestion.py`. Cheap; modest help on queries that name articles or use legal-domain phrases verbatim.

**Sub-cause 1b — Embedding model doesn't bridge legal-jargon vocabulary (layer 4, embedding).**
The course-default model (`multi-qa-MiniLM-L6-cos-v1`, ~22M parameters, trained on Q&A pairs from MS MARCO and similar corpora) was not trained on legal text. It captures common Q&A vocabulary alignments but lacks the conceptual mapping between practitioner terms and regulatory terminology — between "red-teaming" and "risk management system", or between "FRIA" and "fundamental rights impact assessment."

*Mitigation (embedding-layer):* swap to `bge-large-en-v1.5`. The strategic spec's FLEX-3 path explicitly named this model as the escalation target for retrieval-recall problems on legal text.

*Empirical validation (Stages 5 + 6):* we tested this swap in simulation. Stage 5 measured the retrieval-layer change: on Q2 (red-teaming), BGE pulled Article 9 (risk management) into top-5 where MiniLM missed it entirely; on Q1 (multi-facet), BGE pulled Article 14 (human oversight) into top-5 where MiniLM missed it; on Q5 (FRIA), BGE retrieved both paragraphs of Article 27 versus MiniLM's single paragraph. Stage 6 measured the end-to-end effect on Q5: with BGE retrieval feeding the simplified architecture, the LLM's Category 3 citation hallucination on Q5 did not reproduce, and the policy-side analysis became substantively more useful. The vocabulary-bridge hypothesis is empirically validated at retrieval level, and translates to LLM-output improvement in at least the Q5 case. BGE does not fully solve Category 2 (multi-facet) — Q1 still doesn't get all four target articles — confirming that this layer is necessary but not sufficient for that distinct failure mode.

**Sub-cause 1c — Query terminology doesn't match law terminology (query side).**
The compliance officer's natural query vocabulary ("red-teaming", "FRIA", "DPIA") often diverges from the law's vocabulary ("risk management system", "fundamental rights impact assessment", "data protection impact assessment"). Even with a perfect embedding model and perfectly-enriched chunks, this gap cannot be fully closed at the retrieval-time embedding step alone.

*Mitigation (query-side):* query expansion — an LLM rewrites the query into 3–5 alternative phrasings before retrieval, retrieves on each, unions the results. ~1 extra LLM call (~300 tokens). Specifically addresses the practitioner-vs-legal vocabulary gap.

*Deployment alternative:* domain glossary — a pre-computed manual mapping from common practitioner terms to legal-text phrasings, applied deterministically as query rewriting. No LLM cost; requires curation effort upfront.

### Category 2 — Multi-facet retrieval failure (layer 5, retrieval algorithm)

**Specimen:** Q1 (covering Articles 13, 14, 22, 26 in one query) retrieved only Article 26 paragraphs and adjacent articles. Articles 13, 14, and 22 were absent from the top-8 results entirely.

**Root cause:** at the retrieval algorithm layer, dense-vector retrieval embeds the entire query as one vector. A multi-topic query produces an averaged vector that doesn't strongly match any of its constituent topics. This is a fundamental property of single-vector dense retrieval, not an implementation bug — the algorithm cannot return chunks that aren't aligned to the averaged query vector.

This is at a different pipeline layer than Category 1's sub-causes. Better chunks or a better embedding model don't fix it; the *algorithm* needs to handle multi-topic queries differently.

**Mitigations:**
- *Architectural (query-decomposition step):* split the query into focused sub-questions before retrieval. Run retrieval per sub-question, then union. The strategic spec's CHN-01 step does exactly this — and that step was correctly designed. The chain's failures on Q1 are downstream (CHN-03 obligation extraction drift), not at CHN-01. The simplified single-call architecture's Q1 failure happened because we removed CHN-01 along with the rest of the chain; restoring decomposition while keeping the rest simple is the natural recovery path.
- *Cost:* one extra LLM call (~250 tokens) plus per-sub-question retrieval. Comfortably within free-tier budget.

*Empirical confirmation that this category requires its own mitigation (Stage 6):* the BGE upgrade tested in Stages 5 and 6 did *not* fix Q1's multi-facet failure end-to-end. Even with better retrieval, the single-vector embedding still missed Article 13 entirely and confused AI Act Article 22 with GDPR Article 22. Three of four Q1 output sections remained wrong. This confirms empirically that Category 2 is independent of Category 1 — addressing the embedding model alone does not solve multi-facet, and query decomposition is a necessary additional mitigation, not an alternative one.

### Category 3 — Citation/text binding hallucination (LLM layer)

**Specimen 1:** V3 prompt on Q5 produced `[article-27-para-1]` citation followed by text actually drawn from Article 26 paragraph 8. The chunk identifier label and the quoted content did not match.

**Specimen 2:** Q1 on the simplified architecture produced citations to Articles 13, 14, 22 with fabricated legal text under each — text that appears nowhere in those articles. The model, under multi-facet pressure with incomplete retrieval, invented plausible-sounding legal language to fill gaps where retrieval had not supplied source text.

**Root cause:** this category attaches to the *LLM layer*, not retrieval. Given correctly-retrieved chunks, a small generative model (~1.5B–2B parameters) cannot reliably perform two reasoning tasks simultaneously: bind a chunk_id label to its specific source text, AND summarise the source text in legal language. The two tasks compete for the model's available reasoning capacity. Q1's specimen compounds this: when retrieval also fails to supply source text for a topic, the model's tendency is to fabricate plausible content rather than refuse.

**Mitigations:**
- *Prompt engineering (decouple citation from generation):* numbered-chunk indirection — present chunks as a numbered list, ask the model to refer to chunks by number ("chunk 1 establishes..."), then expand numbers to full chunk identifiers in post-processing code. The model only has to track 5–10 small integers, not full chunk-id strings.
- *Architectural (hybrid LLM + rule-based verification):* the LLM produces structure with chunk references; a deterministic post-processor verifies each reference matches the cited chunk's actual content (substring or embedding similarity check). Wrong references are flagged or stripped before the output reaches the user.
- *Deployment (larger model):* citation-binding capacity scales with model size. Larger models (7B+ parameters) handle this task substantially more reliably. Within the free-tier-only constraint, this means moving to a larger Colab-resident model rather than the 1.5B–2B size class.
- *Empirical observation (Stage 6):* the Q5 hallucination specimen here did not reproduce when retrieval was upgraded to BGE-large. This suggests the citation-binding failure was being *triggered* by retrieval surfacing partially-relevant or wrong-topic chunks, not by an inherent inability of the small model. Cleaner retrieval inputs reduce citation-binding errors at the LLM layer even without any of the three mitigations above. This is an interaction effect between Categories 1 and 3, not a substitute for mitigating Category 3 directly — but it does mean that addressing Category 1 first may resolve some Category 3 specimens for free.

### Related observation — training-data prior bias (LLM layer)

Distinct from citation-binding hallucination, but adjacent. On Q2 (red-teaming), Q3 (GDPR Article 22), and Q4 (transparency) the simplified architecture's output included the phrase *"fundamental rights impact assessment"* in the Gap section — even though none of those queries asked about FRIA, and none of their retrieved chunks contained the phrase. The model generated it from a training-data prior: Qwen 1.5B has learned that *EU AI Act + deployer + high-risk + obligation* often co-occurs with FRIA, and surfaces the phrase regardless of whether the retrieved context supports it.

**Empirical evidence:** documented in test passes `v3-qwen-1.5b-local-baseline.md` (Pattern 1, "FRIA leak") and `v4-qwen-1.5b-prompt-hygiene.md` (the V4 attempt to fix it via prompt-hygiene changes — removing FRIA mentions from the system prompt — *did not* eliminate the leak. The phrase still appeared in Q2/Q3/Q4 Section 3 outputs).

**Root cause:** at small-model scale, training-data associations dominate retrieval-context grounding when the retrieved context doesn't strongly signal an answer. The model "fills in" with priors. This is fundamentally a model-capability constraint, not a prompt-design issue. V4 confirmed this empirically.

**Possible mitigations** (none implemented; enumerated for the report's discussion of future work):

- *Prompt-side (limited effectiveness at this model scale):* aggressive negative prompting that explicitly forbids phrases not present in the passages. We did not test the strongest variant; based on V4 evidence, predicted to be marginal at most.
- *Structural (high effectiveness, modest implementation cost):* post-hoc grounding verification — a deterministic Python pass that, after the LLM produces output, checks each named obligation / cited phrase against the retrieved chunks' text content; strips or flags claims the chunks don't support. Implements the "hybrid LLM + rule-based verification" architectural pattern from Category 3's mitigation list.
- *Decoding-time (more dependencies):* constrained-decoding libraries (Outlines, Guidance, lm-format-enforcer) that restrict the LLM's output to a grammar including only allowed tokens or phrases. Adds dependencies; may overfit to citation format at the cost of natural-language quality in the Gap section.
- *Architectural (most aggressive, biggest scope shift):* verbatim-quote-only output — reframe the Gap section as a structured selection from chunks rather than free-form generation. Largest reliability gain but loses the natural-language polish that makes the demo readable.
- *Deployment (the most reliable but model-size-dependent):* larger model. Bigger models hold retrieved context dominant over training-data priors more reliably. Stage 7's chain test on Llama 70B showed correct FRIA obligations extracted from sub-questions where retrieval surfaced Article 27 chunks — i.e., a 70B model behaved as one would hope and a 1.5B model does not.

The current `src/simplified.py` deployment does not implement any of these mitigations. Within the free-tier compute constraint, the practically-realistic ones are *post-hoc grounding verification* (low cost, deterministic, can pair with any model) and *deployment on a 7B+ model on Colab GPU* (proven retrieval-grounding by larger model size).

---

## Architectural insight

The project's distinctive intellectual claim shifts from *"we built a sophisticated multi-step chain with threshold-grounded silence detection"* to:

**"We evaluated a RAG-based compliance gap analysis system across two architectural variants. Retrieval quality is one significant determinant of accuracy — addressed by FLEX-3 (BGE-large) and validated empirically. But the multi-step chain has additional architectural failure modes that retrieval improvement does not fix: decomposition drift (CHN-01 produces off-topic sub-questions), miscalibrated silence detection (the spec's freeze-gate τ recalibration step was never empirically executed), and unreliable provenance labelling (regulatory_provision picks the first retrieved chunk per sub-question rather than tracing per obligation). We pivoted to a simplified single-call architecture that avoids these failure modes by removing the intermediate decomposition and threshold-grounded silence detection steps, and produces useful output on the headline silence target."**

This is a stronger claim. It's evidence-based (Stages 1, 5, 6, 7, 8 each contribute verifiable specimens). It's reproducible (the failure modes have logged outputs; the mitigations have clear paths). It's defendable (each architectural decision has a why and a how-tested-it).

---

## Mapping to assessment criteria

### Component 1 — Written Report (50% of module mark)

| Dimension | Marks | How this document supports |
|---|---|---|
| Presentation & Clarity | 20 | Provides a structured narrative for the report's discussion sections — the journey arc (build → evaluate → diagnose → simplify → iterate) is logical and well-paced. |
| Problem definition, system design & interaction logic | 20 | The "why two architectures" framing is a coherent system-design argument. |
| Implementation | 20 | The implementation itself sits in `src/`; this document supports it by anchoring why each architectural choice was made and how it was tested. |
| **Evaluation protocols** | **20** | This document's Stage 1–8 *is* the evaluation. Tabulated outputs across queries, against a manually-curated gold set (`intentional-gaps.md`), with documented methodology. Stage 5 adds an empirically-measured comparison between embedding models (MiniLM vs BGE-large) and an empirical τ histogram showing the original threshold was likely uncalibrated. Stage 6 closes the Stage 5 scope question end-to-end on Q5 for the simplified architecture. Stage 7 closes the parallel question for the chain — finding that even with BGE the chain has three independent architectural failure modes (decomposition drift, miscalibrated silence detection, unreliable provenance) that retrieval upgrade does not fix. Stage 8 holds prompt and retrieval constant and varies model size (Qwen 1.5B local → Qwen 7B Colab) on the simplified path, demonstrating that the small-model "FRIA leak" is a training-data prior suppressed at 7B+, while retrieval-side and architectural failure modes persist — the cleanest empirical narrative the project has on prompt-vs-scale separation. Currently the report's weakest dimension; this document directly addresses it. |
| **Critical Analysis** | **20** | The brief explicitly names "task misalignment" and "hallucination" as the systematic-error categories strong reports identify. We have specimens of both, plus a third category (retrieval vocabulary mismatch). Mitigations for each. The retrieval-as-bottleneck insight is exactly the "overarching critical analysis of design choices and the performance of the system overall" the rubric describes. |

### Component 2 — Presentation (50% of module mark)

| Dimension | Marks | How this document supports |
|---|---|---|
| Presentation & Clarity | 20 | The journey arc translates well to a slide narrative for non-technical audience: "We built X, tested it, found these specific problems, simplified to Y, found new problems, learned what really matters." |
| Technical Communication | 20 | The retrieval-vs-LLM diagnosis is a powerful explanatory frame for non-technical markers — concrete and visual ("the system can only be as good as the law passages it pulls in"). |
| **Justification of Choices** | **20** | Every architectural choice has a Considered Alternative + Why Not + What We Found. Llama 70B vs local model. Complex chain vs single call. Multi-corpus vs two-corpus. Rich material for this dimension. |
| **Outcomes, Findings & Lessons Learned** | **20** | This entire document is "what worked, what didn't, and why." Q5 worked on the simplified architecture; Q1 and Q2 didn't, and we know specifically why. |
| **Defence** | **20** | Daria can defend each finding from the spec, decisions doc, and this document — without ever needing to read Python. The arguments are grounded in observable model outputs and retrieval scores, not in implementation details. |

### Specific brief requirements explicitly satisfied

> "Document three categories of error types encountered when hand-analysing the outputs (e.g., task misalignment, hallucination, or safety breaches) and the prompt engineering, architectural mitigations or deployment mitigations you could employ to address them."

✅ Three categories documented with specimens, root causes, and three classes of mitigation per category (prompt-engineering, architectural, deployment).

> "Evaluation Strategy: You must define the axes of evaluation for your system... You should justify your choice of evaluation method, such as human judgement, automatic metrics, or validated LLM-as-a-judge, and explain how you account for model uncertainty."

✅ Axes: gap-finding accuracy, citation correctness, format reliability. Method: human judgement against a manually-curated gold set (`intentional-gaps.md`). Model uncertainty: tested across two model-size classes (Llama 70B at the high end via Groq; Qwen 1.5B at the low end locally) to surface size-dependent failure modes — see Category 3 (citation binding) which is small-model-specific.

> "Discuss scaling and automation considerations: how would this system remain reliable if deployed at scale?"

Not directly covered here but easy to add: the daily-token-budget finding (Stage 1) is exactly the scaling consideration. At free-tier rate limits the system supports two queries per day — clearly not deployable at compliance-team scale. Scale-up requires either paid API tier, local model serving infrastructure, or smaller/cheaper-per-query architecture (as our simplified variant attempts).

---

## Suggested usage

### In the written report

The "Findings" section (1 page in my proposed report structure) should be built from § "Three error categories" of this document, with abbreviated treatment but the same three-category structure. Specifically:

1. Open with the retrieval-as-dominant-failure-mode insight as the synthesising claim.
2. Present three error categories, one paragraph each, with specimen + mitigation.
3. Close with the architectural insight (the reframed distinctive claim).

The "Evaluation methodology and results" section (0.75 page) should distil § "The journey, summarised" stages 1–3, with the Q1/Q2/Q5 outcome table.

The "Architectural pivot rationale" section (0.5 page) should explain why the simplified architecture was tested (free-tier compatibility, retrieval-failure-mode visibility) and what it revealed.

The appendix should carry: full prompts (V1, V2, V3), full model output transcripts for Q5 across all three prompts, retrieval-score tables, and the live-API rate-limit error message verbatim.

### In the presentation

Slide pacing for a 10-minute non-technical presentation:

| Slide | Content | Pulls from |
|---|---|---|
| 1 | The problem (compliance gap analysis at scale) | Spec § 1, brief § Information Processing Perspectives "Policy Guidance" |
| 2 | Architecture overview (3-box diagram) | High-level summary |
| 3 | Live demo (Q5 on simplified architecture, working) | Stage 3 Q5 output |
| 4 | First architecture — initial findings | Stage 1 outcomes table |
| 5 | What we discovered — retrieval as bottleneck | Stage 2 diagnosis |
| 6 | Three error categories, one specimen each | Document § "Three error categories" |
| 7 | Mitigations and lessons learned | Each category's mitigation block |
| 8 | Limitations and what we'd change with more resources | Architectural insight |
| 9 | Closing | The reframed distinctive claim |

### In Q&A defence

Most likely marker questions, with defendable answers:

- *"Why did the system fail on Q5?"* → "Two reasons. The retrieval layer co-mixed Article 26 and Article 27 chunks because they're semantically adjacent in the law's structure. The chain's obligation-extraction step then extracted from whatever was retrieved, dominated by Article 26 because it had more paragraphs. We documented this and propose two mitigations: cleaner retrieval through query decomposition, or a hybrid where citations are post-verified."

- *"Why didn't you fix it?"* → "Within the project's free-tier compute constraint, fixing the retrieval issue requires either a larger embedding model (one of our identified mitigations, the spec's FLEX-3 path) or a different retrieval algorithm (BM25 hybrid, excluded by spec D-004). Both are documented as future work. Within scope, we identified the root cause and demonstrated reproducibility."

- *"Why did you use Claude Code?"* → "The course is INST0100: Generative AI for Information Processing. Using a generative AI coding agent to build a generative AI system *is* the course's subject matter. I authored the specifications and architectural decisions; the agent implemented to those specifications. The findings in this document are mine — I can't read Python, but I can read English, and the analytical work is at the architectural level the rubric grades on Defence."

- *"What's the most surprising thing you learned?"* → "That retrieval quality determines accuracy more than model quality. We found Q5 succeeded on a 1.5-billion-parameter local model and failed on a 70-billion-parameter cloud model — because the smaller model was given a cleaner retrieval input. The dominant lever for compliance-RAG systems is corpus structure and retrieval, not LLM capability."

---

## What this document does NOT claim

- That the system will receive a particular grade.
- That the simplified architecture is "better" than the chain — it's better on Q5 and Q2-deployer-side, worse on Q1.
- That all three error categories will be fully mitigated by the proposed mitigations. Sub-cause 1b (embedding-layer vocabulary mismatch) is empirically validated by Stage 5; the others are proposed but not measured.
- That **BGE-large improves end-to-end system output across all queries on either architecture** — Stages 6 and 7 measured this with mixed results. On the *simplified architecture*: Q5 was fixed end-to-end; Q1 and Q2 substantially still wrong. On the *chain*: Q5 retrieval improved (Article 27 chunks correctly used in CHN-03 obligation extraction), but the chain's other architectural issues — decomposition drift, miscalibrated silence detection, unreliable provenance labeling — produced an unusable 20-row register where 10 rows were off-topic and zero rows were classified as silent despite Q5 being the canary silence target. BGE is necessary but not sufficient; the chain has architectural problems beyond retrieval that retrieval upgrade does not fix.
- That the τ histogram observation is fully diagnostic. Stage 5 used 10 hand-crafted obligations rather than chain-extracted obligations. Real CHN-03 output may produce different score distributions. The conclusion ("τ=0.35 was likely too low for the corpus") is empirically suggested but would need confirmation against actual chain extraction output for a stronger claim.
- That every test query has been run live on every architecture/model combination — Q1, Q3, Q4 were not completed live on the chain due to rate-limit exhaustion. Q3 and Q4 have not been tested on the simplified architecture either.
- That `intentional-gaps.md` is a fully validated gold set; it is a manually-curated working set adequate for this evaluation but would need expert review for a higher-confidence claim.

These limitations belong in the report's "Limitations" section. Stating them in writing is itself rubric-positive content.
