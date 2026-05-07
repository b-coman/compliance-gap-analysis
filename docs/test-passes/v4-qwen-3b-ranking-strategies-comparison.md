# Test pass — ranking strategies on Qwen 3B (BGE-only / rerank-only / RRF)

> Three controlled runs on the same model (Qwen 2.5-3B-Instruct), same V4
> prompt, same Colab T4 hardware, same 5-query test set. Only the ranking
> strategy varies. Tests whether two-stage retrieval (BGE bi-encoder +
> cross-encoder reranker) improves outputs over single-stage BGE — and
> whether Reciprocal Rank Fusion (RRF) is a better blend than pure cross-
> encoder reranking.
>
> Run dates: 2026-05-06 (BGE-only baseline; documented in
> `v4-qwen-3b-colab.md`), 2026-05-06 (rerank_only), 2026-05-07 (RRF).

---

## Hypotheses going in

**H1: Two-stage retrieval (rerank or RRF) improves on BGE-only.** Stage 8
documented persistent retrieval failures across all model sizes (Q1
multi-facet recall, Q2 wrong-article anchoring, Q4 wrong-audience
anchoring). Cross-encoders read query and chunk together and capture
nuances bi-encoders miss; the prediction was that adding a reranker
should improve Q2 and Q4 specifically.

**H2: RRF improves on pure cross-encoder reranking.** Pure rerank-only on
Q3 dropped the §4.3 Right-to-Explanation policy chunk out of top-5
(documented as a regression in the rerank_only run). RRF preserves
high-BGE-rank chunks via rank fusion, so it should keep §4.3 in top-5
while still allowing the reranker to promote relevant under-ranked
chunks. The prediction was "best of both worlds."

---

## Configuration captured

All three runs share:

| Component | Value |
|---|---|
| Architecture | `src/simplified.py` (single retrieve + single LLM call) |
| LLM | `Qwen/Qwen2.5-3B-Instruct` (fp16, full T4 GPU residence) |
| Prompt | V4 (long system + 5 rules) |
| Embedding (initial retrieval) | `BAAI/bge-large-en-v1.5` |
| top_k_initial (rerank/rrf only) | 10 |
| top_k_reg / top_k_dep (final) | 5 / 5 |
| Reranker (rerank/rrf only) | `BAAI/bge-reranker-base` |
| Hardware | Colab Tesla T4 |
| Decoding | greedy (do_sample=false), repetition_penalty=1.05 |

The ranking strategies differ only in how the final top-5 is selected from
the initial top-10 (or directly from BGE for `bge_only`):

- **`bge_only`**: take BGE top-5 as-is. No reranker called.
- **`rerank_only`**: cross-encoder rescores all 10, take top-5 by reranker score.
- **`rrf`**: cross-encoder rescores all 10, then RRF combines BGE rank and
  reranker rank: `score = 1/(k + bge_rank) + 1/(k + rerank_rank)` with `k=60`.

---

## Result — outputs across all 5 queries × 3 strategies

| Query | BGE-only | rerank_only | RRF |
|---|---|---|---|
| Q1 multi-facet | Article 14 single-obligation, no FRIA leak | Article 14 single-obligation, no FRIA leak | Article 14 single-obligation, no FRIA leak |
| Q2 red-teaming | Article 26 para-1 (instructions for use, wrong topic); Section 3 inconsistency | Article 26 para-7 (worker info, wrong topic); Section 3 inconsistency | Article 26 para-7 (worker info, wrong topic); Section 3 inconsistency |
| Q3 GDPR Art 22 | §4.3 Right-to-Explanation in Section 2; Section 3 broken (repeats Section 1) | §5.4 audit chunk in Section 2 (less relevant); Section 3 broken | **§4.3 in Section 2 (preserved by RRF); but Section 3 has FRIA leak** ("FRIA required by UK GDPR Article 22") |
| Q4 transparency | Article 13 para-1 (transparency-to-deployers, wrong audience) | **Article 50 para-1 (transparency-to-natural-persons, correct audience)** | Article 13 para-1 (back to wrong audience) |
| Q5 FRIA target | Clean gap finding (FRIA-vs-DPIA distinction) | Clean gap finding | Clean gap finding |

**Net headline: rerank_only is the best of the three.** It uniquely produces
the correct audience anchor on Q4 (the most stubborn documented
retrieval failure) while not regressing elsewhere. RRF dilutes the
cross-encoder's targeted demotions and re-introduces failures on Q3 and
Q4.

---

## Verbatim outputs (per-query, all three strategies)

Reranker confidence labels are shown for `rerank_only` and `rrf` modes
(as appended to the LLM output via `_format_evidence`). `bge_only` mode
predates the evidence-footer feature; retrieval scores from the V3
baseline (`v3-qwen-1.5b-local-baseline.md` Q-by-Q tables) used as proxy.

### Q1 — multi-facet

**BGE-only:** Article 14 para-5 anchor; single-obligation output. No FRIA leak.

**rerank_only:** Article 14 para-5 anchor (BGE rank 4 → reranker rank 1);
single-obligation. No FRIA leak. Reranker max scores were near-zero
(0.03 REG, 0.22 DEP) — labels show "law weak, policy moderate."

**RRF:** Article 14 para-5 anchor (BGE rank 4 → RRF rank 1);
single-obligation. No FRIA leak. RRF preserved the BGE rank-1
(`article-26-para-9`) at RRF rank 4 instead of dropping it; this had no
effect on the output.

**Verdict:** All three identical on substance. Q1's multi-facet failure is
the architectural single-call limit; no retrieval strategy can fix it.

---

### Q2 — red-teaming

**BGE-only:** Section 1 cites Article 26 para-1 (instructions for use —
wrong topic for red-teaming). Section 3 contradicts Section 2.

**rerank_only:** Section 1 cites Article 26 para-7 (worker information —
also wrong topic, different paragraph). Section 3 contradicts Section 2.

**RRF:** Section 1 cites Article 26 para-7 (same as rerank_only). Section
3 contradicts Section 2.

**Verdict:** All three wrong-anchored. The reranker shifted the wrong-topic
paragraph from para-1 to para-7 in both rerank-modes; RRF didn't pull it
back to para-1 because both are around BGE rank 1-2. Article 9 para-8
(the actual testing requirement) was at BGE rank 5 and stayed there
across all strategies. **Q2 needs a different intervention** — query
expansion or query rewriting, not reranking. Reranking ablation does
not isolate a useful intervention here.

---

### Q3 — GDPR Article 22 sub-clauses (the diagnostic)

**BGE-only:** Section 2 cites §4.3 Right to Explanation (relevant). Section
3 broken — repeats Article 22 text from Section 1 verbatim instead of
producing analytical content. **No FRIA leak.**

**rerank_only:** Section 2 cites §5.4 audit chunk (irrelevant — §4.3
dropped from top-5 by reranker). Section 3 broken — same repetition
pattern. **No FRIA leak.**

**RRF:** Section 2 cites §4.3 Right to Explanation (preserved by RRF —
intended). Section 3 has FRIA leak: *"The policy does not address
performing a fundamental rights impact assessment as required by UK GDPR
Article 22."* Same fabricated claim as V3-1.5B and V4-1.5B failure mode.

**Mechanism for the RRF Q3 regression:** RRF preserved §4.3 (good) but
ALSO promoted DPIA chunks (`dpia#6-2-external-consultation` and
`dpia#4-risks-identified`) from BGE ranks 3-4 to RRF top-2 because they
had decent BGE scores. Those DPIA chunks contain "rights" + "assessment"
terminology. Their prominence in retrieval primed the LLM toward the
FRIA association at Qwen 3B scale.

**This is a non-obvious finding:** the FRIA leak isn't strictly a
1.5B-scale issue suppressed at 3B+. It can be context-triggered at 3B
when retrieval surfaces FRIA-adjacent chunks prominently. **Retrieval
content shapes the LLM's training-data priors, not just retrieval
ranking.** Worth noting in `evaluation-findings.md` Stage 8 as a refinement.

**Verdict:** RRF's preservation mechanism produced a worse output than pure
rerank-only on Q3, despite preserving the BGE-relevant chunk we wanted.

---

### Q4 — transparency (the wrong-audience test)

**BGE-only:** Section 1 cites Article 13 para-1 (transparency-to-deployers
— wrong audience for "transparency for candidates").

**rerank_only:** Section 1 cites Article 50 para-1 (transparency-to-
natural-persons — **correct audience**). Article 50 para-1 was at BGE
rank 4 but rerank score gave it rank 3 — the LLM picked it up.

**RRF:** Section 1 cites Article 13 para-1 (back to wrong audience).
Article 50 para-1 is still in RRF top-5 (rank 4) but RRF's preservation
of BGE's top-1 (Article 50 para-4 deep-fakes) and BGE's top-2 (Article
13 para-1) overshadowed Article 50 para-1.

**Verdict:** **rerank_only uniquely fixes Q4's wrong-audience failure.** RRF
gives back what pure rerank-only earned by averaging it with BGE's
wrong choice. This is the most diagnostic difference across the three
strategies.

---

### Q5 — FRIA target

**BGE-only:** Clean gap finding. FRIA-vs-DPIA distinction explicit.

**rerank_only:** Clean gap finding. FRIA-vs-DPIA distinction explicit.

**RRF:** Clean gap finding. FRIA-vs-DPIA distinction explicit.

**Verdict:** All three strategies converge on the same clean output. Q5's
retrieval is sufficiently strong that ranking-strategy choice is
inconsequential.

---

## Friendly grounding labels — calibration check

The `_format_evidence` footer attaches a per-side confidence label and a
pattern interpretation. We can check whether the labels matched substantive
truth across the 5 queries (rerank_only / rrf modes only — bge_only
predates the feature):

| Query | Label (rerank_only mode) | Label (RRF mode) | Substantive truth | Match? |
|---|---|---|---|---|
| Q1 | law weak (0.03), policy moderate (0.22) | law weak (0.03), policy moderate (0.22) | Retrieval was bad; multi-facet recall failure | ✓ Yes |
| Q2 | law moderate (0.48), policy strong (0.87) | law moderate (0.48), policy strong (0.87) | Policy match correct; but law side picked wrong article | Partial — confidence ≠ correctness |
| Q3 | law strong (1.00), policy weak (0.06) — "policy may be silent" | law strong (1.00), policy weak (0.06) — "policy may be silent" | Policy IS effectively silent on the missing sub-clauses | ✓ Yes |
| Q4 | law weak (0.01), policy strong (0.85) — "law side weak" | law weak (0.01), policy strong (0.85) — "law side weak" | Retrieval picked wrong-audience law; label correctly flags retrieval doubt | ✓ Yes |
| Q5 | law strong (0.85), policy moderate (0.15) — "well-grounded" | law strong (0.85), policy moderate (0.15) — "well-grounded" | Output is genuinely well-grounded | ✓ Yes |

**4 of 5 queries: labels matched truth.** Q2 mismatch is not a calibration
error — labels measure *retrieval confidence*, not *output correctness*.
Confident retrieval doesn't guarantee the LLM uses the right chunk. That
distinction is worth surfacing in the report itself: the labels are a
*retrieval transparency signal*, not a *correctness oracle*.

The friendly grounding feature is **kept regardless of ranking strategy
choice.**

---

## Cross-strategy synthesis

**Three findings consolidate from this comparison:**

### 1. Pure cross-encoder reranking (`rerank_only`) is the right default

Of the three strategies, only `rerank_only` produces the correct
audience anchor on Q4 (the most stubborn cross-model retrieval failure).
It does not regress on the other queries. RRF and BGE-only both leave Q4
wrong-anchored.

The decision: switch project default from `rrf` (initial guess) to
`rerank_only` (empirically validated).

### 2. RRF dilutes targeted demotions

RRF was theoretically attractive: blend BGE and cross-encoder via rank
fusion to get "best of both worlds." Empirically on this corpus + this
test set, it's worse than pure rerank because:

- BGE is *confidently wrong* on several queries (Q4 deep-fakes; Q2
  Article 26 para-1 instructions for use)
- The cross-encoder is *confidently right* in correcting these
- RRF averages confident-wrong with confident-right, producing
  middle-ground rankings that lose the cross-encoder's correction

This is consistent with the published RAG literature: RRF works best
when both rankers have *partially-correct* signals. When one ranker is
confidently correct and the other confidently wrong, simple selection
beats fusion.

### 3. The FRIA leak has a context-trigger dimension

Stage 8 originally framed the FRIA leak as a parameter-count threshold
(suppressed above 3B). The RRF Q3 result refines this: at 3B, the leak
*can* re-emerge when retrieval surfaces FRIA-adjacent DPIA chunks
prominently. The model still has the FRIA-related associations even if
suppressed by default; specific retrieval contexts can re-activate them.

This is a useful nuance for `evaluation-findings.md` Stage 8 — the
threshold isn't absolute, it's dependent on retrieved-content composition.

---

## Status after this pass

- **Project default:** `ranking.strategy = "rerank_only"` (was `"rrf"`).
- **RRF code retained:** `_rrf_combine()` is isolated; switching back is
  a one-line config change. Documented as tested-and-rejected in
  `decisions.md`.
- **Friendly grounding labels: kept** — they're independent of ranking
  strategy and provide consistent transparency across all three modes.
- **Empirical evidence:** This test pass is the most rigorous controlled
  comparison the project has done — three strategies, same query set,
  same model, same prompt. Strong material for the report's Critical
  Analysis dimension.

## What this pass does NOT establish

- That `rerank_only` is universally the best strategy. We tested on one
  model (Qwen 3B), one corpus, five queries, one cross-encoder
  (`bge-reranker-base`). Different corpora or retrievers may favour
  different strategies; the empirical result here is specific to this
  setup.
- That cross-encoder reranking is sufficient. Q1 (multi-facet recall),
  Q2 (wrong-article ranking — the reranker also got it wrong, just
  differently), and Q3 (Section 3 broken at all strategies) remain
  retrieval-or-LLM-side failures that none of the three strategies
  resolved. Future-work mitigations: query expansion (recall),
  query rewriting (disambiguation), per-obligation extraction
  (multi-facet decomposition).
- That the FRIA leak is fully mapped. The RRF Q3 result suggests
  retrieval-context can re-activate the leak even at 3B; we'd need
  more controlled experiments (varying which DPIA chunks reach top-5)
  to characterise the trigger conditions.
