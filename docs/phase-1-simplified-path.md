# Phase 1: Simplified Demo Path

> Implementation plan for adding a simplified single-call architecture to the repo, alongside the existing chain. This becomes the demo path for the submission.
>
> Designed to be **reversible** — adds new code, doesn't remove existing code. If Phase 2 cleanup is skipped, this state alone is a defensible submission.

---

## Goal

Add a working simplified-architecture path to the codebase that:

- Uses BGE-large for retrieval (validated on Q5 in `docs/evaluation-findings.md` Stage 6)
- Excludes the ICO operational corpus at query time
- Performs single-call retrieve-and-compare (no chain decomposition, no obligation extraction, no threshold-grounded silence detection)
- Runs locally on Qwen 1.5B (or any HF model the demo machine supports)
- Produces correct gap finding on Q5 (the canary silence target)
- Coexists with the existing chain in `src/chain.py` without modifying chain code

When complete: Daria's demo runs Q5 through this new path; the existing chain stays untouched as the "initial architecture" reference for the report's findings.

## Rationale

The decision to *add* a simplified path alongside the chain rather than refactor the chain itself is grounded in three concerns documented across the evaluation findings:

**1. The chain is empirically the as-evaluated artifact.** `docs/evaluation-findings.md` documents the chain producing wrong-article output on Q5 (Article 26 obligations instead of Article 27) and on Q2 (Article 2 Scope obligations instead of Articles 9/15). These findings are referenced extensively in the report's Critical Analysis dimension. If the chain code is refactored — for BGE, for ICO removal, for any reason — those documented behaviours stop matching what the running code does. The findings become historical claims rather than reproducible empirical observations a marker can verify.

**2. The simplified path validates a different architectural commitment.** The chain commits to threshold-grounded silence detection and obligation-level matching. The simplified path commits to direct retrieve-and-compare. These are *different architectures*, not different configurations of the same architecture. Having both visible in the codebase demonstrates the architectural exploration the rubric's Critical Analysis and Justification of Choices dimensions explicitly reward.

**3. ICO exclusion at query time, not corpus removal.** The simplified path excludes ICO via `corpus_filter=("DEP", "DEP_EXTRAS")` — a configuration choice, not a corpus deletion. This preserves the analytical evidence (the four-corpus state that motivated the contamination finding) while still avoiding the contamination in production output. Removing ICO from the corpus entirely would erase the empirical state that supports the finding "ICO operational guidance pollutes silence detection." Markers reading the report would see the claim but not the artifact that supports it.

**4. Reversibility.** Adding a new module risks nothing in the existing 189-test green state. Refactoring the chain risks regressions across many test files and the integration paths. For a non-coding student depending on Claude Code to make changes, low-risk additions are far safer than invasive refactors.

The simplified path is the *production demo path* and the chain is the *as-evaluated baseline*. Both serve the report; both are visible to markers; the journey from one to the other is the analytical content.

## Scope

### In scope (Phase 1)

| Deliverable | Where | Effort |
|---|---|---|
| `src/simplified.py` — retrieve + single-call compare module | new file | ~80–120 lines |
| `src/ui/simple_chat.py` — UI entry point using the simplified module | new file | ~50–80 lines |
| BGE corpus embedding script + cache integration | extend ingestion or sit standalone | ~30–50 lines |
| `.gitignore` update for `embeddings_bge/` cache | one line | trivial |
| Smoke test for simplified path end-to-end on Q5 | new test in `tests/` | ~30–50 lines |
| Documentation: docstring at top of `simplified.py` linking to findings | inline | trivial |
| Update `docs/build-notes.md` repo-layout section to reference new files | edit | minor |

Total estimate: ~250–350 new lines of code + ~4 hours of focused Claude-Code-assisted work.

### Out of scope (deliberately)

| Item | Why deferred |
|---|---|
| Refactoring `src/chain.py` for BGE | Chain is the as-evaluated baseline — must continue to produce the documented behaviour |
| Removing ICO files from `corpus/` | Removes the empirical evidence that supports the contamination finding |
| Removing `LocalGemma2B` adapter and `routing.py` | Phase 2 territory; not blocking demo |
| Removing `streamlit_app.py` stub | Phase 2 territory; harmless until then |
| Updating chain's τ default | Chain stays at τ=0.35 as the documented baseline |
| Schema change to remove `guidance_evidence` field | Chain still uses it; FLEX-4 invocation deferred |
| Implementing query expansion (Category 1 mitigation for Q2) | Path A accepts Q2 as a documented limitation rather than a fixed defect |
| Implementing query decomposition (Category 2 mitigation for Q1) | Same — Path A accepts Q1 as documented limitation |

### Cleanup deferred to Phase 2

If Phase 1 lands cleanly and there's still time/appetite, Phase 2 can revisit:
- Light deletion of genuinely dead code (Streamlit stub, unused adapters)
- Documentation polish (README, top-level orientation)
- Spec amendment with phase-2 architectural decision

Phase 1 is sufficient as a submission state on its own. Phase 2 is polish.

## Architecture overview

```
User query
    ↓
[src/ui/simple_chat.py]  ← demo entry point
    ↓
[src/simplified.py]
    │
    ├── load BGE retriever (cached embedding)
    ├── retrieve top-5 from REG corpus
    ├── retrieve top-5 from DEP + DEP_EXTRAS corpora    ← ICO excluded here
    ├── construct V3-style prompt
    ├── call LLM (configured: Qwen 1.5B via transformers, or LocalGemma2B)
    └── return raw output
    ↓
Display (print or render)
```

Compare to existing chain (`src/chain.py`):
```
User query
    ↓
[src/chain.py via build_chain()]
    ├── decompose (CHN-01)
    ├── retrieve regulation (CHN-02)
    ├── extract obligations (CHN-03)
    ├── match per-obligation (CHN-04 phases 1+2 — uses ICO in silence detection)
    └── synthesise register (CHN-05)
    ↓
20-row register
```

Both modules coexist. Both runnable. Different architectures.

## Implementation steps

Each step has an explicit acceptance criterion. Execute in order; verify each before proceeding.

### Step 1 — Create `src/simplified.py`

Skeleton:
```python
"""Simplified compliance gap analysis path.

Single-call retrieve-and-compare architecture. Retrieves top-k chunks from
regulation and from deployer + extras corpora, passes them with the user
query to a single LLM call, returns the LLM's structured comparison.

Deliberately excludes ICO operational corpus at query time (see
docs/evaluation-findings.md Stage 5/6 for the contamination finding).

Uses BGE-large embedding model for retrieval (FLEX-3 escalation —
empirically validated on Q5).
"""
```

The module exposes one function: `analyse(query: str) -> str` returning the LLM's text output.

Internally:
- Lazy-loads a BGE-backed retriever (uses `build_retriever(model_name="BAAI/bge-large-en-v1.5", cache_dir=Path("embeddings_bge"))`)
- Encodes query with BGE prefix `"Represent this sentence for searching relevant passages: "`
- Retrieves top-5 from REG, top-5 from `("DEP", "DEP_EXTRAS")`
- Constructs the V3 system + user prompt (port from `/tmp/sim_v3.py`)
- Loads the configured LLM (default: Qwen 1.5B; configurable to `LocalGemma2B` for Colab)
- Calls the LLM with chat-template + repetition_penalty=1.05 + max_new_tokens=550
- Returns the generated text

**Acceptance:** running `analyse(Q5)` from a Python REPL produces a 3-section output with correct gap finding on Q5. Matches the output we observed in `/tmp/sim_v3_bge_q5.log`.

### Step 2 — Create `src/ui/simple_chat.py`

Demo entry point. Mirrors the existing `src/ui/notebook_chat.py` shape but routes queries through `simplified.analyse()` instead of `ComplianceGapChain.run()`.

Provides a simple loop: `chat()` reads queries from stdin, calls `analyse()`, prints output. Quit on `exit`/`quit`/EOF.

**Acceptance:** running `python -c "from src.ui.simple_chat import chat; chat()"` and pasting Q5 produces the expected output to stdout.

### Step 3 — Update `.gitignore`

Add the new BGE embedding cache directory:

```
embeddings_bge/
```

**Acceptance:** the BGE-cached corpus embeddings are not tracked by git.

### Step 4 — Add smoke test

`tests/test_simplified.py`:

```python
"""Smoke test for simplified architecture demo path.

Asserts: simplified.analyse(query) returns a non-empty string with the
expected three section headers. Does not assert correctness of substance
(that's the report's evaluation work, not a unit-test concern).
"""
```

Test: marked `@pytest.mark.live_api` (or local-model equivalent) since it requires the LLM. Asserts:
- Output contains `"What the law requires"` (header marker)
- Output contains `"What the policy says"` (header marker)
- Output contains `"Gap"` (header marker)
- Output is non-empty

Runtime: ~30–60 seconds depending on model.

**Acceptance:** test passes when run with `pytest tests/test_simplified.py -m live_api`.

### Step 5 — Documentation updates

Two small edits:

**a. Add docstring at top of `src/chain.py`** (without changing chain logic):

```python
"""Initial multi-step reasoning chain (CHN-01 through CHN-05).

This is the architecture documented in compliance-gap-analysis-spec.md.
It was evaluated empirically — see docs/evaluation-findings.md for
identified failure modes. The simplified production demo path is in
src/simplified.py; this chain is preserved as the as-evaluated reference
for the report's Critical Analysis findings.
"""
```

**b. Update `docs/build-notes.md`** repo-layout section to add `simplified.py` and `simple_chat.py` rows.

**Acceptance:** running `pytest tests/` still passes; both files have updated docstrings; build-notes mentions both architectures.

### Step 6 — Re-run all 5 test queries through the simplified path on Colab

Once Phase 1 is committed and merged, run a final empirical pass on Colab GPU (cheap compared to local CPU):

- Q1 — multi-facet (predicted: still wrong on simplified, per Stage 6 findings)
- Q2 — red-teaming (predicted: still wrong on simplified, per Stage 6 findings)
- Q3 — Article 22 sub-clauses (untested on simplified)
- Q4 — ambiguous transparency (untested on simplified)
- Q5 — FRIA silence (verified: correct gap finding)

Document outputs in the report appendix as the empirical evaluation set.

**Acceptance:** all 5 outputs saved. Q5's behavior matches `docs/evaluation-findings.md` Stage 6. Q3, Q4 outputs documented as new evidence.

## Test strategy

After each step:
1. Run `pytest tests/` — ensure existing 189-test green state is preserved
2. Spot-check the simplified path with Q5 — confirm output is correct
3. Spot-check the chain still runs (with cached LLM responses) — confirm chain is unaffected

If at any step the chain breaks, **stop**. Adding the simplified path should never affect the chain's behavior.

## Risks and mitigations

**Risk: BGE corpus embedding takes too long for first run.**
- On local CPU: ~10–20 minutes for 1140 chunks (one-time, then cached)
- On Colab GPU: ~2 minutes (one-time)
- Mitigation: pre-warm the embedding cache during Phase 1 setup so demo doesn't wait

**Risk: `LocalGemma2B` import errors block Qwen-only deployment.**
- The simplified module should import the LLM lazily, supporting either Qwen (locally) or Gemma (Colab) without forcing both
- Mitigation: configurable model path; default to Qwen 1.5B for local, switchable for Colab

**Risk: Test fixtures break.** Some tests may have hardcoded assumptions about retrieval scores under MiniLM.
- Mitigation: don't change MiniLM-using test fixtures; the chain stays on MiniLM. New tests for simplified path are independent.

**Risk: Submission packet size grows.** BGE corpus embedding cache is ~5–10 MB; manageable but not trivial.
- Mitigation: gitignore the cache; markers regenerate on first run if needed.

**Risk: Daria can't read the new code, can't debug if Claude Code makes errors.**
- Mitigation: smoke-test after every step; if smoke test fails, roll back the step rather than deepen the change

## What this unlocks

After Phase 1 completes:

**For the demo:** Q5 runs through the simplified path with BGE retrieval, produces correct gap finding. Backup video can be pre-recorded against this path.

**For the report:** the codebase contains both architectures with clear naming. The findings document references the chain's documented behaviours and the simplified path's validated improvement on Q5. Markers can reproduce both.

**For the viva defence:** Daria can articulate the journey ("we built complex per spec, evaluated, found three error categories, simulated mitigations, validated BGE on Q5, productionized as the simplified path") with reference to specific code modules. Each step has a verifiable artifact.

**For Phase 2 (optional):** if time permits, light cleanup can remove genuinely dead code (Streamlit stub, unused adapters/routing if FLEX-6 is invoked). Phase 2 is purely cosmetic; Phase 1 alone is a defensible submission state.

## What this does not do

- Fix Q1 (multi-facet failure persists; this is documented in findings)
- Fix Q2 (red-teaming output still has internal contradiction; documented)
- Improve Q3 / Q4 (untested on simplified; will be tested in Step 6)
- Remove the existing complex architecture (deliberately preserved as evaluated baseline)
- Address τ recalibration (chain stays at τ=0.35 as evaluated; simplified doesn't use τ)

These are *deliberate* scope choices, not omissions. Each is documented in `docs/evaluation-findings.md` with rationale.

---

**Estimated total effort:** 4–6 hours of Claude-Code-assisted work over 1 day, plus Step 6's Colab evaluation pass on a separate session. Daria's role is verification at each step, not coding.
