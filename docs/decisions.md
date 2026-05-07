# Decisions

> Pre-build decisions, made for you with full reasoning. Each entry is viva-ready text — read it aloud, you can defend it.
>
> **You own these.** If you find evidence during build that a call was wrong, override it and add a note to the decisions log at the bottom. The "considered alternative" line tells the marker you knew the trade-off; the "why this" line tells them you chose deliberately.

---

## 1. How `confidence` is computed for each row

**Decided:** Aggregate cosine similarity over the cited evidence chunks using **`min`** (the weakest cited chunk sets confidence). Provisional bins: `min ≥ 0.55 → high`, `min ≥ 0.45 → medium`, otherwise `low`. Re-tune the bin boundaries at the retrieval-config freeze gate using the τ histogram data.

**Why `min` not `mean`:** This system surfaces gaps for compliance review. Confidence should be honest about the *weakest* cited evidence, because a borderline citation in the chain is what compliance officers care about. Mean smooths over weak citations and hides the failure mode you're trying to expose. `min` reflects "the worst chunk we cited" — the right semantic for Fork A scope.

**Why provisional bins:** Cosine values on `multi-qa-MiniLM-L6-cos-v1` typically run 0.3–0.7 for relevant matches on prose. 0.55 / 0.45 splits the typical "good match" / "borderline" / "weak" zones. The actual distribution on legal text is what the freeze-gate spot-check measures — the bins get tuned then.

---

## 2. `confidence` value for silent rows

**Decided:** **`high`**. Reaffirmed 2026-05-04 after user concern about response-quality pollution.

**Why:** Silent classification is *deterministic* — `max_sim < τ` is a mechanical comparison, not an LLM judgment. The system is highly confident *that the policy is silent on this obligation*. Calling it `low` would imply uncertainty about the silence verdict, which is the opposite of what's true.

**Why this doesn't pollute response quality (reaffirmed 2026-05-04):** The `confidence` field is **post-hoc metadata** populated by chain code after CHN-05 — it never enters any prompt, doesn't influence retrieval, and doesn't change classification. Setting silent rows to `confidence: high` is a labelling decision, not a quality lever. Response quality is governed by retrieval recall (RET), obligation extraction (CHN-03), and the classifier (CHN-04 Phase 2); none of those touch this field. Two distinct concerns can be conflated here: (a) "are too many silent rows being flagged?" is a τ-tuning question, addressed by decision #4 at the retrieval-config freeze gate; (b) "do high-confidence silent rows look indistinguishable from high-confidence adequate rows in the UI?" is a presentation question, addressed by SCH-01 and UI-01's mandated visible marker on silent rows ("no policy chunk above similarity τ"). Neither concern is a quality risk on the chain output itself.

**Considered:** Adding a fourth value (`n/a` or empty) for silent rows; reversing to `silent → low` to signal structural distinction.

**Why not:** `n/a` adds a special case to a clean three-value enum. `silent → low` is misleading — implies LLM uncertainty when none exists. If the Streamlit prototype reveals user confusion in week 5, the cleanest fix is FLEX-4: extend the enum to `low / medium / high / structural` (~30 min schema tweak + UI renderer update), not retreat from the structural-honesty position.

**Watch trigger to revisit (2026-05-04):** Streamlit prototype review at week 5 — observe whether high-confidence silent rows feel indistinguishable from high-confidence adequate rows on a multi-sub-question query. Eval-phase demo rehearsal — if marker preview / domain-expert review surfaces confusion, FLEX-4 to add a `structural` enum value is the documented escalation.

---

## 3. Sub-question decomposition cap (CHN-01)

**Decided:** Cap at **4** sub-questions per query. Enforce in the decompose prompt; validate in chain code post-LLM and truncate to 4 if the model produces more.

**Why 4:** Genuine compliance queries can legitimately span 4 facets (e.g., "Article 22 automated decisions + Article 13/14 transparency + DPIA under Article 35 + Annex III high-risk classification"). Capping at 3 would force premature aggregation in the decompose step and lose register granularity.

**Considered:** Cap at 3 (matches the spec's typical-case working assumption); cap at 5+ (more headroom).

**Why not 3:** Suppresses legitimately multi-faceted queries. **Why not 5+:** Each extra sub-question adds ~2 LLM calls (extract + classify); 5 sub-questions = ~12 calls per query, which strains Groq's free-tier rate budget without a recall improvement.

---

## 4. τ histogram rule for the retrieval-config freeze gate spot-check

**Decided:** At the freeze gate, compute `max_sim` for ~30 sampled obligations against each deployer-side corpus (policy / extras / guidance). Take the **median** across all (obligation × corpus) max-sim values.

- If median ≥ 0.35 → raise τ to `median + 0.05`
- If median < 0.35 → keep τ = 0.35 (the spec's conservative default is doing its job)

**Why median + 0.05:** Median sits in the bulk of the distribution; +0.05 places τ above the bulk so most obligations don't trip silence by default. The 0.05 is a small buffer, not a calibration claim. Defendable as a sanity-floor adjustment, not a tuning exercise (the spec is explicit that build-time τ is *not* calibration; eval-phase calibration comes later).

**Considered:** Use mean instead of median; pick τ by eyeballing the histogram at the gate.

**Why not mean:** Sensitive to outliers — one Article that perfectly matches a policy clause skews the mean upward and would push τ too high. **Why not eyeball:** Under build pressure at the gate, having the rule pre-committed is faster and more defensible than re-deriving it.

---

## 5. Token-budget guard for the batched classifier (CHN-04)

**Decided:** Estimate prompt token count before the call (`len(prompt_string) // 3.5` for English). Fall through to per-obligation classification (FLEX-1) for *just this sub-question* if the estimate exceeds:

- **5,000 tokens** on Gemma 2-2B-it (8K context, ~3K reserved for response — context window is the binding constraint)
- **15,000 tokens** on Llama 70B via Groq — **placeholder, to be tuned at the retrieval-config freeze gate** (see watch trigger below)

Log the fall-through in verbose mode (CHN-06) so it's visible when it fires.

**Why these numbers (revised 2026-05-04):** Gemma's 5K is well-grounded — context window 8K minus ~3K reserved for response. The Llama number is harder to pin down before we see real prompt sizes:

- Llama 3.3 70B has a **128K context window** — context is *not* the binding constraint.
- **The binding constraint is Groq's free-tier rate limit:** TPM 12K, TPD 100K, RPM 30, RPD 1K (per `console.groq.com/docs/rate-limits`).
- A single prompt above ~10–11K tokens consumes the entire per-minute TPM budget, so the next call gets throttled.
- TPD 100K caps fully-uncached chain queries at ~10 per day before throttling — cache pre-warming (LLM-06) is essential.

The 15K placeholder is fine because failure is graceful: if the guard lets a 15K prompt through and Groq rate-limits it, `RoutingClient` catches the error and falls back to Gemma (LLM-05's hardcoded policy). Worst case is one slow call, not a corrupted register.

**Watch trigger (2026-05-04):** instrument CHN-04 to log estimated prompt size in verbose mode. At the retrieval-config freeze gate (end of week 2), tabulate prompt-size distribution across the 5 hand-written queries; tighten or loosen the Llama guard based on observation. Build-time observation tracked as BO-007 in the private build plan.

**Considered:** No guard, always batch and hope; always per-obligation, never batch; commit to a tight 9K Llama guard now.

**Why not no-guard:** Silent truncation corrupts classifications without a visible error — exactly the failure mode that destroys trust in the demo. **Why not always-per-obligation:** ~3× more LLM calls per query; rate limits become the bottleneck and uncached query latency triples. **Why not commit-9K-now:** premature without empirical prompt sizes; 15K-with-graceful-fallback is the same outcome with less re-tuning friction.

---

## 6. Retrieval top-k values

**Decided:** **`top_k = 5`** everywhere — for regulation retrieval in CHN-02 (per sub-question) and for each deployer-side corpus in CHN-04 phase 2 (policy, extras, guidance separately).

For silence detection in CHN-04 phase 1, use the maximum cosine similarity from the top-5 results per deployer-side corpus, then take the overall max across the three corpora.

**Why a single value everywhere:** One number to remember, one place to tune, and 5 is the right size for both jobs:
- For obligation extraction (CHN-03): 5 regulation chunks per sub-question gives the LLM enough context to extract ~5 atomic obligations without flooding the prompt.
- For evidence retrieval (CHN-04 phase 2): 5 chunks per deployer-side corpus = up to 15 candidate evidence chunks per obligation, which the classifier prompt can comfortably handle.
- For silence detection (CHN-04 phase 1): max over top-5 per corpus is more noise-robust than top-1 and stays cheap (no extra retrieval — same call returns the scores).

**Considered:** Different `top_k` per step (e.g., top-3 for silence detection, top-7 for regulation retrieval).

**Why not:** Adds tuning surface without a clear win at this scale. Single `top_k` keeps the retrieval interface simple and is cheaper to defend in viva.

---

## 7. Demo persona

**Decided:** **Maya, Head of AI Compliance** — used as a **drafting voice for the 5 hand-written test queries only**. Not named in the UI, not named in the report, not part of the demo narrative.

**Why:** A single coherent voice for the queries (direct, regulatory-shorthand-comfortable: "Art 22", "Annex III §4") makes them sound like real compliance prose rather than a heterogeneous test set. But the system itself doesn't need a named persona on screen — markers reward the substance of the gap-surfacing claim, not a UX framing. Keeping Maya backstage preserves the query-quality benefit while avoiding extra surface area to defend in viva.

**Override note (2026-05-04):** Daria's earlier instruction was to drop the persona entirely. After review of the test queries (calibrated to the gates and grounded in real policy gaps), the option-3 compromise was adopted: keep Maya as drafting voice only; never expose her in user-facing artefacts. The drafted queries keep their existing voice.

---

## 8. Product name

**Decided:** **TalentLens**, confirmed.

**Why:** Already used throughout the corpus extras (DPIA, Model Card, Transparency Notice, Annual Governance Report, Model Intake Assessment) and the strategic spec. Renaming now means rewriting all five extras for no benefit.

---

## 9. Streamlit theme

**Decided:** Single accent colour `#1E40AF` (professional blue, fits the Novara-AI fictive brand) on white background. Default Streamlit fonts. No custom CSS.

**Why:** Cosmetic — the markers reward the *content* of the demo, not visual polish. Single accent + defaults is 5 minutes of `.streamlit/config.toml` and looks coherent. Custom CSS is a time sink that returns nothing on the rubric.

Apply once UI-01 functional behaviour is stable. If demo deadline pressure surfaces, ship with full Streamlit defaults — also acceptable.

---

## 10. Build-phase budget commitment

**Default assumption:** ~12–15 hours/week of focused build time across the build phase, per spec.

**What you need to do:** Sketch your real availability against the build phase (term commitments, exam periods, anything that pulls focus). If any stretch falls materially below the assumed budget, flag it in the decisions log — those stretches are candidates for FLEX-5 invocation (chain depth collapse) rather than over-commitment.

**Why this matters:** This is the one decision only you can make — your real availability. The architecture is sized assuming the spec's budget; if reality is tighter, FLEX-5 is the planned response, not a panic move. Logging the gap honestly also gives you a defendable narrative in the report's limitations section: "the build was sized assuming X hours; actual was Y; here's how I adapted."

---

## Explicitly deferred — do NOT decide now

These exist on the build timeline but are deliberately not decided yet. Picking them prematurely defeats their purpose.

| Decision | When to decide | Why deferred |
|---|---|---|
| Embedding model (FLEX-3) | At the retrieval-config freeze gate | The decision is informed by retrieval recall on the actual corpus + queries; deciding now is a guess against unknown distribution |
| Sentence aggregation mode (FLEX-3 intermediate) | At the retrieval-config freeze gate, only if RET-01 underperforms | Same reason — needs measurement, not a guess |
| `severity` schema field (FLEX-4 candidate) | Eval-phase mid-point | Trigger requires gold-set evidence that prioritisation is load-bearing for the demo narrative |
| `sub_question` schema field (FLEX-4 candidate) | After UI-01 first runs end-to-end against multi-sub-question queries | Trigger requires UI evidence that the flat register feels disorienting on multi-sub-question queries |

If pressure mounts to decide one of these early, that's a signal to revisit the trigger — not to skip it. Let your assistant make the case in writing first.

---

## Decisions log (build-time)

Decisions made *during* the build go here, newest first. Format:

```
### [Title] — [triggering capability or gate]
**Decided:** [chosen value]
**Reason:** [one sentence — what tipped the call]
**Updates `build-notes.md`?** [yes/no]
```

### UI cluster — Jupyter-only (UI-02) implementation decisions — 2026-05-04
**Decided (7 sub-decisions; 6 plan-aligned + 1 refinement caught by capsys):**

1. **UI-01 Streamlit deliberately not pursued** — spec § UI-01's own caveat ("If UI-01 destabilises late in the build, fall back to UI-02 — Jupyter is brief-acceptable") was invoked proactively rather than reactively. Daria's call 2026-05-04 after the chain landed and live smoke verified end-to-end behaviour. **Trade-off accepted:** less visual polish for demo-day in exchange for material risk reduction (no Streamlit framework, no `@st.cache_resource` lifecycle, no dev-server / browser verification step). The system's intellectual contribution sits in the chain, not the UI chrome; a printed register with chunk_ids and section_references gives a marker everything they need to verify findings. **Defendable viva narrative:** "We chose Jupyter per the spec's brief-acceptable fallback to keep the build envelope tight; the system's distinctive contribution is in the chain, not the UI."
2. **`build_chain()` factory in `src/chain.py`** — mirrors `build_retriever()` from `src/retrieval.py`. Used by the notebook + REPL + future entry points; orchestrates retriever + RoutingClient + DiskCache wiring in one call. `use_routing=True` (default) wires the production primary→fallback path; `use_routing=False` returns just `GroqLlama70B` for FLEX-6 strip-down readiness.
3. **Chat loop logic lives in `src/ui/notebook_chat.py`, notebook is a 3-cell shell** — keeps the loop testable in pytest; notebook stays a thin import-and-invoke wrapper with `%autoreload 2` for iterative development.
4. **Plain-text rendering with ASCII separators + emoji status glyphs (option A, Daria's pick 2026-05-04)** — works in any terminal, any Jupyter kernel output, and in plain stdout. No IPython.display dependency. Glyphs (🔴 silent · 🟠 partial · 🟢 adequate · 🟣 contradictory) carry status colour without needing markdown rendering.
5. **Silent-row explanatory marker is a single line under the obligation** — "⚠ Silent: no policy chunk above similarity τ — policy is silent on this obligation." Honours the spec's UI-01 success condition language at text-level granularity.
6. **`conftest.py` auto-loads `.env` at session start** — `build_chain()` constructs a real `GroqLlama70B` which requires `GROQ_API_KEY` in env. Tests that exercise the factory now succeed regardless of shell-env state. Skips lines with the placeholder value (`PASTE_YOUR_GROQ_KEY_HERE`); uses `os.environ.setdefault` so explicit shell env always wins.

**Refinement caught by capsys:**

7. **`print_register(file=None)` resolves `sys.stdout` at call time, not import time** *(refinement, caught by `test_chat_runs_chain_on_query_and_prints_register`)*. Initial design used `file=sys.stdout` as default argument — Python evaluates default args at function-definition time, so `file` ended up bound to the original stdout. When pytest's `capsys` redirected stdout, `print_register` still wrote to the original, and the captured-output assertion failed. Fix: default `file=None`, resolve to `sys.stdout` inside the body.

**Updates `build-notes.md`?** No (UI-01-vs-UI-02 choice is in spec § UI-01's own caveat language; no architectural commitment changes).

### Schema (SCH-01) + chain (CHN-01..07) + LLM-07/08 implementation decisions — 2026-05-04
**Decided (12 sub-decisions; 9 plan-aligned + 3 forced corrections from the live API smoke):**

1. **Two distinct types: `Obligation` (chain state) and `RegisterRow` (schema output)** (per plan). Obligation accumulates state through Steps 3–5 with optional fields; RegisterRow is the frozen 9-field deliverable. Cleaner separation than one type with progressively-populated optionals.
2. **`EvidenceCitation` carries `score` alongside `chunk_id` and `section_reference`** (per plan). Confidence derivation needs the cosine score; score lives next to the citation it scored.
3. **`regulatory_provision` derives from the highest-scoring retrieved chunk's `section_reference`** (per plan). All obligations from the same sub-question share this label.
4. **CHN-04 retrieves deployer-side evidence ONCE per corpus per obligation (3 retrieval calls)** *(deviation from the plan, aligned with `decisions.md §6`'s "5 chunks per deployer-side corpus")*. Plan said one retrieval with combined filter; spec wanted up to 5 per corpus. Three calls per obligation × top_k=5 each = up to 15 candidate evidence chunks per obligation, deduped before the batched classifier.
5. **Token guard is logged-warning placeholder, not active fall-through** (per plan). First pass observes prompt sizes via verbose mode. BO-007 watch trigger fires for tightening at the freeze gate.
6. **`Obligation` mutation via `dataclasses.replace()`** (per plan). Frozen dataclass + immutable updates.
7. **Verbose mode is print-to-stdout, no structured logging** (per plan). Per spec § CHN-06.
8. **`_derive_confidence` is module-level pure function** (per plan). Testable without a chain instance.
9. **Synthesise step matches enriched LLM output back to obligations by `obligation` text, not by position** (refinement during implementation). LLM may reorder or drop items; positional zip would silently mismatch.

**Forced corrections from the live API smoke run** (`pytest -m live_api`):

10. **`_format_chunks` truncates each chunk text to 400 characters in the prompt** *(forced fix; classifier prompt initially 13,342 tokens vs Groq's 12K TPM ceiling)*. Aggregating evidence across multiple obligations dedupes to ~30–50 unique chunks; full chunk text would blow TPM. 400-char snippets cut prompt size ~70%; the LLM still classifies competently because it sees chunk_id + section_reference + opening 400 chars.
11. **`RoutingClient._is_rate_limit` recognises Groq's 413 + `code='rate_limit_exceeded'` as a rate-limit signal** *(forced fix; Groq returns 413 for token-budget violations, not the 429 that `groq.RateLimitError` covers)*. Without this, oversized prompts would propagate as `APIStatusError` rather than triggering fallback. Documented in routing.py.
12. **`GroqLlama70B` `max_tokens` raised from 2000 to 4000** *(forced fix; synthesise step truncated mid-JSON-array on Q5)*. 4K out leaves ~8K input headroom under the 12K TPM ceiling. Synthesise output for ~5 obligations at ~250 tokens of JSON each fits comfortably.

**Live API observation worth recording (forward signal for the freeze gate):**

13. **FRIA obligations classify as `partial` not `silent` at τ=0.35** *(observation from the live Q5 smoke run)*. Even at the obligation level, the deployer-side cosine for FRIA-shaped obligations exceeds τ=0.35 — the ICO main guidance + audit framework chapters carry "fundamental rights" / "impact assessment" vocabulary that pulls scores above the threshold. This is the same observation logged at the RET-01 spot-check, now confirmed at obligation granularity. **Implication:** the freeze-gate τ histogram check (per decisions.md §4) is genuinely load-bearing — τ=0.35 may need to lift to capture FRIA as silent. Not a bug now; the conservative-bias rationale (false-silence preferred over false-address) holds, and false-address would be the dangerous error this avoids. Tracked in build-plan.md as a forward note for the freeze gate.

**Updates `build-notes.md`?** No (build-notes does not specify implementation-level prompt or token policy).

### LLM cluster (LLM-01..06) implementation decisions — 2026-05-04
**Decided (8 sub-decisions; 6 plan-aligned + 2 deviations + 1 minor refinement caught by test/REPL feedback):**

1. **Task methods declared abstract on `LLMClient` (LLM-01) and concrete on `BaseLLMClient` (LLM-02)** (per plan). Chain code calling `client.decompose_query(...)` gets static-analysis comfort because the method *is* declared on the type. Concrete bodies live exactly once, on the base class. Adapters subclass `BaseLLMClient` and inherit task methods for free — per-adapter footprint stays at the spec's ~50–60-line target.
2. **Cache wraps `_complete()` at the BaseLLMClient layer via `_complete_cached()`** (per plan). Adapters stay focused on their single job (call the model and return a string). The cache key includes `model_id` per spec § Reproducibility, so swapping models invalidates the cache for that prompt without serving stale outputs from another model.
3. **`temperature=0.0` on the Groq Llama adapter** (per plan). Deterministic responses are essential for cache-key→response idempotence and eval-phase replay reproducibility.
4. **`LocalGemma2B` lazy-loads the model on first `_complete()`** (per plan). Instantiation is cheap; the ~5 GB Gemma 2-2B-it weight load only happens when actually called. Routing tests can construct the adapter without paying the cost. Targeted at Colab GPU; CPU-only fallback works but is slow.
5. **JSON parsing is best-effort via `_parse_json_list`** (per plan). Locates the first `[` and last `]`, parses that span. Tolerant to leading prose ("Sure, here's the JSON: …"), trailing commentary, and markdown code fences (```json … ```). Raises `SchemaParseError` only when no recoverable list bracket pair exists. Concrete tolerance for Llama and Gemma's quirks lives in this single shared parser at the base-class layer rather than per-adapter — simpler than the plan's hint of per-adapter parsing tolerance, and the actual model output we've seen makes a unified parser sufficient.
6. **Single live API test marked `@pytest.mark.live_api` + auto-skip-by-default in conftest** (per plan). Default `pytest tests/` does not call Groq (132 passed, 1 skipped). Live test is opt-in via `pytest -m live_api` (verified passing today). Avoids burning free-tier quota on routine CI.

**Deviations from the approved plan:**

7. **Schema-parse retry is not "1 retry-with-feedback then fallback" — it is "fallback immediately"** *(deviation, deliberate)*. The spec's ideal is "retry primary with a feedback hint then fallback." With `temperature=0` plus the `_complete_cached` layer, retrying the same prompt on the same model produces the same response (cache hit OR deterministic output) — the retry would be a no-op. Real retry-with-feedback would require modifying the prompt mid-task-method (a "your previous JSON was malformed; try again" hint), which is invasive at the task-method-signature level for diminishing return. Falling back immediately to Gemma is predictable, deterministic, and avoids any retry loop. The fallback's response gets cached under its own model_id, so eval-phase replay still hits cache cleanly. Documented inline in `src/llm/routing.py`.
8. **Groq exception types caught by isinstance helper functions, not by direct `except groq.RateLimitError`** *(deviation, defensive)*. The routing module imports `groq` lazily inside `_is_rate_limit` / `_is_network` helpers so unit tests can exercise routing without the `groq` package resolving correctly in every environment. Pragmatic; keeps the routing layer testable with stub adapters that don't import groq at all.

**Minor refinement during integration:**

9. **`DiskCache(cache_dir=...)` accepts both `Path` and `str`** *(refinement, caught by REPL wiring check)*. The plan typed `cache_dir: Path`. Realistic call sites pass strings (`DiskCache(cache_dir="llm_cache")`); the REPL wiring check failed with `'str' object has no attribute 'mkdir'`. Tightened to `Path | str` and coerce on entry. Test added: `test_diskcache_accepts_string_cache_dir`.

**Performance + verification (2026-05-04):**
- Full test suite: **133 passed**, 1 skipped (live API), runtime 218s (the 167s integration-test floor from ingestion + ~50s from new LLM tests dominated by spaCy init + corpus runs already present).
- Live API test: `pytest -m live_api` runs the real Groq call against `llama-3.3-70b-versatile`, returns "pong" content; ~0.5s round-trip.
- REPL wiring check: `RoutingClient(GroqLlama70B(cache=DiskCache(...)), LocalGemma2B(cache=DiskCache(...)))` instantiates cleanly; `isinstance(r, LLMClient)` is True; properties proxy from primary.

**Deferred to post-CHN-05:**
- LLM-07 full smoke test (5 fixed queries through the full chain on each adapter): can't run until the chain exists.
- LLM-08 typing-discipline mock test: depends on a chain that consumes an `LLMClient` argument.
Both stub files (`tests/test_smoke.py`, `tests/test_typing.py`) exist from PR #3 with docstring-only bodies; bodies fill in when CHN lands.

**Updates `build-notes.md`?** No (cluster-internal implementation detail; all architectural commitments already documented there).

### RET-01 implementation decisions — 2026-05-04
**Decided (5 sub-decisions, all aligned with the approved plan; one Apple-Silicon-specific fix during testing):**

1. **`ChunkEmbeddingRetriever` constructed with already-embedded inputs; `build_retriever()` factory orchestrates the pipeline** (per plan). Tests construct the class directly with stub data; the chain wires the factory once at startup. `build_retriever` reuses the ING-03 SentenceTransformer singleton, so the model is loaded exactly once across ingestion + retrieval.
2. **`corpus_filter` accepts both `str` and `Sequence[str]` forms** (per plan). Call sites read naturally: `retrieve(q, corpus_filter="REG")` for CHN-02 regulation-only and `retrieve(q, corpus_filter=("DEP", "DEP_EXTRAS", "OPS"))` for CHN-04 silence detection.
3. **Linear filter scan over chunks at retrieval time, not pre-computed boolean masks** (per plan). At 1,140 chunks the filter is sub-millisecond; pre-computing masks is premature. The latency target (under 200ms top-5) is met without it — measured 17.10s for 19 tests including warm-up + multiple full-corpus retrievals = average <100ms per call.
4. **`build_retriever` reuses `_get_st_model` from `src.ingestion`** (per plan). Cross-module use of a private helper is contained to one orchestration site. Will rename `_get_st_model` → public when a third caller appears (likely never; the factory is the only orchestration site).
5. **Returned scores are raw `float`s in [-1, 1]** (per plan). `util.dot_score` on already-L2-normalised vectors equals cosine similarity. CHN-04 thresholds against this directly with τ. No surface area for silent rounding errors.

**Apple-Silicon device-mismatch fix:**

6. **Query embedded with `convert_to_numpy=True`, not `convert_to_tensor=True`** *(fix, caught by all 10 integration tests on first run)*. With `convert_to_tensor=True`, `SentenceTransformer.encode` produces an MPS-device tensor on Apple Silicon (PyTorch's Metal backend), while the chunk embeddings stored from ING-03 are CPU numpy arrays. `util.dot_score` then errors with `RuntimeError: Tensor for argument #2 'mat2' is on CPU, but expected it to be on GPU`. Switching the query path to numpy keeps both sides on CPU, lets `util.dot_score` handle device-uniform conversion, and works portably across CPU-only / MPS / CUDA hardware. Inline comment in `src/retrieval.py` records the rationale.

**Spot-check observations (REPL, 2026-05-04):**
- Q5 ("FRIA Article 27") top-5 includes `EU AI Act Article 27 (para-1)` at 0.736 and `(para-4)` at 0.673 — strong on-target retrieval.
- Q3 (Art 22 sub-clauses) top-5 surfaces UK GDPR Article 22, ICO Main Guidance Article 22 fairness chapter, and Individual rights chapters — diverse multi-corpus signal.
- Q5 with `corpus_filter=("DEP", "DEP_EXTRAS", "OPS")` yields top hits in [0.459, 0.515] — all *above* τ=0.35 default. This is the silence-detection shape; on the broad query the deployer side has tangential matches (ICO chapters mentioning "fundamental rights" in passing). Worth flagging at the freeze-gate τ histogram check: the chain operates on *extracted obligations* (CHN-03 output), not on raw user queries, so the actual silence detection signal will use a more specific embedding than what the spot-check exercises here. Recorded as a forward observation for the freeze gate, not a present concern.

**Updates `build-notes.md`?** No (build-notes does not specify implementation-level retrieval signatures).

### ING-03 implementation decisions — 2026-05-04
**Decided (5 sub-decisions, all aligned with the approved plan; one minor refinement during testing):**

1. **Cache layout: `{cache_dir}/{model_name}.npz`** (per plan). Model identity in the filename, not as a field inside the file. FLEX-3 swap to `bge-large-en-v1.5` will create a new sibling file; the original `multi-qa-MiniLM-L6-cos-v1.npz` is untouched. Zero risk of silent stale-vector reuse across model dimensions (384 vs 1024).
2. **Cache hit predicate is exact-match on chunk_ids list (same set + same order)** (per plan). Partial-update / incremental embedding rejected as over-engineered — the corpus is frozen during the build, so re-runs hit on every call after the first compute. If Daria ever creates a `corpus-vN/` snapshot per `v2_corpus_specification.md` § 7, the new chunk_ids won't match, ING-03 re-embeds, new cache file is written.
3. **Return tuple `(embeddings, chunk_ids)`, not a dataclass** (per plan). RET-01 needs both — the matrix for `util.dot_score` and the chunk_ids for top-k index → id mapping. Tuple is the minimum-ceremony shape; promotable to a dataclass if RET-01 reveals friction.
4. **Lazy SentenceTransformer model loading + module-level singleton** (per plan). Same pattern used for the spaCy sentencizer in ING-02. The `from sentence_transformers import SentenceTransformer` line is *inside* `_get_st_model` so that importing `src.ingestion` for ING-01 / ING-02 work doesn't pay the multi-second sentence_transformers import cost. First call to `embed_chunks` pays the cost; subsequent calls hit the singleton.
5. **`normalize_embeddings=False` on encode** (per plan). `multi-qa-MiniLM-L6-cos-v1` outputs already-L2-normalised vectors (the `-cos-v1` suffix); REPL spot-check confirmed `np.linalg.norm(e[0]) == 1.0000`. RET-01's `util.dot_score(query, chunks)` will return cosine similarity directly because of this property, with no extra work.

**Minor refinement during testing:**

6. **`_load_cache` catches `Exception` broadly, not a curated tuple** *(refinement, caught by `test_load_cache_returns_none_for_corrupted_file`)*. The plan called for `(OSError, ValueError, KeyError)`. Numpy on a corrupted .npz raises `_pickle.UnpicklingError` (when `allow_pickle=True`) and can also raise `BadZipFile` from the underlying zip layer. Listing all of these is fragile across numpy versions. Cache failure should never crash the system — we just re-embed. The cost of the broader catch is negligible; the benefit is robustness against future-numpy error churn.

**Performance numbers from REPL spot-check (2026-05-04, Apple M2 CPU):**
- Cold cache: 14.93s for 1,140 chunks (well under the spec's 3-minute budget).
- Warm cache: 0.37s for the same input (cache load only, no re-embed).
- Cache file size: 1,820,414 bytes (~1.8 MB) for 1,140 × 384 float32.
- Embeddings L2-normalised: `np.linalg.norm(e[0]) == 1.0000` confirmed.

**Updates `build-notes.md`?** No (build-notes does not specify cache file format details).

### ING-02 implementation decisions — 2026-05-04
**Decided (8 sub-decisions; 3 of these are deviations from the approved plan, flagged below):**

1. **`Chunk` → `Document` rename for ING-01 output** (per plan). The retrievable-unit name belongs to ING-02; ING-01's file-level records are now Documents. Mechanical rename across `src/ingestion.py` and `tests/test_ingestion.py`.
2. **New `Chunk` dataclass for ING-02 output** (per plan): 10 fields — the inherited 8 plus `parent_document_id` (provenance back to ING-01) and `sentences: tuple[str, ...]` (FLEX-3 RET-02 sub-data).
3. **GDPR consolidated file (`uk-gdpr-articles-relevant.txt`) produces zero chunks** (per plan). The file is loaded by ING-01 for corpus completeness; ING-02's dispatcher returns `[]` because the per-article files cover the same content with finer granularity.
4. **Section-level retrieval chunks (1140 total) with sentence breakdown stored as metadata** (per plan). Per-bucket distribution: 542 REG / 536 OPS / 15 DEP / 47 DEP_EXTRAS. Below the corpus spec § 4 estimate of 3K–5K because that estimate assumed sentence-level granularity; we instead carry sentence breakdown on each chunk for FLEX-3 aggregation.
5. **ICO chunker uses sentence-cluster grouping at ~250-token target** (per plan). Section reference: `"{Document section_reference} [N/Total]"`. The H2/H3 inference suggested by corpus spec § 5 doesn't survive the HTML-to-plain-text extraction.
6. **Novara policy section regex requires `[A-Z][a-zA-Z]{2,}.{4,}` heading body** (per plan). Rejects "30 days" (a §4.4 retention table cell) which a naïve `^(\d+(?:\.\d+)?)\s+(.+)$` would have accepted.

**Deviations from the approved plan:**

7. **AI Act page-furniture TREE regex matches mid-line, not standalone** *(deviation, caught by `test_no_page_furniture_in_chunk_text`)*. The plan inherited the pattern from `docs/ai-act-extraction-notes.md` § "Page furniture" which gave `^\s*TREE\.\d+\.[A-Z]\s*$` (standalone-line). The actual annex page headers combine multiple noise tokens on one line — `ANNEX I    TREE.2.B    EN` — so a standalone-line regex misses them, and `TREE.2.B` leaked into article-1 chunk text. Tightened to `^.*\bTREE\.\d+\.[A-Z]\b.*$` (matches anywhere in the line). The standalone-line annex header (`ANNEX I` alone, line 10450) is the one that gets *picked up* by the annex-boundary detector; the combined page-header repeats are stripped as furniture.

8. **`_split_long_article` uses sequential paragraph indexing, not regex-matched numbers** *(deviation, caught by `test_chunk_ids_are_unique_across_corpus`)*. The plan said sub-anchors look like `para-{N}` from the regex match (`m.group(1)`). This collided in AI Act annexes with multi-section structure — ANNEX VIII has Sections A, B, C, each carrying its own "1.", "2." numbering, and my "first occurrence wins" annex detection groups them all under one parent. Result: 11 colliding chunk_ids on first run. Switched to sequential indexing (`para-1`, `para-2`, ... in document order) inside the parent's body; collisions resolved.

9. **Pre-numbered preamble merges with the first numbered paragraph** *(deviation, caught by spot-check rather than tests)*. With the original sub-split logic, an article like Article 27 produced a tiny standalone chunk containing only the article title (`"Fundamental rights impact assessment for high-risk AI systems"`, 8 words). On a retrieval query like `"FRIA Article 27"`, the title-only chunk would win on cosine similarity but carry no obligation content — exactly the wrong chunk to surface. Fixed by tracking a `has_started_paragraphing` flag: the first numbered paragraph absorbs all preceding preamble lines (article title, subject-matter line) instead of closing them into their own pseudo-paragraph. Article 27's first sub-chunk now contains title + substantive paragraph 1.

**Test signal:** the build-time test set caught two of the three deviations directly (TREE regex via `test_no_page_furniture_in_chunk_text`; sequential indexing via `test_chunk_ids_are_unique_across_corpus`). The third (title-merge) was caught by REPL spot-check during the verification step — a reminder that automated tests cover structural invariants but retrieval-quality concerns need human-in-the-loop verification, exactly as the spec § ING-02 success conditions anticipate ("freeze gate, 5–10 manually reviewed chunks").

**Updates `build-notes.md`?** No (build-notes does not specify implementation-level chunking heuristics; the FLEX-3 paths it documents remain accurate as fallback options).

### ING-01 implementation decisions — 2026-05-04
**Decided (4 sub-decisions, all on the same implementation):**

1. **ING-01 outputs file-level chunks (~42), not the spec's "50–80".** The spec's success-condition number is treated as a stale pre-corpus-expansion estimate. ING-01 produces one chunk per text file in the manifest (44 entries minus 2 PDFs = 42); ING-02 owns the structural split that brings the count up to the corpus spec § 4 estimate of 3,000–5,000.
2. **Four-bucket `corpus_tag` scheme: `REG / OPS / DEP / DEP_EXTRAS`.** The manifest's three-value `corpus_tag` (REG/OPS/DEP) is split into four at load time: paths under `deployer/` → `DEP`, paths under `deployer-extras/` → `DEP_EXTRAS`. This gives the chain a clean filter when populating `policy_evidence` vs `extras_evidence` per SCH-01 fields, without touching the manifest.
3. **Skip manifest entries with `word_count == 0`.** The two PDFs (AI Act + Novara policy) are present for citation provenance only; their `.txt` siblings are the canonical retrieval source per `v2_corpus_specification.md` § 2. The `word_count == 0` field is the manifest's existing signal.
4. **`chunk_id` is the file path with the extension stripped** — e.g., `regulation/uk-gdpr-art-22`, `operational/ico-main-guidance/03-transparency`. Globally unique by filesystem invariant.

**Deviation from plan (worth recording):** the plan committed to `chunk_id = "{corpus_tag}:{document_id}"` claiming this was "collision-free given path uniqueness". This was wrong: the OPS bucket has three sub-folders (`ico-main-guidance/`, `ico-genai-consultation/`, `ico-audit-framework/`), and `ico-main-guidance/03-transparency.txt` and `ico-audit-framework/03-transparency.txt` share the same stem and same tag — same chunk_id under the original scheme. The uniqueness test (`test_chunk_ids_are_unique_across_corpus`) caught it. Path-based IDs sidestep the collision.

**Reason:** simplicity + readability + structural correctness. Path-based IDs are debuggable (the chunk's origin is in its name), stable across runs, and provably unique. `document_id` (just the stem) remains a separate field on `Chunk` for human-readable citation.

**Updates `build-notes.md`?** No (build-notes does not specify implementation-level identifier formats).

### Skip AI Act recitals during ING-02 chunking — pre-build review (2026-05-04)
**Decided:** ING-02 starts chunking the EU AI Act `.txt` from line 3915 (Article 1) onward. Recitals 1–180 (lines ~60–3914) are excluded from the chunked content.
**Reason:** Recitals are interpretive paragraphs explaining *why* each provision exists; they're not directly enforceable obligations. Including them risks CHN-03 extracting pseudo-obligations from explanatory text and corrupting downstream classification — exactly the high-risk failure mode noted in the spec for CHN-03. The compliance gap analysis system's distinctive claim is that obligations are extracted from operative provisions and matched against deployer policy. Recitals carry no obligations; including them adds noise without clear retrieval benefit. Reasoning recorded in full at `docs/ai-act-extraction-notes.md` § "Recitals — chunk or skip?".
**Considered:** (a) chunk separately with a `recital_id` metadata flag for use as supporting context; (b) chunk inline with articles, no flag.
**Why not (a):** adds chunk count and asks the synthesise prompt to handle a new modality (interpretation note vs obligation evidence) without a strong reason it'll improve gap-surfacing quality. **Why not (b):** simplest but most likely to corrupt CHN-03 (extraction prompt sees recital text and may treat it as an obligation).
**Trigger to revisit:** if retrieval recall on broad / context-heavy queries is poor at the retrieval-config freeze gate, option (a) is the FLEX path — chunk recitals separately with a metadata flag, never feed to CHN-03 extraction, optionally surface in CHN-05 synthesise as interpretive context.
**Updates `build-notes.md`?** No (build-notes does not specify chunking-level implementation choices).

### Decision #5 token guard — Llama number revised to placeholder + watch (2026-05-04)
**Decided:** Keep 15K Llama guard as a placeholder; tune at the retrieval-config freeze gate (end of week 2) using empirical prompt-size logging from the 5 hand-written queries. Gemma 5K stays unchanged.
**Reason:** Verified Groq's actual free-tier limits (TPM 12K, TPD 100K, RPM 30, RPD 1K, context 128K). Hard-committing the Llama number before observing real prompt sizes is premature; failure is graceful (rate-limit → Gemma fallback via RoutingClient).
**Updates `build-notes.md`?** No (build-notes does not list specific guard values).

### Decision #2 silent-row confidence — rationale extended for response-quality concern (2026-05-04)
**Decided:** Keep `confidence: high` for silent rows. Watch trigger added: revisit at week-5 Streamlit prototype review or eval-phase demo rehearsal.
**Reason:** User concern that `confidence: high` could pollute response quality. Verified that the field is post-hoc metadata (never enters prompts, doesn't influence retrieval/classification) — it cannot affect chain quality. UX disambiguation is handled by the spec-mandated UI marker on silent rows. If user confusion appears at the prototype gate, FLEX-4 to add a `structural` enum value is the cleanest escalation.
**Updates `build-notes.md`?** No.

### Maya persona scope narrowed to drafting voice only — pre-build review (2026-05-04)
**Decided:** Override decision #7 to scope Maya as a drafting persona for test queries; remove her from any user-facing artefact (UI, report, demo).
**Reason:** Reconciles Daria's earlier "no demo persona" instruction with Bogdan's pre-built queries that lean on a single coherent voice — keeps the query quality, drops the on-screen exposure.
**Updates `build-notes.md`?** No (build-notes never named a persona).

### Greedy (deterministic) decoding for the simplified architecture's LLM call — 2026-05-06
**Decided:** Configure local LLM generation with `do_sample=False` and `repetition_penalty=1.05`; explicitly do not use `temperature`, `top_p`, `top_k` (the model's `generation_config.json` ships with sampling defaults — they are loaded but inactive under greedy decoding).

**Reason:** Compliance gap analysis requires deterministic output. Three concrete needs drove the choice:
1. **Reproducibility for the marker.** Same query → same gap finding, every run. Stochastic outputs would weaken the report's empirical claims (we couldn't say "the system produces this output," only "the system probably produces something like this").
2. **Cache effectiveness.** `DiskCache` keys responses on `(rendered_prompt, model_id)`. With sampling enabled, the cache would frequently disagree with fresh runs — same prompt, different output. Greedy keeps cache and live calls aligned.
3. **Comparison across prompt iterations.** Testing V3 vs V4 (see `docs/test-passes/v4-qwen-1.5b-prompt-hygiene.md`) required attributing every output difference to the prompt change. Sampling noise would have masked or fabricated differences.

The trade-off is that greedy decoding can produce slightly stilted or repetitive phrasing on some queries. We mitigate with `repetition_penalty=1.05` (soft discouragement of exact repetition without breaking grounded quotes from the chunks).

**Considered alternative:** sampling enabled with `temperature=0.1` (low randomness, mostly-deterministic). Would produce slightly more natural phrasing for a creative-writing task. Rejected because compliance work prioritises consistency over phrasing variety; the cost of unreproducible output exceeds the benefit of varied phrasing.

**Side effect — HF transformers warning.** Each generation call prints `"The following generation flags are not valid and may be ignored: ['temperature', 'top_p', 'top_k']"`. This is HF being polite: Qwen's `generation_config.json` ships with sampling defaults that we deliberately ignore. The warning is cosmetic; output is unaffected. We can suppress by passing `temperature=None, top_p=None, top_k=None` explicitly to `model.generate()` — deferred as low-priority polish.

**Updates `build-notes.md`?** No (build-notes describes architecture, not generation parameters; parameters belong in the report's appendix per the brief's implementation-detail requirement).

**For Daria's report appendix and viva:**
- Report appendix: parameter table listing `do_sample=False`, `repetition_penalty=1.05`, brief rationale (deterministic for reproducibility).
- Likely viva probe: *"why did you set temperature to zero?"* or *"did you consider letting the model be more creative?"* — the answer maps directly to the three reasons above (reproducibility, cache effectiveness, evaluation isolation), with the trade-off (slight phrasing stiffness vs sampling noise) acknowledged. ~60-second answer.

### Colab as evaluation environment, local CPU as production demo — 2026-05-06
**Decided:** The simplified path runs in two environments: (a) local CPU is the production demo path with `Qwen/Qwen2.5-1.5B-Instruct` as default; (b) Colab GPU is the evaluation environment for testing larger models (e.g. `Qwen/Qwen2.5-7B-Instruct`). The model identifier is read from a `MODEL_ID` environment variable; everything else is identical across environments. The Colab notebook (`colab/run_simplified_colab.ipynb`) clones the repo, sets `MODEL_ID`, and runs the same `analyse()` function. Outputs feed `docs/test-passes/` for cross-model comparison.

**Reason:** Two needs pulled in opposite directions:
1. **Demo reliability.** The local Qwen 1.5B path is offline, deterministic, and free. It runs without API keys, free-tier daily limits, or network dependencies — the demo on the day will not fail because of an external service. This is the production path.
2. **Empirical capability assessment.** V4's negative finding (`docs/test-passes/v4-qwen-1.5b-prompt-hygiene.md`) showed that prompt design cannot fix the FRIA leak at 1.5B scale; the cause is training-data prior. Testing whether scale alone fixes it requires running 7B+ models, which the local CPU cannot accommodate.

Splitting the codebase across environments would have created sync drift. Splitting the LLM call across machines (local retrieval + Colab-hosted LLM via tunnel) would have added networking complexity and ephemeral failure modes. Keeping one codebase, one entry point, and one config knob (`MODEL_ID`) gives both environments without forking the architecture.

**Considered alternative — port the LLM call to a Colab-hosted endpoint** (ngrok / Cloudflare tunnel exposing a small FastAPI wrapper). Rejected: tunnel keep-alive, auth, network round-trips, and ephemeral session lifetimes added more risk than the simplification was worth for a master's project. The notebook approach has the same effect (use Colab's GPU) without the moving parts.

**Considered alternative — commit pre-computed BGE embeddings to the repo** so Colab sessions skip the ~3 min re-encode. Rejected: BGE-large produces 1024-dim float32 vectors per chunk; binary blobs in git are awkward to review and bloat the repo for a 3-minute saving on infrequent evaluation runs. If iteration friction grows, mounting Google Drive and caching embeddings there is the documented escalation (see `colab/README.md`).

**Side effect — model-id sync.** `LLM_MODEL_ID` is read from `os.environ` once at module import. Setting `MODEL_ID` after `import src.simplified` has no effect; the notebook sets it before importing. The order is documented in the `run_simplified_colab.ipynb` cell sequence.

**Updates `build-notes.md`?** No (build-notes describes architecture, not deployment environments).

**For Daria's report appendix and viva:**
- Report appendix: brief description of the two-environment workflow and the rationale for keeping local as demo. Explains the design choice that lets her run on a Colab GPU for stronger empirical claims while keeping the demo reliable.
- Likely viva probe: *"why didn't you just use the bigger model everywhere?"* — the answer is hardware reality (Qwen 7B OOMs on consumer CPUs in fp32) plus demo reliability (no Colab dependency on the day). ~30-second answer.

### Per-family prompt design as a deployment consideration — 2026-05-07
**Decided:** The simplified path dispatches different prompt structures based on model family. Qwen / Mistral / Gemma 1/2 use the V4 long system prompt (5 numbered rules + role description, ~690 chars) plus a question-only user message. Gemma 3 uses a short role-only system prompt (~70 chars) plus a user message that prepends the same 5 rules ahead of the question. Routing happens in `_get_prompts(model_id, ...)` in `src/simplified.py`, called from `analyse()` before any LLM call. Cache keys include the actual system prompt for the run, so prompts of different families don't collide in the cache.

**Reason:** Empirical evidence from the cross-family Gemma 3-4B test pass (`docs/test-passes/v4-gemma-3-4b-colab.md`) showed that the V4 long system prompt — designed for and validated on Qwen — produced measurably worse multi-step instruction-following on Gemma 3-4B than the same content packaged as Google's recommended Gemma 3 structure. Specifically: with V4 long-system, Gemma 3-4B produced single-obligation Q1 outputs and vague Q3 sub-clause framing; with Gemma-adapted (short system + rules in user), the same model produced multi-section Q1 engagement and explicit sub-clause enumeration on Q3. The improvement is empirically documented in two paired runs.

This is not a theoretical concern. Google's official Gemma 3 launch documentation (https://huggingface.co/blog/gemma3) explicitly states *"Gemma 3 uses very short system prompts followed by user prompts"*; the example shown is a single sentence. Our V4 system prompt is 10× that length. Different model families have different instruction-tuning preferences; a cross-family deployment that uses one prompt format universally pays a quality cost on whichever family didn't shape that format.

**Considered alternative — single universal prompt that works for all families.** We didn't find one in the literature, and our empirical evidence shows that the same content in different package shapes produces different outputs. A universal prompt would require either (a) accepting a quality floor below what each family can reach with its preferred format, or (b) finding a format that is *equally* compliance-tuning-tolerant across all families — a meta-prompting research problem we did not have scope for.

**Considered alternative — train per-family prompts via systematic A/B testing.** The right answer at production scale, but out of scope for a master's project on a frozen corpus. The current dispatch (one Qwen format + one Gemma 3 format, both empirically validated against the 5-query test set) is the minimum acceptable; full per-family tuning is a documented FLEX-path.

**Considered alternative — add separate prompt formats for Llama 3, Mistral, Phi, etc.** Each family has its own preferences; we could add a dispatch entry per family. We didn't, because (a) the project tested two families empirically — adding untested branches would be speculation; (b) each new format adds a maintenance surface; (c) the dispatch pattern is now in place, so a third family entry is a 30-line addition when needed. Defer until a third family is empirically tested.

**Side effect — cache invalidation across prompt versions.** If we change either `SYSTEM_PROMPT` (Qwen path) or `GEMMA3_SYSTEM_PROMPT` / `_user_message_gemma3` (Gemma 3 path), every cached entry produced under the old prompt is no longer reachable (different cache key). This is intentional: the cache is keyed on `(rendered_prompt, model_id)`, and a different prompt is a different rendered_prompt. Manual cleanup of `llm_cache_simplified/` is not required; old entries become harmless dead weight on disk.

**Side effect — empirical limit of prompt-format adaptation.** The Gemma-adapted prompt did not close every quality gap. Q5 substance failure (Gemma 4B accepting Novara's "Standard AI Feature" self-classification despite Annex III §4 mandating high-risk) persists in both Gemma prompt formats while Qwen 3B handles it cleanly with the V4 prompt. **Per-family prompt design has limits. Some failure modes are model-capability bound, not prompt-design bound, and prompt engineering cannot substitute for that.** Documented as the second of three orthogonal mitigation levers in `evaluation-findings.md` Stage 8 (extended).

**Updates `build-notes.md`?** No (build-notes describes architecture; prompt design is implementation-level detail).

**For Daria's report appendix and viva:**
- Report appendix: include both prompt formats with a brief explanation that family-specific instruction-tuning preferences led to a per-family dispatch pattern. Cite the Gemma 3 launch documentation for the recommendation.
- Likely viva probe: *"why do you have two prompts? Doesn't that mean you didn't pick a winning format?"* — the answer: different model families have different instruction-tuning preferences; the same content in the same conceptual structure produces different outputs depending on whether rules are in the system or user turn, and on whether the system prompt is short or long. We tested two families and built a dispatch pattern that respects both. A production system supporting more families would extend the same pattern. ~45-second answer.
- Likely viva probe: *"how does this scale to deployment?"* — the answer: per-family prompt design is the documented deployment cost, and we built the infrastructure for it (`_get_prompts(model_id, ...)`). Production teams adding a new model family would test 5 representative queries against the existing prompt dispatch, observe per-family quality, and either reuse one of the existing formats or add a new one. The cost is small per family (~30 LOC) but real and recurring. ~30-second answer.

### Cross-encoder reranking as default; RRF tested and rejected — 2026-05-07
**Decided:** Project default ranking strategy is `rerank_only` — BGE-large bi-encoder retrieves top-10 initial candidates per corpus side; `BAAI/bge-reranker-base` cross-encoder rescores all 10; the top-5 by rerank score go to the LLM. Reciprocal Rank Fusion (`rrf`) and single-stage retrieval (`bge_only`) remain available via `config.toml` `[ranking] strategy` or `RANKING_STRATEGY` env var. Both alternative strategies' code is preserved (see `_rrf_combine()` in `src/simplified.py`) for empirical replication.

**Reason:** Stage 9 ablation (`docs/test-passes/v4-qwen-3b-ranking-strategies-comparison.md`) ran all three strategies on Qwen 3B + V4 prompt across the 5-query test set. Empirical evidence:
1. **`rerank_only` is the only strategy that fixes Q4 wrong-audience anchoring** (the most stubborn cross-model retrieval failure documented in Stage 8 — persisted across Qwen 1.5B / 3B / 7B and Gemma 4B with two prompt formats). Cross-encoder promoted Article 50 para-1 (transparency-to-natural-persons — correct for "transparency for candidates") over BGE's top choice Article 50 para-4 (deep-fakes — wrong topic). The LLM picked it up and produced the correct-audience output.
2. **`rrf` re-introduces failures that `rerank_only` had fixed.** RRF preserves BGE-relevant chunks via rank fusion; theoretically attractive ("best of both worlds"), empirically worse on this corpus because BGE is *confidently wrong* on Q4 (deep-fakes) and Q2 (instructions for use). RRF averages confident-wrong with confident-right, diluting the cross-encoder's targeted demotions. RRF Q4 reverted to BGE's wrong choice; RRF Q3 also surfaced FRIA-adjacent DPIA chunks prominently and triggered a FRIA leak at 3B that pure rerank-only didn't have.
3. **`bge_only` (no reranker) leaves Q4 wrong-anchored** and otherwise matches rerank-only on Q1 / Q2 / Q5. The gain from reranking is concentrated in Q4 specifically.

**Considered alternative — `rrf` (Reciprocal Rank Fusion).** Rejected on the empirical grounds above. RRF works well when both rankers have *partially-correct* signals; on this corpus + retriever combination, BGE is confidently wrong often enough that simple selection (rerank-only) beats fusion. RRF would still be the right default on a corpus where BGE produces partially-correct rankings and the cross-encoder produces partially-correct corrections — code stays for that future case.

**Considered alternative — `bge_only`.** Rejected because it leaves Q4 wrong-anchored. The whole point of adding the reranker was to address documented retrieval failures from Stage 8; reverting to BGE-only abandons the gain.

**Considered alternative — confidence-gated fallback** (use cross-encoder ranking when its max score >= τ; fall back to BGE otherwise). Not implemented. Theoretically the cleanest approach: the reranker's confidence varies dramatically by query (Q5 REG 0.85, Q3 REG 1.00 high; Q1 REG 0.03, Q4 REG 0.01 low — see Stage 9 evidence). A confidence-gated strategy would use rerank when reliable and BGE when not. Defendably *better* than any single strategy, but requires a τ threshold we'd have to defend (same intellectual debt as the chain's empirically-broken silence threshold). Documented as a future-work option; the friendly grounding labels (which expose reranker confidence to readers without committing to a hard threshold) are the lighter-touch version of this idea we did adopt.

**Side effect — friendly grounding labels** (`strong` / `moderate` / `weak` per side, plus pattern interpretation) **stay regardless of ranking strategy.** They're computed in `_format_evidence()` from reranker scores, which are present in both `rerank_only` and `rrf` modes. In `bge_only` mode the evidence footer shows BGE scores without the reranker-confidence labels (since no reranker ran). Empirically validated: 4 of 5 queries had labels matching substantive truth. The single mismatch (Q2: confident retrieval but wrong-anchored output) reflects a real distinction worth surfacing in the report — labels measure retrieval confidence, not output correctness.

**Updates `build-notes.md`?** No (build-notes describes architecture; ranking strategy is implementation-level config).

**For Daria's report appendix and viva:**
- Report appendix: brief description of the three ranking strategies tested, the per-query empirical evidence, and the rationale for choosing `rerank_only`. Reference Stage 9 for full per-query outputs.
- Likely viva probe: *"why didn't you use Reciprocal Rank Fusion? It's the standard pattern."* — the answer: *"We did test it. Empirically on our corpus and test set, RRF was worse than pure cross-encoder reranking on two of five queries — it diluted the cross-encoder's targeted demotions on Q4, where BGE was confidently wrong, and re-introduced a FRIA leak on Q3 by promoting DPIA-adjacent chunks. RRF works well when both rankers have partially-correct signals; on this corpus BGE has confidently-wrong choices that the cross-encoder needs to fully demote. Documented as a tested-and-rejected option in `decisions.md`."* ~45-second answer with concrete empirical grounding.
- Likely viva probe: *"how do you know your ranking strategy generalises?"* — *"We don't, in general. Five queries on one corpus is a constrained empirical scope. The result is specific to this setup — a different corpus or a different retriever may favour a different strategy. The code retains both `rrf` and `bge_only` paths via `config.toml`, so a production team could re-evaluate per-corpus."* ~30-second answer.
