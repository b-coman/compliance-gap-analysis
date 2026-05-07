# Implementation Guide — Bringing the Simplified Path Into Your Repo

> **For Daria.** Step-by-step instructions for porting the simplified-path
> implementation into your own GitHub repository, using Claude Code as your
> implementer. You won't need to write Python yourself — Claude Code will do
> that. Your job is to drive it: read this guide, give Claude Code one
> step at a time, verify the result, and commit when you're confident.
>
> Bogdan reviewed and approved the implementation in his fork
> (`b-coman/compliance-gap-analysis@colab-refactor`). This guide tells you
> how to bring the same code into your repo (`dariacoman/compliance-gap-analysis`).
>
> **You do not need to repeat any of the empirical evaluation work** — the
> 5 test pass documents, the 9 stages of `evaluation-findings.md`, and the
> decisions records are already complete. You're just porting code + docs
> into your repo and verifying it runs locally.

---

## What you'll have at the end

After completing this guide, your repo will contain:

1. **`src/simplified.py`** — the production path: a single function
   `analyse(query)` that takes a compliance question and returns a gap
   analysis. Single LLM call, no chain.
2. **`src/ui/simple_chat.py`** — a small command-line chat loop you can
   run for the demo: ask a question, get an answer.
3. **`colab/`** — a notebook to run the same code on Colab GPU with
   bigger models for the empirical comparison.
4. **`config.toml`** — single source of truth for tunables (top-k, ranking
   strategy, generation parameters). Editable from a Colab cell at runtime.
5. **`docs/`** — the test passes, evaluation findings, decisions, and
   architecture diagrams that document the empirical work.
6. **A working local demo:** you can type `PYTHONPATH=$PWD python -m
   src.ui.simple_chat`, ask Q5 (the FRIA query), and get the
   demo-quality output.

---

## Before you start: the rules

These are the things Bogdan and Claude (in his session) decided based on
empirical evidence. Do **not** let your Claude Code re-litigate them.
If your Claude Code starts proposing changes to anything in this list,
say *"that decision is settled per the implementation guide; please
follow the existing implementation."*

### Settled decisions (don't change)

| Topic | Decision | Why |
|---|---|---|
| Default LLM (local) | `Qwen/Qwen2.5-1.5B-Instruct` | Demo-reliability default; offline, no API keys, fits your Mac CPU |
| Embedding model | `BAAI/bge-large-en-v1.5` | Stage 5 empirically validated vs MiniLM |
| Reranker | `BAAI/bge-reranker-base` | Stage 9 empirically validated; fixes Q4 wrong-audience |
| Default ranking strategy | `rerank_only` | Stage 9 ablation; RRF tested and rejected |
| Decoding | Greedy (`do_sample=false`, `repetition_penalty=1.05`) | `decisions.md`: deterministic for reproducibility |
| Prompt for Qwen | V4 long-system (5 numbered rules + role) | `decisions.md`; Stage 8 validated |
| Prompt for Gemma 3 | Short role-only system + rules in user turn | Stage 8 extended; per Google's Gemma 3 launch documentation |
| ICO operational corpus | Excluded at query time in the simplified path | Documented in `evaluation-findings.md` Stage 7 |
| `config.toml` location | Repo root | Single source of truth for tunables |
| Cache strategy | Greedy + DiskCache keyed on `(rendered_prompt, model_id)` | `decisions.md`: enables fast re-runs |

If any of these don't make sense to you and you want to understand the
rationale, read the corresponding section of `docs/decisions.md` or
`docs/evaluation-findings.md`. Both are self-contained. Bogdan can
also explain.

### Things that are open to evolve

You CAN adjust these without invalidating the empirical work:

- **Cosmetic UI changes** to `src/ui/simple_chat.py` (welcome message, prompts to user)
- **Adding new fields to `config.toml`** (e.g., for new features later)
- **Documentation tone or formatting** in your own additions
- **Adding new test queries** beyond the 5 in `docs/test-queries.md`

If unsure whether something is "settled" or "evolvable," ask Bogdan
before letting Claude Code change it.

---

## Reference implementation — where to read from

The validated implementation lives in Bogdan's fork:

- **URL:** https://github.com/b-coman/compliance-gap-analysis
- **Branch:** `colab-refactor`
- **HEAD commit at time of writing:** `d899c0a`

Your Claude Code can read files from this branch directly. The simplest
way: have it clone the branch into a sibling reference folder, then
copy/adapt files from there.

**Suggested first prompt to your Claude Code (verbatim, paste this):**

```
We're going to port code from a reference branch into this repo.

Reference: https://github.com/b-coman/compliance-gap-analysis branch
colab-refactor.

First, clone the reference branch into a sibling folder called
`reference-impl/` so we can read from it without merging:

  git clone --branch colab-refactor https://github.com/b-coman/compliance-gap-analysis.git reference-impl

Then read `reference-impl/docs/implementation-guide-for-daria.md` —
that's the guide we'll follow.

Confirm you've cloned and read the guide, then wait for me to give you
the first step.
```

---

## The implementation steps

Each step is a discrete unit of work. **Tell Claude Code one step at a
time.** After Claude Code finishes a step, verify it (the "How to verify"
line tells you what to look for), then commit before moving on.

### Step 1 — Add `config.toml` at the repo root

**What:** A single TOML file that holds all tunable defaults.

**Prompt:**
```
Read reference-impl/config.toml and create the same file at the root of
this repo. Don't change any values — these defaults are empirically
validated. Show me the file you're about to create before writing it.
```

**How to verify:**
- The file `config.toml` exists at the repo root
- Running `python3 -c "import tomllib; print(tomllib.load(open('config.toml','rb'))['ranking']['strategy'])"` prints `rerank_only`

**Commit message suggestion:**
```
Add config.toml — single source of truth for simplified-path tunables
```

---

### Step 2 — Update `requirements.txt`

**What:** Pin the libraries the simplified path needs (transformers, accelerate, diskcache).

**Prompt:**
```
Read reference-impl/requirements.txt. Compare to our current
requirements.txt and update ours to match. Show me the diff. Three
lines should be added: transformers, accelerate, diskcache.
```

**How to verify:**
- `requirements.txt` now contains `transformers==4.46.3`, `accelerate==1.1.1`, `diskcache==5.6.3`

**Commit message suggestion:**
```
requirements.txt: pin transformers/accelerate/diskcache for simplified path
```

---

### Step 3 — Add `src/simplified.py`

**What:** The core single-call architecture. About 760 lines.

**Prompt:**
```
Read reference-impl/src/simplified.py and copy it to src/simplified.py
in this repo, exactly as-is. This file is the production path —
it has been empirically validated through 9 stages of evaluation. Do
not rewrite or refactor any logic. Show me a summary of what it
contains (functions and what they do) before writing.
```

**How to verify:**
- `src/simplified.py` exists in your repo
- Running this passes:
  ```bash
  PYTHONPATH=$PWD python3 -c "from src.simplified import analyse, CONFIG; print('OK', CONFIG['ranking']['strategy'])"
  ```
  Output: `OK rerank_only`

**Commit message suggestion:**
```
Add src/simplified.py — Phase 1 single-call architecture
```

---

### Step 4 — Add `src/ui/simple_chat.py`

**What:** A small CLI chat loop. ~50 lines.

**Prompt:**
```
Read reference-impl/src/ui/simple_chat.py and copy it to
src/ui/simple_chat.py. Don't refactor.
```

**How to verify (do NOT run yet — Step 6 covers that):**
- File exists

**Commit message suggestion:**
```
Add src/ui/simple_chat.py — interactive CLI for the simplified path
```

---

### Step 5 — Update `.gitignore` and `src/chain.py` docstring

**What:** Two small changes to existing files.

**Prompt:**
```
Two file updates:
1. Read reference-impl/.gitignore and reference-impl/src/chain.py.
2. For .gitignore: add the entries that the reference has but ours
   doesn't (should be `embeddings_bge/` and `llm_cache_simplified/`).
3. For src/chain.py: copy the updated docstring at the top of the
   file (it explains that chain.py is the as-evaluated baseline and
   that the production path is now src/simplified.py). Don't change
   any chain code.
Show me both diffs.
```

**How to verify:**
- `.gitignore` contains `embeddings_bge/` and `llm_cache_simplified/`
- `src/chain.py` first comment block mentions "production demo path is the simplified single-call architecture in src/simplified.py"

**Commit message suggestion:**
```
.gitignore + chain.py docstring: add simplified-path artefacts and rationale
```

---

### Step 6 — Run the simplified path locally to verify it works

**What:** Run a single query to confirm everything is wired up. The first
run will download model weights (~3 GB total: BGE-large + reranker + Qwen 1.5B)
and embed the corpus (~30s on a Mac CPU). Subsequent runs are fast.

**Prompt:**
```
Run this command in the repo root and tell me what happens:

  PYTHONPATH=$PWD .venv/bin/python -c "from src.simplified import analyse; print(analyse('Have we performed a Fundamental Rights Impact Assessment under EU AI Act Article 27 for TalentLens as a deployer of an Annex III high-risk system?'))"

It will take 2-3 minutes the first time (model downloads + corpus
embedding). If anything fails, show me the full error and we'll
debug together.
```

**How to verify:**
- Output contains `### What the law requires`, `### What the policy says`, `### Gap`
- The Gap section mentions FRIA being missing from the policy
- A retrieval-evidence footer appears below the three sections

**If something fails, common issues:**
- Missing dependency → run `pip install -r requirements.txt` in the venv
- HuggingFace not authenticated → the BGE/reranker/Qwen models are public, no auth needed; if you see an auth error, check that you have internet
- Module import error → make sure `PYTHONPATH=$PWD` is set so `src.simplified` resolves

**Don't commit anything in this step.** This is verification only.

---

### Step 7 — Add the Colab notebook

**What:** A 16-cell notebook for running the simplified path on Colab GPU.

**Prompt:**
```
Copy reference-impl/colab/run_simplified_colab.ipynb and
reference-impl/colab/README.md to colab/ in our repo. The notebook
contains a clone command pointing at b-coman's fork — please update
that one cell to clone from `dariacoman/compliance-gap-analysis`
instead, on whatever branch I'm working on right now (check the
current branch with git). Show me the diff of the clone cell.
```

**How to verify:**
- `colab/run_simplified_colab.ipynb` exists
- `colab/README.md` exists
- The clone cell points at your repo (`dariacoman/...`), not Bogdan's fork

**Commit message suggestion:**
```
Add Colab notebook — evaluation environment with config-tweak cell
```

---

### Step 8 — Add the documentation

**What:** All the docs that capture the empirical work and design decisions.

**Prompt:**
```
Copy these files from reference-impl/docs/ to docs/ in our repo:

  - architecture-diagrams.html
  - architecture-diagrams-slides.html
  - decisions.md (replace ours if it exists)
  - evaluation-findings.md
  - phase-1-simplified-path.md
  - test-passes/ (entire folder)

These are evaluation records and design rationale documents. Don't
change any content; the empirical findings are settled. Show me a
summary of what got copied (file count, total size).
```

**How to verify:**
- `docs/decisions.md` exists with a section "## 1. How `confidence` is computed for each row"
- `docs/evaluation-findings.md` exists, contains "## The journey, summarised" and 9 stages
- `docs/test-passes/` contains 6 files (README + 5 test passes)
- `docs/architecture-diagrams.html` opens in a browser and shows two architecture diagrams

**Commit message suggestion:**
```
Add documentation — decisions, evaluation findings, test passes, diagrams
```

---

### Step 9 — Verify the local CLI demo end-to-end

**What:** The full demo flow Daria will use in the live presentation.

**Prompt:**
```
I want to verify the demo works. Run this command:

  PYTHONPATH=$PWD .venv/bin/python -m src.ui.simple_chat

It should prompt me for a question. I'll type the FRIA query and
read the output. If it works, we're done with the implementation.
```

When the chat prompts you, paste this query verbatim (it's the demo Q5):

```
Have we performed a Fundamental Rights Impact Assessment under EU AI Act Article 27 for TalentLens as a deployer of an Annex III high-risk system?
```

**How to verify:**
- Three sections: "What the law requires", "What the policy says", "Gap"
- The Gap section identifies that the policy doesn't address FRIA
- A retrieval-evidence footer appears with chunk IDs and scores

If yes — **the implementation is complete.**

---

### Step 10 — (Optional) Run the Colab notebook to capture the bigger-model evidence

This is optional. Skip if you're short on time; the empirical work in
`docs/test-passes/` already exists and you don't need to re-run it.

But if you want to:

**Prompt:**
```
Walk me through running the Colab notebook colab/run_simplified_colab.ipynb.
I'll need:
1. To open it in Colab (file → open notebook → GitHub → my repo URL)
2. To pick a model (Qwen 3B is the default — fast and clean)
3. To run all cells

Tell me what to do at each step. I'll do the clicks; you tell me
what to check at the end.
```

**How to verify:**
- All 5 queries produce output without errors
- The `colab_outputs.md` file is created in the Colab session
- You can download `colab_outputs.md` and compare it to one of the
  reference test passes in `docs/test-passes/`

---

## How to use Claude Code well

You don't need to know Python. You do need to know how to drive Claude
Code effectively.

### Things that work

1. **One step at a time.** Don't paste the whole guide; give Claude Code
   one numbered step from the list above.
2. **Ask "show me before doing."** Whenever Claude Code is about to
   create or modify a file, ask it to show you the change first.
3. **Verify each step.** The "How to verify" line in each step tells you
   what to check. If verification fails, tell Claude Code what you expected
   vs what you got.
4. **Commit between steps.** Each step ends with a commit message
   suggestion. Smaller commits = easier to revert if something goes wrong.
5. **Quote settled decisions back at it.** If Claude Code suggests
   changing one of the items in the "Settled decisions" table above, say:
   *"that decision is settled per the implementation guide; please follow
   the existing implementation."*

### Things to avoid

1. **Don't ask Claude Code to "improve" or "refactor" the code.** The
   reference implementation is already empirically validated. Refactors
   would risk breaking that validation.
2. **Don't let Claude Code change `config.toml` defaults** unless Bogdan
   asks you to.
3. **Don't combine multiple steps into one prompt.** Each step has its own
   verification; combining them makes failures harder to debug.
4. **Don't run code without understanding what it does.** If Claude Code
   suggests a command and you don't know what it does, ask: *"what does
   this command do, in plain English?"*

### When to ask Bogdan

- Anything in the "Settled decisions" table seems like it shouldn't be
  settled — verify with Bogdan before changing
- A step's verification doesn't pass and Claude Code can't fix it within
  3 attempts — ask Bogdan
- The architecture diagrams or any of the empirical documents seem wrong
  to you — ask Bogdan
- You're considering opening a PR or pushing to your `main` branch and
  want a review

---

## Acceptance checklist (when you're done)

After all 9 (or 10) steps complete, your repo should pass this checklist:

- [ ] `config.toml` exists at the repo root
- [ ] `src/simplified.py` exists and `from src.simplified import analyse` works
- [ ] `src/ui/simple_chat.py` exists
- [ ] `colab/run_simplified_colab.ipynb` exists and the clone cell points at your repo
- [ ] `colab/README.md` exists
- [ ] `requirements.txt` contains transformers, accelerate, diskcache
- [ ] `.gitignore` contains `embeddings_bge/` and `llm_cache_simplified/`
- [ ] `src/chain.py` first comment block notes the simplified path takes over as production demo
- [ ] `docs/decisions.md` exists with multiple numbered + dated sections
- [ ] `docs/evaluation-findings.md` exists with 9 stages
- [ ] `docs/test-passes/` contains 5 test pass docs + a README
- [ ] `docs/architecture-diagrams.html` opens in a browser
- [ ] Local demo: `PYTHONPATH=$PWD python -m src.ui.simple_chat` runs and the FRIA query produces a 3-section output with retrieval evidence
- [ ] All commits pushed to your branch

If all 14 boxes tick, the simplified path is fully ported and verified
in your repo. You can then open a PR to your own `main` branch (or
merge directly), document it, and move on to writing the report.

---

## After implementation: what's next

Once the simplified path is in your `main`:

1. **Practice the demo.** Run `python -m src.ui.simple_chat` a few times
   so you're fluent with the CLI on demo day. Try Q5 (FRIA — the canary
   silence target). Ask the gap finding to a couple of compliance
   officers if you can; verify they understand the output.

2. **Read `docs/evaluation-findings.md` end-to-end.** This is the
   strongest material for your report's Critical Analysis dimension.
   The 9 stages map onto the empirical narrative the marker will grade
   you on.

3. **Read `docs/decisions.md` end-to-end.** This is your viva-defence
   reference. Each entry has a "Likely viva probe + answer" that you can
   practice.

4. **Use the architecture diagrams** (`docs/architecture-diagrams-slides.html`)
   for slide screenshots. Three slides cover the simplified path, the
   chain, and the comparison.

5. **Start writing the report.** The empirical work is saturated; the
   writing is what gets graded.

If anything is unclear, ask Bogdan. He has full context on every decision
and every empirical finding.

---

## Glossary (in case Claude Code uses jargon you don't know)

- **BGE-large** — the embedding model that converts text into 1024-dimensional vectors so we can find similar passages by cosine similarity. Stays the same regardless of which LLM is used.
- **Cross-encoder reranker** — a smaller model that re-scores the top-10 chunks to improve ranking. Slower per chunk than BGE but more accurate.
- **RRF (Reciprocal Rank Fusion)** — an alternative way to combine BGE rank and reranker rank. Tested and rejected (Stage 9).
- **V4 prompt** — the system prompt template for Qwen models. Long, with 5 numbered rules.
- **Gemma-adapted prompt** — the system prompt template for Gemma 3. Short, with rules moved to the user turn.
- **CONFIG dict** — the Python dictionary loaded from `config.toml` at module import. Mutable from a Colab cell.
- **Stage X** — a numbered section of `docs/evaluation-findings.md`, each documenting a specific empirical finding.
- **Test pass** — a verbatim run-output document in `docs/test-passes/`, dated and configured.
- **Cache hit / cache miss** — the system caches LLM responses keyed on `(prompt, model_id)`. A "hit" means we return the cached response without re-calling the LLM. Faster but means the same query gives the same answer every time.
- **`PYTHONPATH=$PWD`** — environment variable that tells Python to look for modules starting from the current directory. Required because `src.simplified` lives in `src/`, not the standard Python path.

---

This guide is one self-contained document. If you (or your Claude Code)
need to refer back to any decision, code file, or empirical finding,
the references in this guide point at the right place. Good luck.
