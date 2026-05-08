# Compliance Gap Analysis

A retrieval-augmented compliance gap-finding system for AI Act + GDPR
deployer obligations, evaluated against the Novara TalentLens policy
corpus. Single-call architecture: BGE-large bi-encoder retrieval +
single LLM inference call (Gemma 2-2B).

> **INST0100 — Generative AI for Information Processing.**
> Submission for the AI Compliance Gap Analysis project brief.

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=$PWD python -m src.ui.simple_chat
```

Type a compliance question. The system returns a three-section gap
analysis with a retrieval-grounding footer.

The local LLM (`google/gemma-2-2b-it`) is gated on HuggingFace; you'll
need a `HF_TOKEN` environment variable or `huggingface-cli login` once
to access it.

---

## How it works

1. **Retrieval.** BGE-large bi-encoder embeds the user query, retrieves
   top-5 relevant law passages and top-5 relevant deployer-policy
   passages from a 600-chunk legal corpus (EU AI Act, UK GDPR, Novara
   policy and supporting documents).
2. **Generation.** Single LLM call (Gemma 2-2B, greedy decoding).
   Prompt template merges the rules into the user turn (Gemma 2 chat
   templates do not support a system role).
3. **Grounding.** A footer reports the retrieval-pool confidence per
   side using mean / max / min cosine similarity (the three summary
   statistics taught in the INST0100 class example) plus a verbal
   label and pattern interpretation.

Output shape:

```
### What the law requires
[regulation/...] <obligation in verbatim legal phrasing>

### What the policy says
[deployer/...] <description of policy coverage, or explicit "policy
does not address ..." statement>

### Gap
[regulation/...] <gap finding>

---
Retrieval grounding (based on BGE cosine across top-5 chunks):
  Law passages:    strong   (mean=0.74, max=0.79, min=0.72)
  Policy passages: strong   (mean=0.73, max=0.76, min=0.71)
  Pattern: well-grounded on both sides.
```

---

## Demo

A Colab notebook is available for the live demonstration:

- [`colab/run_simplified_colab.ipynb`](colab/run_simplified_colab.ipynb)

Open in Colab → Runtime → GPU → Run all. Section 4 of the notebook
runs a single configurable query; Section 5 (optional) runs all five
evaluation queries and saves the outputs.

---

## Repository layout

```
src/
  simplified.py         single-call architecture; the production path
  ingestion.py          corpus loading and chunking
  ui/simple_chat.py     interactive CLI
  llm/cache.py          DiskCache utility
corpus/                 EU AI Act + UK GDPR + Novara policy + supporting docs
colab/                  Colab notebook for the demonstration
docs/                   evaluation findings, decisions, test passes, diagrams
tests/                  pytest smoke tests
```

---

## Where the empirical work lives

- [`docs/evaluation-findings.md`](docs/evaluation-findings.md) — nine
  stages mapping the empirical journey: chain → simplified, model
  scaling, family comparison, prompt format, ranking strategy. The
  primary source for the report's Critical Analysis dimension.
- [`docs/decisions.md`](docs/decisions.md) — per-decision rationale
  with viva-ready answers for likely probes.
- [`docs/test-passes/`](docs/test-passes/) — five verbatim run-outputs
  from cross-model and cross-strategy ablations.
- [`docs/architecture-diagrams.html`](docs/architecture-diagrams.html)
  — visual reference for the two architectures evaluated (chain and
  simplified).

---

## License

[MIT](LICENSE).
