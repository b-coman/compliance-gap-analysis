# Colab — evaluation environment

This folder holds Colab notebooks used to evaluate the simplified architecture
(`src/simplified.py`) on larger LLMs than the local CPU can run.

**Local CPU stays the production demo path.** Colab is for the model-scale
comparison: same code, bigger model, GPU acceleration. Outputs feed
`docs/test-passes/` for the report appendix and viva narrative.

## Files

- `run_simplified_colab.ipynb` — clones the repo, installs deps, sets `MODEL_ID`,
  runs the 5 standard test queries, writes results to `colab_outputs.md`.

## How to use

1. Open the notebook in Colab (File → Open notebook → GitHub → paste this repo URL → pick the file).
2. Runtime → Change runtime type → **GPU** (T4 is fine for most models on the menu).
3. In the model-selector cell, uncomment exactly one line — see the table in the notebook.
4. Run all cells.

The notebook installs only `sentence-transformers transformers accelerate diskcache autoawq` — Colab's preinstalled torch/numpy stay in place to keep the CUDA wheel alignment intact. Do not run `pip install -r requirements.txt` on Colab; that file is for the local Mac CPU environment.

## Model menu

| Model | Size | Fits T4? | HF gated? | Purpose |
|---|---|---|---|---|
| `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | Yes | No | Control — same as local default |
| `Qwen/Qwen2.5-3B-Instruct` | 3B | Yes | No | Intermediate scale |
| `Qwen/Qwen2.5-7B-Instruct` | 7B | No (CPU offload) | No | Slow on T4 (~3 min/query) |
| `Qwen/Qwen2.5-7B-Instruct-AWQ` | 7B (4-bit) | Yes | No | Fast 7B on T4 |
| `google/gemma-2-2b-it` | 2.6B | Yes | **Yes** | Alternative family |

For Gemma: accept the license on its HuggingFace model page, then add `HF_TOKEN` as a Colab secret (left sidebar → key icon). The notebook auto-detects gated models and pulls the token.

## Notes

- AWQ models require the `autoawq` package (already installed by the pip cell). They load through the same `transformers.from_pretrained` path; no additional code changes.
- For Gemma 2-9B or Llama 3-8B, T4 is too small for fp16 — would need bitsandbytes 4-bit quantisation, which is a code change in `src/simplified.py` not currently implemented.
5. Download `colab_outputs.md` from the Files panel; rename and commit it under
   `docs/test-passes/` with the convention `<prompt-version>-<model>-colab.md`.

## Why a separate environment

See `docs/decisions.md` entry "Colab as evaluation environment, local CPU as
demo path" for the rationale and trade-offs.
