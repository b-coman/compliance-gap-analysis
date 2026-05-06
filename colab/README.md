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

The notebook installs only `sentence-transformers transformers accelerate diskcache "numpy<2.2"` — Colab's preinstalled torch/numpy stay in place to keep the CUDA wheel alignment intact. Do not run `pip install -r requirements.txt` on Colab; that file is for the local Mac CPU environment.

5. Download `colab_outputs.md` from the Files panel; rename and commit it under
   `docs/test-passes/` with the convention `<prompt-version>-<model>-colab.md`.

## Model menu

| Model | Size | Fits T4? | HF gated? | Purpose |
|---|---|---|---|---|
| `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | Yes | No | Control — same as local default |
| `Qwen/Qwen2.5-3B-Instruct` | 3B | Yes | No | Intermediate scale, fast on T4 |
| `Qwen/Qwen2.5-7B-Instruct` | 7B | No (CPU offload) | No | Full 7B baseline (~3 min/query on T4) |
| `google/gemma-2-2b-it` | 2.6B | Yes | **Yes** | Older Gemma family — smaller comparison point |
| `google/gemma-3-4b-it` | 4B | Yes | **Yes** | Newer Gemma family — direct family-vs-family comparison to Qwen 3B (recommended Gemma) |

For Gemma: accept the license on its HuggingFace model page, then add `HF_TOKEN` as a Colab secret (left sidebar → key icon). The notebook auto-detects gated models and pulls the token.

**Gemma 3 caveat:** Gemma 3 (March 2025) requires `transformers >= 4.49`. If Colab's preinstalled transformers is older, the model load will raise `Unknown architecture: Gemma3ForCausalLM`. Fix: `!pip install --upgrade transformers`, restart runtime, re-run. The upgrade may cascade into other dep conflicts; fallback to Gemma 2-2B.

## Notes

- An AWQ-quantised 7B option (`Qwen/Qwen2.5-7B-Instruct-AWQ`) was previously in the menu as a fast-on-T4 alternative. The required runtime libraries (`autoawq`, `gptqmodel`) had cross-incompatibilities with Colab's preinstalled scientific stack at run time and we removed the option. Qwen 3B fp16 is the substitute for fast Colab runs.
- For Gemma 2-9B or Llama 3-8B, T4 is too small for fp16 — would need bitsandbytes 4-bit quantisation, which is a code change in `src/simplified.py` not currently implemented.

## Why a separate environment

See `docs/decisions.md` entry "Colab as evaluation environment, local CPU as
demo path" for the rationale and trade-offs.
