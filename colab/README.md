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
2. Runtime → Change runtime type → **GPU** (T4 is fine for 7B in fp16).
3. Edit the `MODEL_ID` env var cell to pick the model (default: `Qwen/Qwen2.5-7B-Instruct`).
4. Run all cells.
5. Download `colab_outputs.md` from the Files panel; rename and commit it under
   `docs/test-passes/` with the convention `<prompt-version>-<model>-colab.md`.

## Why a separate environment

See `docs/decisions.md` entry "Colab as evaluation environment, local CPU as
demo path" for the rationale and trade-offs.
