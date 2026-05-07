"""Simplified compliance gap analysis path — single-call architecture.

This module is the production demo path. It is deliberately isolated from
the multi-step chain in `src/chain.py` and the chain-specific LLM cluster
in `src/llm/`. The simplified path operates entirely on local models and
does not depend on any external API service.

Architecture (single LLM call per query):

  user query
      │
      ├── retrieve top-k REG chunks (BGE-large embedding, no LLM call)
      ├── retrieve top-k DEP+DEP_EXTRAS chunks (BGE-large, no LLM call)
      ├── construct system + user message with retrieved chunks
      └── single LLM call (local model: Qwen 1.5B by default)
              │
              └── return generated text (3-section compliance assessment)

Design rationale (see docs/evaluation-findings.md Stages 5-7 for evidence):

  - BGE-large empirically improves retrieval recall over MiniLM, particularly
    on queries with vocabulary mismatch (verified Stage 5/6).
  - Excluding ICO operational guidance at query time avoids the contamination
    we documented at chain CHN-04 Phase 1 silence detection.
  - Single LLM call avoids the chain's three architectural failure modes
    (decomposition drift, miscalibrated silence detection, mislabelled
    provenance) documented in Stage 7.
  - Local model (no Groq dependency) sidesteps free-tier daily token limits
    that blocked iterative development on the chain. The brief's compute
    constraint (consumer hardware or free-tier API) is met via consumer
    hardware.

Isolation invariant: this module imports only from `src.retrieval` (shared
infrastructure), `src.llm.cache` (general-purpose disk cache utility), and
external libraries (sentence_transformers, transformers, torch). It does
NOT import from src.chain, src.schema, or any other src.llm.* module.
This invariant is mechanically verifiable via grep on the imports below.
"""
from __future__ import annotations

import os
import time
import tomllib
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# Shared / general-purpose imports only — no chain or chain-LLM imports
from src.retrieval import build_retriever
from src.llm.cache import DiskCache


# ───── Configuration ─────
#
# All tunables live in `config.toml` at the repo root. CONFIG is loaded once
# at module import. Three override mechanisms with clear precedence:
#   1. Environment variables (highest)        — env-specific overrides
#   2. CONFIG dict mutation (session-scoped)  — cell-driven experimentation
#   3. config.toml file (project default)     — committed defaults
#
# Hot-mutable (CONFIG dict edits take effect on the next analyse() call):
#   retrieval.top_k_reg / top_k_dep / top_k_initial / snippet_chars_limit
#   ranking.strategy, ranking.rrf.k
#   reranker.confidence_thresholds.strong / moderate
#   llm.generation.*
#   output.show_evidence
#
# Restart-required (resolved at module import time):
#   llm.model_id, embedding.model_id, reranker.model_id, paths.*

_CONFIG_PATH = Path(__file__).parent.parent / "config.toml"
with open(_CONFIG_PATH, "rb") as _f:
    CONFIG: dict = tomllib.load(_f)


def _get(path: list[str], env_var: str | None = None):
    """Resolve a config value with optional environment-variable override.

    Precedence: env var > CONFIG dict (which may have been mutated at runtime).
    Reads `CONFIG` fresh each call, so cell-driven mutations are picked up
    on the next analyse() invocation.
    """
    if env_var and env_var in os.environ:
        return os.environ[env_var]
    cur = CONFIG
    for k in path:
        cur = cur[k]
    return cur


# Init-time constants (resolved once at module load; restart required to swap).
# These are exposed at module level for backward compatibility — external code
# may import them. Runtime CONFIG mutation does NOT affect them; use env vars
# (MODEL_ID, EMBED_MODEL_ID) for per-session overrides.
LLM_MODEL_ID       = _get(["llm", "model_id"], env_var="MODEL_ID")
EMBED_MODEL_ID     = _get(["embedding", "model_id"], env_var="EMBED_MODEL_ID")
EMBED_QUERY_PREFIX = _get(["embedding", "query_prefix"])
EMBED_CACHE_DIR    = Path(_get(["paths", "embeddings_bge_cache"]))
LLM_CACHE_DIR      = Path(_get(["paths", "llm_cache_simplified"]))
RERANKER_MODEL_ID  = _get(["reranker", "model_id"])


# ───── Prompt templates ─────
#
# Prompt V4 (deployed). See docs/evaluation-findings.md Stage 4 for V1→V3 evolution.
# V4 change vs V3: topic-specific examples removed from system prompt. Empirical
# baseline (docs/simplified-baseline.md) showed that small models (Qwen 1.5B at
# this scale) treat illustrative phrases in instructions as content-to-surface,
# producing a "FRIA leak" on queries unrelated to fundamental rights. Best
# practice for small-model prompt hygiene is to keep system prompts strictly
# procedural with no topical examples; if examples are needed, use few-shot
# turns in the chat template rather than embedded illustrative text.
# See: web.dev "Practical prompt engineering for smaller LLMs"; OWASP LLM07 on
# system prompt leakage.

SYSTEM_PROMPT = (
    "You are a compliance research assistant for a Head of AI Compliance. "
    "You produce concise compliance assessments comparing what the law requires "
    "against what a company's policy says.\n\n"
    "RULES:\n"
    "1. Cite chunk IDs verbatim in [square brackets].\n"
    "2. Quote key legal phrases from the law passages verbatim. Do not paraphrase.\n"
    "3. Do not invent obligations or policy provisions not present in the passages provided.\n"
    "4. Stay on the specific topic in the question. Different legal frameworks have "
    "different obligations even when topics seem similar; do not conflate them.\n"
    "5. If the policy passages do not address the obligation in the question, "
    "say so directly. Do not stretch unrelated content to fit."
)


# Gemma 3 prompt variant.
#
# Per Google's official Gemma 3 launch documentation
# (https://huggingface.co/blog/gemma3): "Gemma 3 uses very short system prompts
# followed by user prompts." The example shown there is a single sentence.
# Our V4 SYSTEM_PROMPT (5 numbered rules + role description) is substantially
# longer than what Gemma 3's instruction-tuning was optimised for, and we
# observed empirically that Gemma 3-4B partially under-weighted those rules
# on Q5 (accepted Novara's self-classification, missed the FRIA-vs-DPIA
# distinction that Qwen 3B and 7B handled correctly with the same prompt).
# Hypothesis: rules in a long system prompt have less behavioural pull on
# Gemma 3 than rules placed in the user turn. This Gemma-specific variant
# tests that hypothesis: short role-only system prompt, rules in the user
# turn ahead of the question. See `docs/test-passes/v4-gemma-3-4b-colab.md`
# for the empirical motivation; results from this variant will be captured
# in a follow-up test pass.

GEMMA3_SYSTEM_PROMPT = (
    "You are a compliance research assistant for a Head of AI Compliance."
)


def _user_message(query: str, reg_text: str, dep_text: str) -> str:
    return f"""QUESTION: {query}

LAW PASSAGES:
{reg_text}

POLICY PASSAGES:
{dep_text}

Output your assessment in three Markdown sections. Instructions for each section:

- "What the law requires": one sentence stating the specific obligation from the law passages. Use verbatim legal phrases (e.g. "fundamental rights impact assessment"). Begin with the relevant law chunk_id in square brackets.

- "What the policy says": one sentence describing what the policy says about this obligation. Cite policy chunk_ids in square brackets if relevant. If the policy does not address the obligation, write a complete sentence stating which specific obligation is missing — name it in plain English (for example, "The policy does not address performing a fundamental rights impact assessment"). Do not output bracketed placeholders or instruction text.

- "Gap": one sentence directly stating the gap between the law and the policy, citing the relevant law chunk_id in square brackets.

Now produce your assessment using this exact format:

### What the law requires

### What the policy says

### Gap"""


def _user_message_gemma3(query: str, reg_text: str, dep_text: str) -> str:
    """User message for Gemma 3: rules at the top, then question + chunks +
    template. Pairs with `GEMMA3_SYSTEM_PROMPT` (role-only).

    Same content as `_user_message` plus the 5 V4 rules prepended. Topic-
    specific examples (e.g., "fundamental rights impact assessment") removed
    from the rule text to avoid biasing Gemma toward those phrases — same
    hygiene principle that motivated V3 → V4 for the Qwen system prompt.
    """
    return f"""Follow these rules in your response:
1. Cite chunk IDs verbatim in [square brackets].
2. Quote key legal phrases from the law passages verbatim. Do not paraphrase.
3. Do not invent obligations or policy provisions not present in the passages provided.
4. Stay on the specific topic in the question. Different legal frameworks have different obligations even when topics seem similar; do not conflate them.
5. If the policy passages do not address the obligation in the question, say so directly. Do not stretch unrelated content to fit.

QUESTION: {query}

LAW PASSAGES:
{reg_text}

POLICY PASSAGES:
{dep_text}

Output your assessment in three Markdown sections. Instructions for each section:

- "What the law requires": one sentence stating the specific obligation from the law passages. Use verbatim legal phrases. Begin with the relevant law chunk_id in square brackets.

- "What the policy says": one sentence describing what the policy says about this obligation. Cite policy chunk_ids in square brackets if relevant. If the policy does not address the obligation, write a complete sentence stating which specific obligation is missing — name it in plain English. Do not output bracketed placeholders or instruction text.

- "Gap": one sentence directly stating the gap between the law and the policy, citing the relevant law chunk_id in square brackets.

Now produce your assessment using this exact format:

### What the law requires

### What the policy says

### Gap"""


def _get_prompts(model_id: str, query: str, reg_text: str, dep_text: str) -> tuple[str, str]:
    """Return the (system_prompt, user_message) pair appropriate for the
    selected model family. Gemma 3 gets a short system + rules-in-user
    structure (per Google's prompt-engineering guidance); other families
    get the standard V4 long system prompt + question-only user message.
    """
    if _is_gemma3(model_id):
        return GEMMA3_SYSTEM_PROMPT, _user_message_gemma3(query, reg_text, dep_text)
    return SYSTEM_PROMPT, _user_message(query, reg_text, dep_text)


# ───── BGE retriever wrapper ─────

class _BGERetriever:
    """Single-purpose retriever using BGE-large with the model's recommended
    query-instruction prefix. Re-uses the chunks from the standard
    `build_retriever()` so chunking + corpus loading is shared with the chain.
    Embeddings are computed independently with BGE.
    """

    def __init__(self, chunks, model: SentenceTransformer, embeddings: np.ndarray):
        self.chunks = chunks
        self._model = model
        self._embeddings = embeddings  # already L2-normalized

    def retrieve(self, query: str, top_k: int,
                 corpus_filter: str | Sequence[str] | None = None
                 ) -> list[tuple]:
        prefixed = EMBED_QUERY_PREFIX + query
        q_emb = self._model.encode(
            prefixed, convert_to_numpy=True, normalize_embeddings=True
        )

        if corpus_filter is None:
            idx = list(range(len(self.chunks)))
        else:
            tags = (
                (corpus_filter,) if isinstance(corpus_filter, str)
                else tuple(corpus_filter)
            )
            idx = [i for i, c in enumerate(self.chunks) if c.corpus_tag in tags]

        if not idx:
            return []

        filtered = self._embeddings[idx]
        scores = (filtered @ q_emb).tolist()
        top = sorted(enumerate(scores), key=lambda x: -x[1])[:top_k]
        return [(self.chunks[idx[i]], float(s)) for i, s in top]


# ───── Reranker (cross-encoder) + ranking strategy dispatch ─────

# Lazy-loaded singleton. See _ensure_reranker() in the singletons section.
_reranker = None


def _rrf_combine(bge_ranking: list, rerank_ranking: list, top_k: int, k: int = 60) -> list:
    """Reciprocal Rank Fusion. Each chunk's RRF score is the sum of
    1/(k + rank) across the BGE and reranker rankings. k=60 is the
    original-paper default (no per-query tuning required).

    Standard pattern when blending bi-encoder and cross-encoder signals:
    a chunk that's strong in BGE but weak in reranker still gets credit
    (preserved in top-K), and vice versa. Mitigates the failure mode where
    a low-confidence reranker drops a BGE-relevant chunk out of top-K.

    Args:
        bge_ranking:    [(chunk, bge_score), ...] sorted by bge_score desc
        rerank_ranking: [(chunk, rerank_score), ...] sorted by rerank_score desc
        top_k: number of final hits to return
        k: smoothing constant
    Returns:
        [(chunk, rrf_score), ...] sorted by rrf_score desc, length top_k
    """
    bge_ranks = {c.chunk_id: r for r, (c, _) in enumerate(bge_ranking, start=1)}
    rerank_ranks = {c.chunk_id: r for r, (c, _) in enumerate(rerank_ranking, start=1)}
    chunks_by_id = {c.chunk_id: c for c, _ in bge_ranking}
    fallback = max(len(bge_ranking), len(rerank_ranking)) + 1

    rrf_scores = []
    for cid in chunks_by_id:
        score = (
            1 / (k + bge_ranks.get(cid, fallback))
            + 1 / (k + rerank_ranks.get(cid, fallback))
        )
        rrf_scores.append((chunks_by_id[cid], score))

    rrf_scores.sort(key=lambda x: -x[1])
    return rrf_scores[:top_k]


def _rerank_with_evidence(query: str, initial_hits: list, top_k: int) -> tuple[list, list]:
    """Rescore retrieved chunks and return final hits + audit trail.

    Strategy is read from CONFIG (or RANKING_STRATEGY env var). Three modes:
      "rrf"          (default) — blend BGE rank and cross-encoder rank via RRF
      "rerank_only"            — cross-encoder fully replaces BGE order
      "bge_only"               — disable reranker; BGE top_k unchanged

    Returns:
        final_hits: [(chunk, score)] — top_k for the prompt. The score is
            BGE / RRF / rerank depending on strategy.
        evidence:   list of dicts with keys
            {chunk, bge_score, rerank_score (or None), bge_rank, final_rank}
            for each chunk that survived to top_k. Drives _format_evidence().
    """
    if not initial_hits:
        return [], []

    strategy = _get(["ranking", "strategy"], env_var="RANKING_STRATEGY")

    if strategy == "bge_only":
        # No reranker; just take top_k from BGE order
        final = list(initial_hits[:top_k])
        evidence = [
            {
                "chunk": c, "bge_score": float(bge), "rerank_score": None,
                "bge_rank": i, "final_rank": i,
            }
            for i, (c, bge) in enumerate(final, start=1)
        ]
        return final, evidence

    # Both rrf and rerank_only need the cross-encoder scores
    reranker = _ensure_reranker()
    pairs = [(query, c.chunk_text) for c, _ in initial_hits]
    rerank_scores = reranker.predict(pairs)
    extended = [
        (chunk, float(bge), float(rs), bge_rank)
        for bge_rank, ((chunk, bge), rs) in enumerate(
            zip(initial_hits, rerank_scores), start=1
        )
    ]

    if strategy == "rerank_only":
        extended.sort(key=lambda x: -x[2])  # descending by rerank score
        final = [(c, rs) for c, _bge, rs, _br in extended[:top_k]]
        evidence = [
            {
                "chunk": c, "bge_score": bge, "rerank_score": rs,
                "bge_rank": bge_rank, "final_rank": final_rank,
            }
            for final_rank, (c, bge, rs, bge_rank) in enumerate(
                extended[:top_k], start=1
            )
        ]
        return final, evidence

    if strategy == "rrf":
        bge_ranking = [(c, bge) for c, bge, _rs, _br in extended]
        rerank_ranking = sorted(
            [(c, rs) for c, _bge, rs, _br in extended],
            key=lambda x: -x[1],
        )
        rrf_k = int(_get(["ranking", "rrf", "k"]))
        rrf_top = _rrf_combine(bge_ranking, rerank_ranking, top_k, k=rrf_k)

        # Map chunk_id back to its (bge, rerank, bge_rank) for audit
        ext_by_id = {c.chunk_id: (bge, rs, bge_rank) for c, bge, rs, bge_rank in extended}
        evidence = [
            {
                "chunk": c,
                "bge_score": ext_by_id[c.chunk_id][0],
                "rerank_score": ext_by_id[c.chunk_id][1],
                "bge_rank": ext_by_id[c.chunk_id][2],
                "final_rank": final_rank,
            }
            for final_rank, (c, _rrf) in enumerate(rrf_top, start=1)
        ]
        return rrf_top, evidence

    raise ValueError(
        f"Unknown ranking strategy: {strategy!r}. "
        f"Expected 'rrf', 'rerank_only', or 'bge_only'."
    )


def _classify_confidence(max_score: float) -> str:
    """Map max reranker score to a verbal confidence label."""
    strong = float(_get(["reranker", "confidence_thresholds", "strong"]))
    moderate = float(_get(["reranker", "confidence_thresholds", "moderate"]))
    if max_score >= strong:
        return "strong"
    if max_score >= moderate:
        return "moderate"
    return "weak"


def _pattern_label(reg_label: str, dep_label: str) -> str:
    """Map (law confidence, policy confidence) to a pattern interpretation
    written in plain English for a non-technical compliance reader."""
    reg_high = reg_label in ("strong", "moderate")
    dep_high = dep_label in ("strong", "moderate")
    if reg_high and dep_high:
        return "well-grounded on both sides."
    if reg_high and not dep_high:
        return "policy may be silent on this obligation."
    if not reg_high and dep_high:
        return "law side weak — query may not match the law corpus well."
    return "low confidence on both sides — consider rephrasing the query."


def _format_evidence(reg_evidence: list, dep_evidence: list) -> str:
    """Render the retrieval audit trail with two parts:
      1. Friendly grounding summary (Option C: per-side confidence label +
         pattern interpretation), readable to a non-technical compliance audience
      2. Detailed per-chunk audit (BGE score, rerank score, rank changes),
         for the test pass docs and marker probes
    """
    if not reg_evidence and not dep_evidence:
        return ""

    has_rerank = (
        reg_evidence and reg_evidence[0]["rerank_score"] is not None
    )
    strategy = _get(["ranking", "strategy"], env_var="RANKING_STRATEGY")

    lines = ["---"]

    if has_rerank:
        reg_max = max(e["rerank_score"] for e in reg_evidence)
        dep_max = max(e["rerank_score"] for e in dep_evidence)
        reg_label = _classify_confidence(reg_max)
        dep_label = _classify_confidence(dep_max)
        pattern = _pattern_label(reg_label, dep_label)

        lines += [
            "Retrieval grounding:",
            f"  Law passages:    {reg_label:<10} (max reranker confidence {reg_max:.2f})",
            f"  Policy passages: {dep_label:<10} (max reranker confidence {dep_max:.2f})",
            f"  Pattern: {pattern}",
            "",
        ]

        if strategy == "rrf":
            heading = "Detailed retrieval evidence (after RRF combining BGE + cross-encoder):"
        else:
            heading = "Detailed retrieval evidence (after cross-encoder reranking):"
    else:
        heading = "Detailed retrieval evidence (BGE only, no reranker applied):"

    lines.append(heading)

    def _block(label: str, evidence: list) -> list[str]:
        block = ["", f"{label}:"]
        for e in evidence:
            chunk = e["chunk"]
            bge = e["bge_score"]
            rs = e["rerank_score"]
            bge_rank = e["bge_rank"]
            final_rank = e["final_rank"]

            score_part = f"BGE {bge:.3f}"
            if rs is not None:
                score_part += f" / rerank {rs:.2f}"

            rank_part = (
                f"rank {bge_rank} → {final_rank}"
                if bge_rank != final_rank else f"rank {bge_rank} (unchanged)"
            )
            block.append(
                f"  {final_rank}. {chunk.chunk_id}  ({score_part}, {rank_part})"
            )
        return block

    lines += _block("Law passages (final top-5)", reg_evidence)
    lines += _block("Policy passages (final top-5)", dep_evidence)
    return "\n".join(lines)


# ───── Chunk formatting for the prompt ─────

def _format_chunks(hits, snippet_chars: int | None = None) -> str:
    if snippet_chars is None:
        snippet_chars = int(_get(["retrieval", "snippet_chars_limit"]))
    lines = []
    for chunk, _score in hits:
        text = chunk.chunk_text
        if len(text) > snippet_chars:
            text = text[:snippet_chars] + "…"
        lines.append(f"[{chunk.chunk_id}] {chunk.section_reference}\n{text}")
    return "\n\n".join(lines)


# ───── LLM (local) ─────

def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _is_gemma3(model_id: str) -> bool:
    """Family detector. Gemma 3 (4B+) is multimodal and needs a different
    transformers class + chat-template format than the standard causal-LM
    path used for Qwen / Mistral / Gemma 2. Detection is by model_id at load
    time and by `model.config.model_type == "gemma3"` at call time.
    """
    return "gemma-3" in model_id.lower()


def _load_llm_default(model_id: str):
    """Load via AutoModelForCausalLM. Used for Qwen, Mistral, Gemma 1/2, etc.

    Device-aware: CUDA → fp16 with `device_map="auto"` (avoids the Colab T4
    CPU-RAM OOM on 7B models — accelerate streams weights to GPU during load).
    Otherwise CPU → fp32 (also the safe path on Apple Silicon, where MPS has
    a matmul shape bug on some decoder layouts).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.float16, device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.float32
        ).to("cpu")
    return tokenizer, model


def _load_llm_gemma3(model_id: str):
    """Load Gemma 3 via Gemma3ForConditionalGeneration + AutoProcessor.

    Gemma 3 4B+ are multimodal models. Google's loaders expect the
    conditional-generation class plus an `AutoProcessor` (not a plain
    tokenizer) to handle the multimodal input format. For text-only
    inference we use the same class but format messages with content as a
    list of `{"type": "text", "text": ...}` dicts (see _call_llm_gemma3).
    bfloat16 is the recommended dtype per Google's model card.

    On CPU we fall back to fp32, but Gemma 3 on CPU is impractical
    (multi-minute generation per query). Intended target is Colab GPU.
    """
    from transformers import Gemma3ForConditionalGeneration, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    if torch.cuda.is_available():
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="auto"
        ).eval()
    else:
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id, dtype=torch.float32
        ).to("cpu").eval()
    return processor, model


def _load_llm(model_id: str = LLM_MODEL_ID):
    """Family dispatcher. Routes to the right loader based on model_id."""
    if _is_gemma3(model_id):
        return _load_llm_gemma3(model_id)
    return _load_llm_default(model_id)


def _call_llm_default(tokenizer, model, system: str, user: str,
                      max_new_tokens: int | None = None,
                      repetition_penalty: float | None = None) -> str:
    """Default inference path: Qwen, Mistral, Gemma 1/2, etc.

    Gemma 1 and Gemma 2 chat templates do not support a `system` role
    (only `user` and `model`); passing one drops it silently or produces a
    malformed prompt. Detect Gemma and merge system content into the user
    turn. Gemma 3 has its own dedicated path (handles system natively via
    content-list format) and never reaches this function.
    """
    if max_new_tokens is None:
        max_new_tokens = int(_get(["llm", "generation", "max_new_tokens"]))
    if repetition_penalty is None:
        repetition_penalty = float(_get(["llm", "generation", "repetition_penalty"]))

    is_gemma_legacy = (
        getattr(model.config, "model_type", "").lower().startswith("gemma")
    )
    if is_gemma_legacy:
        messages = [
            {"role": "user", "content": f"{system}\n\n{user}"},
        ]
    else:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    # Gemma's natural stop is <end_of_turn>, not <eos>. Letting transformers
    # use the model's generation_config (which carries the right stops) is
    # safer than forcing eos_token_id. For Qwen, the explicit eos avoids a
    # pad-token warning at generation time.
    generate_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.eos_token_id,
    )
    if not is_gemma_legacy:
        generate_kwargs["eos_token_id"] = tokenizer.eos_token_id

    with torch.no_grad():
        outputs = model.generate(**inputs, **generate_kwargs)
    gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def _call_llm_gemma3(processor, model, system: str, user: str,
                     max_new_tokens: int | None = None,
                     repetition_penalty: float | None = None) -> str:
    """Gemma 3 inference path.

    Gemma 3's chat template expects content as a list of `{"type": "text",
    "text": ...}` dicts (the multimodal content envelope). Plain strings
    silently produce malformed prompts and empty generation. Uses
    `processor.apply_chat_template` rather than `tokenizer.apply_chat_template`,
    with `tokenize=True, return_dict=True` to get input_ids+attention_mask
    in one call. Stop tokens are handled by the model's generation_config
    (Gemma uses `<end_of_turn>`).
    """
    if max_new_tokens is None:
        max_new_tokens = int(_get(["llm", "generation", "max_new_tokens"]))
    if repetition_penalty is None:
        repetition_penalty = float(_get(["llm", "generation", "repetition_penalty"]))

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": [{"type": "text", "text": user}]},
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=repetition_penalty,
        )
    gen_ids = outputs[0][input_len:]
    return processor.decode(gen_ids, skip_special_tokens=True)


def _call_llm(tokenizer_or_processor, model, system: str, user: str,
              max_new_tokens: int | None = None,
              repetition_penalty: float | None = None) -> str:
    """Family dispatcher. Routes to the right inference path based on the
    loaded model's `config.model_type`.
    """
    is_gemma3 = getattr(model.config, "model_type", "").lower() == "gemma3"
    if is_gemma3:
        return _call_llm_gemma3(
            tokenizer_or_processor, model, system, user,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
        )
    return _call_llm_default(
        tokenizer_or_processor, model, system, user,
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
    )


# ───── Module-level state (lazy-loaded singletons) ─────
_retriever: _BGERetriever | None = None
_llm_tokenizer = None
_llm_model = None
_llm_cache: DiskCache | None = None


def _ensure_retriever() -> _BGERetriever:
    """Build the BGE retriever once per process."""
    global _retriever
    if _retriever is not None:
        return _retriever

    print(f"[simplified] Loading chunks via build_retriever()...")
    standard = build_retriever()
    chunks = standard.chunks

    print(f"[simplified] Loading BGE model ({EMBED_MODEL_ID}) on {_device()}...")
    bge_model = SentenceTransformer(EMBED_MODEL_ID, device=_device())

    cache_file = EMBED_CACHE_DIR / "embeddings.npy"
    if cache_file.exists():
        print(f"[simplified] Loading cached BGE embeddings from {cache_file}")
        embeddings = np.load(cache_file)
    else:
        print(f"[simplified] Embedding {len(chunks)} chunks with BGE (one-time)...")
        t0 = time.time()
        embeddings = bge_model.encode(
            [c.chunk_text for c in chunks],
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )
        EMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(cache_file, embeddings)
        print(f"[simplified]   embedded in {time.time()-t0:.1f}s; cached to {cache_file}")

    _retriever = _BGERetriever(chunks, bge_model, embeddings)
    return _retriever


def _ensure_llm():
    """Load the local LLM once per process."""
    global _llm_tokenizer, _llm_model
    if _llm_model is not None:
        return _llm_tokenizer, _llm_model
    print(f"[simplified] Loading LLM ({LLM_MODEL_ID})...")
    t0 = time.time()
    _llm_tokenizer, _llm_model = _load_llm(LLM_MODEL_ID)
    print(f"[simplified]   loaded in {time.time()-t0:.1f}s")
    return _llm_tokenizer, _llm_model


def _ensure_cache() -> DiskCache:
    global _llm_cache
    if _llm_cache is None:
        _llm_cache = DiskCache(cache_dir=LLM_CACHE_DIR)
    return _llm_cache


def _ensure_reranker():
    """Lazy-load the cross-encoder reranker once per process."""
    global _reranker
    if _reranker is not None:
        return _reranker
    from sentence_transformers import CrossEncoder
    print(f"[simplified] Loading reranker ({RERANKER_MODEL_ID}) on {_device()}...")
    t0 = time.time()
    _reranker = CrossEncoder(RERANKER_MODEL_ID, device=_device())
    print(f"[simplified]   reranker loaded in {time.time()-t0:.1f}s")
    return _reranker


# ───── Public API ─────

def analyse(query: str, *, use_cache: bool = True) -> str:
    """Run the simplified compliance gap analysis on `query` and return
    the LLM's text output. Single LLM call.

    All tunables (top-k, ranking strategy, generation parameters,
    show_evidence) are read from `CONFIG` at call time. Mutate `CONFIG`
    from a notebook cell to experiment without restarting the runtime.
    Set environment variables (RANKING_STRATEGY, etc.) for the highest-
    precedence override.

    Args:
        query: the compliance question.
        use_cache: read/write the LLM response cache.

    Caches LLM responses on (rendered_prompt, model_id) so re-runs are fast.
    Cache key naturally diverges across ranking strategies because the
    rendered prompt depends on which chunks are selected.
    """
    # Read all hot-mutable config values at call time so cell-driven
    # CONFIG mutations and env-var overrides take effect on the next call.
    top_k_reg     = int(_get(["retrieval", "top_k_reg"]))
    top_k_dep     = int(_get(["retrieval", "top_k_dep"]))
    top_k_initial = int(_get(["retrieval", "top_k_initial"]))
    strategy      = _get(["ranking", "strategy"], env_var="RANKING_STRATEGY")
    show_evidence = bool(_get(["output", "show_evidence"]))
    max_new_tokens     = int(_get(["llm", "generation", "max_new_tokens"]))
    repetition_penalty = float(_get(["llm", "generation", "repetition_penalty"]))

    retriever = _ensure_retriever()

    if strategy == "bge_only":
        # Single-stage retrieval — directly take BGE top_k. Build evidence
        # for transparency (rerank_score=None, no rank changes).
        reg_hits = retriever.retrieve(query, top_k=top_k_reg, corpus_filter="REG")
        dep_hits = retriever.retrieve(
            query, top_k=top_k_dep, corpus_filter=("DEP", "DEP_EXTRAS")
        )
        _, reg_evidence = _rerank_with_evidence(query, reg_hits, top_k_reg)
        _, dep_evidence = _rerank_with_evidence(query, dep_hits, top_k_dep)
    else:
        # Two-stage retrieval: wider BGE retrieve, then rerank/RRF to top_k.
        reg_initial = retriever.retrieve(
            query, top_k=top_k_initial, corpus_filter="REG"
        )
        dep_initial = retriever.retrieve(
            query, top_k=top_k_initial, corpus_filter=("DEP", "DEP_EXTRAS")
        )
        reg_hits, reg_evidence = _rerank_with_evidence(query, reg_initial, top_k_reg)
        dep_hits, dep_evidence = _rerank_with_evidence(query, dep_initial, top_k_dep)

    # Family-appropriate prompt pair. Gemma 3 gets a short role-only system
    # + rules-in-user; everything else gets V4 (role + 5 numbered rules in
    # system, question-only in user).
    system_prompt, user_msg = _get_prompts(
        LLM_MODEL_ID, query, _format_chunks(reg_hits), _format_chunks(dep_hits)
    )

    # Cache key uses the actual system_prompt for this run (not the module
    # SYSTEM_PROMPT constant) so Gemma and Qwen cache entries don't collide.
    # Rendered prompt depends on chunks → ranking strategies cache-separate.
    cache_key = system_prompt + "\n---\n" + user_msg

    if use_cache:
        cache = _ensure_cache()
        cached = cache.get(cache_key, LLM_MODEL_ID)
        if cached is not None:
            print(f"[simplified] Cache hit")
            return cached

    tokenizer, model = _ensure_llm()
    hint = "5-10s on GPU" if torch.cuda.is_available() else "20-30s on CPU"
    print(f"[simplified] Generating response (~{hint})...")
    t0 = time.time()
    output = _call_llm(
        tokenizer, model, system_prompt, user_msg,
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
    )
    print(f"[simplified]   generated in {time.time()-t0:.1f}s")

    if show_evidence:
        output = output.rstrip() + "\n\n" + _format_evidence(reg_evidence, dep_evidence)

    if use_cache:
        cache = _ensure_cache()
        cache.set(cache_key, LLM_MODEL_ID, output)

    return output
