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
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# Shared / general-purpose imports only — no chain or chain-LLM imports
from src.retrieval import build_retriever
from src.llm.cache import DiskCache


# ───── Configuration ─────

EMBED_MODEL_ID = "BAAI/bge-large-en-v1.5"
EMBED_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
EMBED_CACHE_DIR = Path("embeddings_bge")

# Default LLM. Override via the MODEL_ID environment variable when running on
# Colab GPU (e.g. MODEL_ID=Qwen/Qwen2.5-7B-Instruct). Local CPU stays on the
# 1.5B default; bigger models OOM or run unusably slowly on consumer CPUs.
LLM_MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")

LLM_CACHE_DIR = Path("llm_cache_simplified")

DEFAULT_TOP_K_REG = 5
DEFAULT_TOP_K_DEP = 5
SNIPPET_CHARS_LIMIT = 400

DEFAULT_MAX_NEW_TOKENS = 550
DEFAULT_REPETITION_PENALTY = 1.05


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


# ───── Chunk formatting for the prompt ─────

def _format_chunks(hits, snippet_chars: int = SNIPPET_CHARS_LIMIT) -> str:
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
                      max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
                      repetition_penalty: float = DEFAULT_REPETITION_PENALTY) -> str:
    """Default inference path: Qwen, Mistral, Gemma 1/2, etc.

    Gemma 1 and Gemma 2 chat templates do not support a `system` role
    (only `user` and `model`); passing one drops it silently or produces a
    malformed prompt. Detect Gemma and merge system content into the user
    turn. Gemma 3 has its own dedicated path (handles system natively via
    content-list format) and never reaches this function.
    """
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
                     max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
                     repetition_penalty: float = DEFAULT_REPETITION_PENALTY) -> str:
    """Gemma 3 inference path.

    Gemma 3's chat template expects content as a list of `{"type": "text",
    "text": ...}` dicts (the multimodal content envelope). Plain strings
    silently produce malformed prompts and empty generation. Uses
    `processor.apply_chat_template` rather than `tokenizer.apply_chat_template`,
    with `tokenize=True, return_dict=True` to get input_ids+attention_mask
    in one call. Stop tokens are handled by the model's generation_config
    (Gemma uses `<end_of_turn>`).
    """
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
              max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
              repetition_penalty: float = DEFAULT_REPETITION_PENALTY) -> str:
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


# ───── Public API ─────

def analyse(query: str, *,
            top_k_reg: int = DEFAULT_TOP_K_REG,
            top_k_dep: int = DEFAULT_TOP_K_DEP,
            use_cache: bool = True) -> str:
    """Run the simplified compliance gap analysis on `query` and return
    the LLM's text output. Single LLM call.

    Caches LLM responses on (rendered_prompt, model_id) so re-runs are fast.
    """
    retriever = _ensure_retriever()
    reg_hits = retriever.retrieve(query, top_k=top_k_reg, corpus_filter="REG")
    dep_hits = retriever.retrieve(
        query, top_k=top_k_dep, corpus_filter=("DEP", "DEP_EXTRAS")
    )
    user_msg = _user_message(query, _format_chunks(reg_hits), _format_chunks(dep_hits))

    if use_cache:
        cache = _ensure_cache()
        # Combine system + user as the cache key prompt
        cache_key = SYSTEM_PROMPT + "\n---\n" + user_msg
        cached = cache.get(cache_key, LLM_MODEL_ID)
        if cached is not None:
            print(f"[simplified] Cache hit")
            return cached

    tokenizer, model = _ensure_llm()
    hint = "5-10s on GPU" if torch.cuda.is_available() else "20-30s on CPU"
    print(f"[simplified] Generating response (~{hint})...")
    t0 = time.time()
    output = _call_llm(tokenizer, model, SYSTEM_PROMPT, user_msg)
    print(f"[simplified]   generated in {time.time()-t0:.1f}s")

    if use_cache:
        cache = _ensure_cache()
        cache_key = SYSTEM_PROMPT + "\n---\n" + user_msg
        cache.set(cache_key, LLM_MODEL_ID, output)

    return output
