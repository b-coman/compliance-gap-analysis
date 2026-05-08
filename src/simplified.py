"""Simplified compliance gap analysis path — single-call architecture.

This is the production path. Given a compliance question, it:
  1. Retrieves the top-5 most relevant law passages (BGE-large bi-encoder)
  2. Retrieves the top-5 most relevant policy passages (same retriever)
  3. Constructs a user message with rules + question + retrieved passages
  4. Calls the LLM once
  5. Returns a 3-section text response (Law / Policy / Gap)

Empirical evidence for each design choice is documented in
`docs/evaluation-findings.md` and `docs/decisions.md`.

Deliberately not included (tested and simplified out for production):
  - Cross-encoder reranking (Stage 9 — improved Q4 only; the additional
    code surface and the second model dependency were not justified by
    the marginal gain on the test set)
  - Multi-step chain decomposition (Stage 7 — the chain produced a
    20-row register with documented architectural failure modes)
  - Multi-family prompt dispatch (Stage 8 extended — single-model
    deployment for the production path)
  - Configuration files / env-var overrides (single deployment target,
    constants are clearer to read and review than indirect config lookup)
  - LLM response cache key on system + user separately (single user-turn
    template, so the user message itself is the cache key)

For empirical comparisons across models / prompts / ranking strategies,
see `docs/test-passes/`. Those test passes used a richer code surface
that this production path has been simplified away from.

Isolation invariant: this module imports only from `src.ingestion`
(corpus loading), `src.llm.cache` (general-purpose disk cache utility),
and external libraries. It does NOT import from src.chain, src.schema,
or src.llm.* (other than cache).
"""
from __future__ import annotations

import contextlib
import io
import logging
import os
import sys
import warnings

# Silence verbose loading from HF Hub / transformers / sentence-transformers
# so the demo output stays focused on system status and the LLM response.
# Our own [simplified] status prints remain visible. Real errors propagate
# as exceptions; only routine info-level chatter is suppressed.
#
# Env vars must be set BEFORE importing transformers / sentence_transformers.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.ingestion import chunk_corpus, load_corpus
from src.llm.cache import DiskCache


@contextlib.contextmanager
def _quiet_load():
    """Suppress stdout/stderr during third-party model loading.

    sentence-transformers' BertModel LOAD REPORT and some HuggingFace Hub
    status lines are emitted via direct print() / warnings.warn() rather
    than Python logging, so the logger-level config above doesn't catch
    them. This context manager redirects both streams to discard buffers
    during the heavy load operations. Real errors still propagate as
    exceptions; only printed chatter is silenced.
    """
    saved_out, saved_err = sys.stdout, sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = saved_out
        sys.stderr = saved_err


# ───── Configuration (project defaults) ─────

# BGE-large empirically outperforms the class default (multi-qa-MiniLM-L6-cos-v1)
# on legal queries. See evaluation-findings.md Stage 5 for the comparison.
EMBED_MODEL_ID = "BAAI/bge-large-en-v1.5"

# BGE's recommended query-instruction prefix; pre-pended to the user query
# before encoding so the model interprets the input as a search query.
EMBED_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Local LLM. Gemma 2-2B is the model used in the INST0100 course context;
# it fits CPU and small GPUs cleanly. Gemma is gated on HuggingFace, so a
# valid HF_TOKEN is required at runtime.
LLM_MODEL_ID = "google/gemma-2-2b-it"

# Cache directories on disk
EMBED_CACHE_DIR = Path("embeddings_bge")
LLM_CACHE_DIR = Path("llm_cache_simplified")

# Retrieval depth: how many chunks per corpus side reach the LLM.
TOP_K_REG = 5  # regulation passages
TOP_K_DEP = 5  # deployer policy passages

# Maximum length of any chunk text injected into the prompt; prevents the
# combined prompt from exceeding Gemma 2's 8K context window.
SNIPPET_CHARS_LIMIT = 400

# Generation parameters; greedy decoding for reproducibility.
# See decisions.md "Greedy (deterministic) decoding for the simplified
# architecture's LLM call" for the rationale.
MAX_NEW_TOKENS = 550
REPETITION_PENALTY = 1.05


# ───── Prompt template ─────
#
# Gemma 2's chat template does not support a `system` role — only `user` and
# `model`. We merge the rules into the user turn instead. This is the
# Gemma-recommended pattern, validated empirically against the alternative
# (long-system-prompt) in docs/test-passes/v4-gemma-3-4b-colab.md § "Run B".

USER_MESSAGE_TEMPLATE = """Follow these rules in your response:
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

Output your assessment in three Markdown sections:

### What the law requires
(one sentence stating the specific obligation, beginning with the relevant law chunk_id in square brackets)

### What the policy says
(one sentence describing what the policy says about this obligation; cite policy chunk_ids if relevant; if the policy does not address the obligation, write a complete sentence stating which specific obligation is missing in plain English)

### Gap
(one sentence directly stating the gap between the law and the policy, citing the relevant law chunk_id in square brackets)
"""


# ───── BGE retriever ─────

class _BGERetriever:
    """Single-purpose BGE-large retriever.

    Embeds all corpus chunks once at startup, then ranks them against query
    embeddings via cosine similarity. Vectors are L2-normalised, so cosine
    similarity equals the dot product (which is what `@` computes).
    """

    def __init__(self, chunks, model: SentenceTransformer, embeddings: np.ndarray):
        self.chunks = chunks
        self._model = model
        self._embeddings = embeddings  # L2-normalised, shape (n_chunks, 1024)

    def retrieve(self, query: str, top_k: int,
                 corpus_filter: str | Sequence[str] | None = None
                 ) -> list[tuple]:
        """Return the top_k chunks most similar to the query, optionally
        filtered to specific corpus tags ('REG', 'DEP', 'DEP_EXTRAS').
        """
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


def _format_chunks(hits) -> str:
    """Render retrieved chunks for prompt injection. Truncates each chunk to
    SNIPPET_CHARS_LIMIT to keep the combined prompt within the model's
    context window.
    """
    lines = []
    for chunk, _score in hits:
        text = chunk.chunk_text
        if len(text) > SNIPPET_CHARS_LIMIT:
            text = text[:SNIPPET_CHARS_LIMIT] + "…"
        lines.append(f"[{chunk.chunk_id}] {chunk.section_reference}\n{text}")
    return "\n\n".join(lines)


# ───── Retrieval grounding (transparency footer appended to LLM output) ─────
#
# We summarise the top-K retrieved chunks with the three statistics taught
# in the INST0100 class example (mean / max / min — see Task 2 of the class
# notebook), applied here at chunk level over the top-K pool. We then bin
# the mean into a verbal confidence label per side and produce a plain-
# English pattern interpretation. Thresholds are empirically calibrated
# against the 5-query test set in docs/test-passes/.
#
# The labels measure retrieval-pool confidence, not output correctness.
# Cases where retrieval is confident but the output fails (architectural
# limits, LLM reasoning errors) are documented in
# docs/evaluation-findings.md. The footer is a transparency signal, not
# a correctness oracle.

LABEL_THRESHOLD_STRONG = 0.70    # mean cosine ≥ this → "strong"
LABEL_THRESHOLD_MODERATE = 0.60  # mean cosine ≥ this → "moderate", below → "weak"


def _label(mean_score: float) -> str:
    """Classify retrieval confidence based on mean cosine across top-K chunks."""
    if mean_score >= LABEL_THRESHOLD_STRONG:
        return "strong"
    if mean_score >= LABEL_THRESHOLD_MODERATE:
        return "moderate"
    return "weak"


def _pattern(reg_label: str, dep_label: str) -> str:
    """Map (law confidence, policy confidence) to a plain-English interpretation."""
    reg_high = reg_label != "weak"
    dep_high = dep_label != "weak"
    if reg_high and dep_high:
        return "well-grounded on both sides."
    if reg_high and not dep_high:
        return "policy may be silent on this obligation."
    if not reg_high and dep_high:
        return "law side weak — query may not match the law corpus well."
    return "low confidence on both sides — consider rephrasing the query."


def _format_retrieval_grounding(reg_hits: list, dep_hits: list) -> str:
    """Render the retrieval grounding footer with mean / max / min summary
    statistics plus verbal confidence labels per side and a pattern
    interpretation. Appended to the LLM output by analyse().
    """
    if not reg_hits or not dep_hits:
        return ""

    reg_scores = [s for _c, s in reg_hits]
    dep_scores = [s for _c, s in dep_hits]
    reg_mean = sum(reg_scores) / len(reg_scores)
    dep_mean = sum(dep_scores) / len(dep_scores)

    reg_label = _label(reg_mean)
    dep_label = _label(dep_mean)
    pattern = _pattern(reg_label, dep_label)

    return (
        f"\n\n---\n"
        f"Retrieval grounding (based on BGE cosine across top-{len(reg_scores)} chunks):\n"
        f"  Law passages:    {reg_label:<8} "
        f"(mean={reg_mean:.2f}, max={max(reg_scores):.2f}, min={min(reg_scores):.2f})\n"
        f"  Policy passages: {dep_label:<8} "
        f"(mean={dep_mean:.2f}, max={max(dep_scores):.2f}, min={min(dep_scores):.2f})\n"
        f"  Pattern: {pattern}\n"
    )


# ───── Module-level singletons ─────
# Loaded once per process; subsequent calls reuse the same instances.

_retriever: _BGERetriever | None = None
_llm_tokenizer = None
_llm_model = None
_llm_cache: DiskCache | None = None


def _device() -> str:
    """Pick the best available device. CUDA on Colab/Linux GPUs, CPU otherwise."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _ensure_retriever() -> _BGERetriever:
    """Build the retriever once per process. Embeddings are persisted to
    `embeddings_bge/embeddings.npy` so subsequent runs skip the embedding
    step (~30s on first run, near-instant after).
    """
    global _retriever
    if _retriever is not None:
        return _retriever

    print("[simplified] Loading corpus chunks...")
    chunks = chunk_corpus(load_corpus(Path("corpus/manifest.json")))

    print(f"[simplified] Loading BGE model ({EMBED_MODEL_ID}) on {_device()}...")
    with _quiet_load():
        bge_model = SentenceTransformer(EMBED_MODEL_ID, device=_device())

    cache_file = EMBED_CACHE_DIR / "embeddings.npy"
    if cache_file.exists():
        cached = np.load(cache_file)
        if cached.shape[0] == len(chunks):
            print(f"[simplified] Loading cached BGE embeddings from {cache_file}")
            embeddings = cached
        else:
            print(
                f"[simplified] Cache mismatch ({cached.shape[0]} cached vs "
                f"{len(chunks)} chunks); re-embedding..."
            )
            cache_file.unlink()
            embeddings = None
    else:
        embeddings = None

    if embeddings is None:
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
    """Load the LLM once per process. Standard transformers loading.
    fp16 on CUDA, fp32 on CPU (Gemma 2-2B fits both cleanly).
    """
    global _llm_tokenizer, _llm_model
    if _llm_model is not None:
        return _llm_tokenizer, _llm_model

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[simplified] Loading LLM ({LLM_MODEL_ID})...")
    t0 = time.time()
    with _quiet_load():
        _llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
        if torch.cuda.is_available():
            _llm_model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_ID, dtype=torch.float16
            ).to("cuda")
        else:
            _llm_model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_ID, dtype=torch.float32
            ).to("cpu")
    print(f"[simplified]   loaded in {time.time()-t0:.1f}s")
    return _llm_tokenizer, _llm_model


def _ensure_cache() -> DiskCache:
    """Lazy-load the LLM response cache. Keyed on (rendered_user_message,
    model_id), so identical queries return identical cached outputs.
    """
    global _llm_cache
    if _llm_cache is None:
        _llm_cache = DiskCache(cache_dir=LLM_CACHE_DIR)
    return _llm_cache


def _call_llm(tokenizer, model, user_msg: str) -> str:
    """Single LLM call with greedy decoding.

    Uses Gemma 2's chat template (single user turn, no system role).
    Returns the model's generated text only — the prompt portion is
    sliced off via input_ids length.
    """
    messages = [{"role": "user", "content": user_msg}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=REPETITION_PENALTY,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


# ───── Public API ─────

def analyse(query: str, *, use_cache: bool = True) -> str:
    """Run compliance gap analysis on `query`. Returns a 3-section text
    response with a retrieval-grounding footer appended.

    Single LLM call, single retrieval pass:
      1. Retrieve top-5 regulation chunks (corpus_tag == "REG")
      2. Retrieve top-5 deployer chunks (corpus_tag in {"DEP", "DEP_EXTRAS"})
      3. Build user message with rules + question + chunks
      4. Call LLM once
      5. Append retrieval grounding footer (mean/max/min + label + pattern)
      6. Return combined text

    Caches the LLM response (not the footer) on (rendered_prompt, model_id)
    so re-runs of the same query are sub-second. The footer is recomputed
    each call from the retrieval scores — both LLM output and footer are
    deterministic given the same query.
    """
    retriever = _ensure_retriever()
    reg_hits = retriever.retrieve(query, top_k=TOP_K_REG, corpus_filter="REG")
    dep_hits = retriever.retrieve(
        query, top_k=TOP_K_DEP, corpus_filter=("DEP", "DEP_EXTRAS")
    )

    user_msg = USER_MESSAGE_TEMPLATE.format(
        query=query,
        reg_text=_format_chunks(reg_hits),
        dep_text=_format_chunks(dep_hits),
    )

    cached_llm_output = None
    if use_cache:
        cache = _ensure_cache()
        cached_llm_output = cache.get(user_msg, LLM_MODEL_ID)

    if cached_llm_output is not None:
        print("[simplified] Cache hit")
        llm_output = cached_llm_output
    else:
        tokenizer, model = _ensure_llm()
        print("[simplified] Generating response...")
        t0 = time.time()
        llm_output = _call_llm(tokenizer, model, user_msg)
        print(f"[simplified]   generated in {time.time()-t0:.1f}s")
        if use_cache:
            cache.set(user_msg, LLM_MODEL_ID, llm_output)

    return llm_output + _format_retrieval_grounding(reg_hits, dep_hits)
