"""Smoke tests for the simplified path.

Verifies module imports cleanly, public API is callable, the corpus
loads without ICO entries (Pass 2 cleanup), and the retrieval
grounding helpers produce the expected verbal labels at threshold
boundaries.

Does NOT call analyse() end-to-end — that requires loading the LLM
(~5 GB for Gemma 2-2B) and is impractical in CI. The full pipeline
is exercised via the Colab notebook (`colab/run_simplified_colab.ipynb`)
and `python -m src.ui.simple_chat`.
"""
from __future__ import annotations

import inspect

import pytest

from src.simplified import (
    USER_MESSAGE_TEMPLATE,
    EMBED_MODEL_ID,
    LLM_MODEL_ID,
    TOP_K_DEP,
    TOP_K_REG,
    _BGERetriever,
    _format_retrieval_grounding,
    _label,
    _pattern,
    analyse,
)


class TestPublicAPI:
    """The simplified path exposes a clean public API."""

    def test_analyse_signature(self):
        sig = inspect.signature(analyse)
        params = list(sig.parameters.keys())
        assert params == ["query", "use_cache"]

    def test_default_models(self):
        assert LLM_MODEL_ID == "google/gemma-2-2b-it"
        assert EMBED_MODEL_ID == "BAAI/bge-large-en-v1.5"

    def test_top_k_defaults(self):
        assert TOP_K_REG == 5
        assert TOP_K_DEP == 5

    def test_user_template_has_placeholders(self):
        assert "{query}" in USER_MESSAGE_TEMPLATE
        assert "{reg_text}" in USER_MESSAGE_TEMPLATE
        assert "{dep_text}" in USER_MESSAGE_TEMPLATE


class TestLabelClassifier:
    """_label bins mean cosine into strong / moderate / weak."""

    def test_strong_at_threshold(self):
        assert _label(0.70) == "strong"

    def test_strong_above(self):
        assert _label(0.85) == "strong"

    def test_moderate_at_threshold(self):
        assert _label(0.60) == "moderate"

    def test_moderate_below_strong(self):
        assert _label(0.69) == "moderate"

    def test_weak_below_moderate(self):
        assert _label(0.59) == "weak"

    def test_weak_at_zero(self):
        assert _label(0.0) == "weak"


class TestPatternLabels:
    """_pattern produces a plain-English interpretation per (law, policy) pair."""

    def test_both_strong(self):
        assert "well-grounded" in _pattern("strong", "strong")

    def test_strong_law_weak_policy(self):
        assert "silent" in _pattern("strong", "weak")

    def test_weak_law_strong_policy(self):
        assert "law side weak" in _pattern("weak", "strong")

    def test_both_weak(self):
        assert "low confidence" in _pattern("weak", "weak")


class TestEvidenceFooter:
    """_format_retrieval_grounding renders the footer with statistics + labels."""

    def _mock_hits(self, scores):
        from collections import namedtuple
        Chunk = namedtuple("Chunk", ["chunk_id", "section_reference", "chunk_text"])
        return [
            (Chunk(f"chunk-{i}", "ref", "text"), s)
            for i, s in enumerate(scores)
        ]

    def test_footer_contains_label_and_pattern(self):
        reg = self._mock_hits([0.787, 0.741, 0.731, 0.727, 0.719])
        dep = self._mock_hits([0.759, 0.737, 0.732, 0.721, 0.706])
        out = _format_retrieval_grounding(reg, dep)
        assert "strong" in out
        assert "well-grounded" in out
        assert "mean=" in out
        assert "max=" in out
        assert "min=" in out

    def test_footer_flags_weak_law_side(self):
        reg = self._mock_hits([0.574, 0.568, 0.560, 0.549, 0.547])
        dep = self._mock_hits([0.752, 0.739, 0.735, 0.729, 0.726])
        out = _format_retrieval_grounding(reg, dep)
        assert "weak" in out
        assert "law side weak" in out

    def test_footer_empty_for_empty_hits(self):
        assert _format_retrieval_grounding([], []) == ""


class TestCorpusLoads:
    """The corpus loads cleanly without ICO entries (Pass 2 cleanup)."""

    def test_no_ico_chunks_after_cleanup(self):
        from pathlib import Path
        from src.ingestion import chunk_corpus, load_corpus
        chunks = chunk_corpus(load_corpus(Path("corpus/manifest.json")))
        ico_chunks = [c for c in chunks if "operational/ico" in c.chunk_id]
        assert ico_chunks == [], (
            f"ICO chunks should have been dropped in Pass 2; "
            f"found {len(ico_chunks)}: {[c.chunk_id for c in ico_chunks[:3]]}"
        )

    def test_corpus_tags_only_reg_dep_dep_extras(self):
        from pathlib import Path
        from src.ingestion import chunk_corpus, load_corpus
        chunks = chunk_corpus(load_corpus(Path("corpus/manifest.json")))
        tags = {c.corpus_tag for c in chunks}
        assert tags == {"REG", "DEP", "DEP_EXTRAS"}, (
            f"After Pass 2, only REG/DEP/DEP_EXTRAS tags should remain; got {tags}"
        )
