"""ING-01 + ING-02 unit + integration tests."""

from __future__ import annotations

import re
import time
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

import numpy as np

from src.ingestion import (
    Chunk,
    Document,
    _cache_path,
    _chunk_ai_act,
    _chunk_id,
    _chunk_novara_extras,
    _chunk_novara_policy,
    _derive_corpus_tag,
    _derive_document_id,
    _derive_section_reference,
    _estimate_tokens,
    _load_cache,
    _save_cache,
    _section_chunk_id,
    _sentences,
    _strip_page_furniture,
    chunk_corpus,
    embed_chunks,
    load_corpus,
)


# === ING-01 unit tests ====================================================


@pytest.mark.parametrize("path,expected", [
    ("regulation/uk-gdpr-art-22.txt", "REG"),
    ("regulation/eu-ai-act-2024-1689.txt", "REG"),
    ("deployer/novara-ai-policy-v3.1.txt", "DEP"),
    ("deployer-extras/novara-talentlens-dpia.md", "DEP_EXTRAS"),
])
def test_derive_corpus_tag_for_each_bucket(path: str, expected: str) -> None:
    assert _derive_corpus_tag(path) == expected


def test_derive_corpus_tag_rejects_unknown_bucket() -> None:
    with pytest.raises(ValueError):
        _derive_corpus_tag("unknown/some-file.txt")


def test_derive_document_id_strips_extension() -> None:
    assert _derive_document_id("regulation/uk-gdpr-art-22.txt") == "uk-gdpr-art-22"
    assert _derive_document_id("deployer-extras/novara-talentlens-dpia.md") == "novara-talentlens-dpia"
    assert _derive_document_id("deployer/novara-ai-policy-v3.1.txt") == "novara-ai-policy-v3.1"


@pytest.mark.parametrize("path,expected", [
    ("regulation/uk-gdpr-art-22.txt", "UK GDPR Article 22"),
    ("regulation/uk-gdpr-art-5.txt", "UK GDPR Article 5"),
    ("regulation/eu-ai-act-2024-1689.txt", "EU AI Act (Regulation 2024/1689)"),
    ("deployer/novara-ai-policy-v3.1.txt", "Novara AI Policy v3.1"),
    ("deployer-extras/novara-talentlens-dpia.md", "Novara TalentLens DPIA"),
])
def test_derive_section_reference_table(path: str, expected: str) -> None:
    assert _derive_section_reference(path) == expected


def test_chunk_id_is_deterministic() -> None:
    assert _chunk_id("regulation/uk-gdpr-art-22.txt") == _chunk_id("regulation/uk-gdpr-art-22.txt")


def test_chunk_id_strips_extension() -> None:
    assert _chunk_id("regulation/uk-gdpr-art-22.txt") == "regulation/uk-gdpr-art-22"
    assert _chunk_id("deployer-extras/novara-talentlens-dpia.md") == "deployer-extras/novara-talentlens-dpia"


def test_document_dataclass_is_frozen() -> None:
    doc = Document(
        chunk_id="REG:x", corpus_tag="REG", document_id="x",
        section_reference="X", source_url="", chunk_text="content",
        file_path="regulation/x.txt", sha256_short="abc",
    )
    with pytest.raises(FrozenInstanceError):
        doc.chunk_text = "different"  # type: ignore[misc]


# === ING-01 integration tests ============================================


def test_load_corpus_returns_15_documents(corpus_documents) -> None:
    # Pass 2 cleanup dropped 26 ICO operational entries; the redundant
    # uk-gdpr-articles-relevant.txt was removed in a follow-up. Remaining:
    # 15 = 9 REG (8 GDPR articles + EU AI Act) + 1 DEP + 5 DEP_EXTRAS
    # (44 - 26 - 1 - 2 PDF metadata entries).
    assert len(corpus_documents) == 15


def test_every_document_has_required_fields_populated(corpus_documents) -> None:
    for d in corpus_documents:
        assert d.chunk_id and d.corpus_tag and d.document_id
        assert d.section_reference and d.source_url and d.chunk_text
        assert d.file_path and d.sha256_short


def test_every_document_has_one_of_three_corpus_tags(corpus_documents) -> None:
    # OPS bucket dropped in Pass 2 cleanup (ICO operational guidance evicted).
    valid = {"REG", "DEP", "DEP_EXTRAS"}
    for d in corpus_documents:
        assert d.corpus_tag in valid


def test_corpus_tag_distribution(corpus_documents) -> None:
    counts: dict[str, int] = {}
    for d in corpus_documents:
        counts[d.corpus_tag] = counts.get(d.corpus_tag, 0) + 1
    assert counts == {"REG": 9, "DEP": 1, "DEP_EXTRAS": 5}


def test_document_ids_are_unique_across_corpus(corpus_documents) -> None:
    ids = [d.chunk_id for d in corpus_documents]
    assert len(ids) == len(set(ids))


def test_load_corpus_is_idempotent(corpus_manifest_path) -> None:
    a = load_corpus(corpus_manifest_path)
    b = load_corpus(corpus_manifest_path)
    assert [d.chunk_id for d in a] == [d.chunk_id for d in b]


def test_document_text_meets_minimum_word_floor(corpus_documents) -> None:
    for d in corpus_documents:
        assert len(d.chunk_text.split()) >= 50


def test_load_corpus_completes_in_under_60s(corpus_manifest_path) -> None:
    start = time.perf_counter()
    docs = load_corpus(corpus_manifest_path)
    assert (time.perf_counter() - start) < 60.0
    assert docs


# === ING-02 unit tests ====================================================


def test_estimate_tokens_returns_int() -> None:
    assert _estimate_tokens("") == 0
    assert _estimate_tokens("a" * 35) == 10  # 35 / 3.5 = 10
    assert isinstance(_estimate_tokens("hello world"), int)


def test_strip_page_furniture_removes_known_noise() -> None:
    raw = (
        "PE-CONS 24/24    AD/DOS    TREE.2.B    EN\n"
        "actual body line\n"
        "TREE.2.B\n"
        "another body line\n"
        "EN\n"
        "42\n"
        "    Article 1\n"
    )
    cleaned = _strip_page_furniture(raw)
    assert "PE-CONS" not in cleaned
    assert "TREE.2.B" not in cleaned
    assert "actual body line" in cleaned
    assert "another body line" in cleaned
    assert "Article 1" in cleaned


def test_section_chunk_id_concatenates_path_and_anchor() -> None:
    assert _section_chunk_id("regulation/uk-gdpr-art-22", "para-3") == "regulation/uk-gdpr-art-22#para-3"


def test_sentences_returns_tuple() -> None:
    out = _sentences("First sentence. Second sentence. Third one.")
    assert isinstance(out, tuple)
    assert len(out) == 3


def test_chunk_ai_act_synthetic() -> None:
    raw = (
        "Whereas (1) recital text\n"
        "PE-CONS 24/24    EN\n"
        "                                          Article 1\n"
        "Subject matter\n"
        "1. This Regulation lays down rules.\n"
        "PE-CONS 24/24    EN\n"
        "                                          Article 2\n"
        "Scope\n"
        "1. This Regulation applies to providers.\n"
    )
    out = _chunk_ai_act(raw)
    labels = [label for _anchor, label, _body in out]
    assert "EU AI Act Article 1" in labels
    assert "EU AI Act Article 2" in labels
    # Recital is skipped (everything before the first "Article 1" line).
    bodies = " ".join(b for _a, _l, b in out)
    assert "recital text" not in bodies
    assert "PE-CONS" not in bodies


def test_novara_policy_section_regex_rejects_table_cells() -> None:
    # The naïve `^(\d+(?:\.\d+)?)\s+(.+)$` would match "30 days" (a table
    # cell in §4.4 retention table). The tightened regex requires a
    # capital-letter-led title with minimum length.
    text = "3.1 Model Selection and Procurement\nbody body body\n30 days\nmore body\n"
    fake_doc = Document(
        chunk_id="deployer/novara-ai-policy-v3.1",
        corpus_tag="DEP",
        document_id="novara-ai-policy-v3.1",
        section_reference="Novara AI Policy v3.1",
        source_url="",
        chunk_text=text,
        file_path="deployer/novara-ai-policy-v3.1.txt",
        sha256_short="abc",
    )
    out = _chunk_novara_policy(fake_doc)
    labels = [label for _a, label, _b in out]
    assert any("§3.1 Model Selection and Procurement" in label for label in labels)
    assert not any("30 days" in label for label in labels)


def test_novara_extras_emits_section_chunks() -> None:
    text = (
        "# Document Title\n\n"
        "leading preamble enough to count as content with at least fifty words "
        "yes lots of words here filler filler filler filler filler filler filler "
        "filler filler filler filler filler filler filler filler filler filler\n\n"
        "## 1. First Section\n\n"
        "first section body\n\n"
        "## 2. Second Section\n\n"
        "second section body\n"
    )
    fake_doc = Document(
        chunk_id="deployer-extras/novara-talentlens-dpia",
        corpus_tag="DEP_EXTRAS",
        document_id="novara-talentlens-dpia",
        section_reference="Novara TalentLens DPIA",
        source_url="",
        chunk_text=text,
        file_path="deployer-extras/novara-talentlens-dpia.md",
        sha256_short="abc",
    )
    out = _chunk_novara_extras(fake_doc)
    labels = [label for _a, label, _b in out]
    assert any("First Section" in label for label in labels)
    assert any("Second Section" in label for label in labels)


# === ING-02 integration tests ============================================


def test_chunk_corpus_total_count_in_estimated_range(corpus_chunks) -> None:
    assert 300 <= len(corpus_chunks) <= 1500


def test_chunk_corpus_per_bucket_distribution_sensible(corpus_chunks) -> None:
    counts: dict[str, int] = {}
    for c in corpus_chunks:
        counts[c.corpus_tag] = counts.get(c.corpus_tag, 0) + 1
    # Three buckets after Pass 2 cleanup (OPS dropped along with ICO corpus).
    assert counts.get("REG", 0) >= 10
    assert counts.get("DEP", 0) >= 5
    assert counts.get("DEP_EXTRAS", 0) >= 10


def test_no_chunk_straddles_two_articles(corpus_chunks) -> None:
    # AI Act chunks should not contain a second "Article N" boundary inside
    # their body. (Minor occurrences of 'Article 22' as inline references
    # are fine — what's forbidden is a *boundary line*: 10+ leading spaces
    # then "Article N".)
    boundary = re.compile(r"\n\s{10,}Article\s+\d+\s*$", re.MULTILINE)
    for c in corpus_chunks:
        if c.document_id == "eu-ai-act-2024-1689":
            assert not boundary.search(c.chunk_text), f"boundary inside {c.chunk_id}"


def test_no_page_furniture_in_chunk_text(corpus_chunks) -> None:
    for c in corpus_chunks:
        if c.document_id == "eu-ai-act-2024-1689":
            assert "PE-CONS" not in c.chunk_text, f"page furniture in {c.chunk_id}"
            assert "TREE.2.B" not in c.chunk_text, f"page furniture in {c.chunk_id}"


def test_recitals_are_skipped(corpus_chunks) -> None:
    # Recitals are the "Whereas (1) ..." paragraphs preceding Article 1.
    # No AI Act chunk should contain "Whereas:" header. (Substring 'Whereas'
    # could legitimately appear in operative text; we test the recital
    # header form specifically.)
    for c in corpus_chunks:
        if c.document_id == "eu-ai-act-2024-1689":
            assert "Whereas:" not in c.chunk_text, f"recital header in {c.chunk_id}"


def test_ai_act_article_27_is_a_distinct_chunk(corpus_chunks) -> None:
    # Q5 demo target — must be retrievable as its own chunk.
    matches = [c for c in corpus_chunks if c.section_reference.startswith("EU AI Act Article 27")]
    assert len(matches) >= 1


def test_gdpr_article_22_is_a_distinct_chunk(corpus_chunks) -> None:
    matches = [c for c in corpus_chunks if c.document_id == "uk-gdpr-art-22"]
    assert len(matches) >= 1


def test_novara_policy_section_3_4_is_a_distinct_chunk(corpus_chunks) -> None:
    # Q2 demo target — red-teaming.
    matches = [c for c in corpus_chunks
               if c.document_id == "novara-ai-policy-v3.1"
               and "§3.4" in c.section_reference]
    assert len(matches) == 1, f"expected 1 chunk, got {len(matches)}: {[m.section_reference for m in matches]}"


def test_every_chunk_has_required_fields_populated(corpus_chunks) -> None:
    for c in corpus_chunks:
        assert c.chunk_id and c.parent_document_id and c.corpus_tag
        assert c.document_id and c.section_reference
        assert c.chunk_text and c.file_path and c.sha256_short
        assert isinstance(c.sentences, tuple)
        assert len(c.sentences) >= 1


def test_parent_document_id_links_to_real_document(corpus_documents, corpus_chunks) -> None:
    doc_ids = {d.chunk_id for d in corpus_documents}
    for c in corpus_chunks:
        assert c.parent_document_id in doc_ids, f"orphan chunk {c.chunk_id}"


def test_chunk_ids_are_unique_across_corpus(corpus_chunks) -> None:
    ids = [c.chunk_id for c in corpus_chunks]
    assert len(ids) == len(set(ids))


def test_chunk_corpus_is_idempotent(corpus_documents) -> None:
    a = chunk_corpus(corpus_documents)
    b = chunk_corpus(corpus_documents)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert [c.chunk_text for c in a] == [c.chunk_text for c in b]


def test_chunk_size_distribution_sane(corpus_chunks) -> None:
    sizes = [_estimate_tokens(c.chunk_text) for c in corpus_chunks]
    # Sanity floor on outliers — most chunks should sit in [50, 1500].
    inside = sum(1 for s in sizes if 50 <= s <= 1500)
    fraction = inside / len(sizes)
    assert fraction >= 0.85, f"only {fraction:.2%} of chunks within [50, 1500] tokens"


def test_chunk_corpus_completes_in_under_30s(corpus_documents) -> None:
    start = time.perf_counter()
    chunk_corpus(corpus_documents)
    assert (time.perf_counter() - start) < 30.0


def test_chunk_dataclass_is_frozen() -> None:
    chunk = Chunk(
        chunk_id="x", parent_document_id="x-parent",
        corpus_tag="REG", document_id="x", section_reference="X",
        source_url="", chunk_text="content",
        file_path="regulation/x.txt", sha256_short="abc",
        sentences=("only one.",),
    )
    with pytest.raises(FrozenInstanceError):
        chunk.chunk_text = "different"  # type: ignore[misc]


# === ING-03 unit tests ====================================================


def _make_chunk(chunk_id: str, text: str = "filler text content body") -> Chunk:
    return Chunk(
        chunk_id=chunk_id, parent_document_id=chunk_id,
        corpus_tag="REG", document_id=chunk_id, section_reference=chunk_id,
        source_url="", chunk_text=text,
        file_path=f"regulation/{chunk_id}.txt", sha256_short="abc",
        sentences=(text,),
    )


def test_cache_path_format(tmp_path: Path) -> None:
    p = _cache_path(tmp_path, "multi-qa-MiniLM-L6-cos-v1")
    assert p == tmp_path / "multi-qa-MiniLM-L6-cos-v1.npz"


def test_save_then_load_cache_roundtrip(tmp_path: Path) -> None:
    embeddings = np.random.rand(3, 8).astype(np.float32)
    ids = ["a", "b", "c"]
    cache_path = tmp_path / "test.npz"
    _save_cache(cache_path, embeddings, ids)
    loaded = _load_cache(cache_path)
    assert loaded is not None
    np.testing.assert_array_equal(loaded[0], embeddings)
    assert loaded[1] == ids


def test_load_cache_returns_none_for_missing_file(tmp_path: Path) -> None:
    assert _load_cache(tmp_path / "nope.npz") is None


def test_load_cache_returns_none_for_corrupted_file(tmp_path: Path) -> None:
    bad = tmp_path / "bad.npz"
    bad.write_bytes(b"this is not a real npz file")
    assert _load_cache(bad) is None


def test_embed_chunks_empty_input(tmp_path: Path) -> None:
    embeddings, ids = embed_chunks([], cache_dir=tmp_path)
    assert embeddings.shape == (0, 0)
    assert ids == []


def test_embed_chunks_writes_cache_and_hits_on_re_run(tmp_path: Path,
                                                     corpus_chunks) -> None:
    # First call: cache miss + compute. Second call: cache hit.
    chunks_subset = corpus_chunks[:5]  # small subset to keep test fast
    cache_path = _cache_path(tmp_path, "multi-qa-MiniLM-L6-cos-v1")
    assert not cache_path.exists()

    e1, ids1 = embed_chunks(chunks_subset, cache_dir=tmp_path)
    assert cache_path.exists()
    assert e1.shape[0] == len(chunks_subset)

    # Second call should return identical arrays from cache.
    e2, ids2 = embed_chunks(chunks_subset, cache_dir=tmp_path)
    np.testing.assert_array_equal(e1, e2)
    assert ids1 == ids2


def test_embed_chunks_cache_invalidates_on_chunk_change(tmp_path: Path) -> None:
    a = [_make_chunk("a", "text one"), _make_chunk("b", "text two")]
    b = [_make_chunk("a", "text one"), _make_chunk("c", "text three")]  # different ids
    e_a, ids_a = embed_chunks(a, cache_dir=tmp_path)
    e_b, ids_b = embed_chunks(b, cache_dir=tmp_path)
    assert ids_a != ids_b
    assert e_a.shape[0] == 2 and e_b.shape[0] == 2


# === ING-03 integration tests (real corpus, session-scoped fixture) =======


def test_corpus_embeddings_shape(corpus_embeddings, corpus_chunks) -> None:
    embeddings, ids = corpus_embeddings
    assert embeddings.shape == (len(corpus_chunks), 384)


def test_corpus_embeddings_dtype_is_float32(corpus_embeddings) -> None:
    embeddings, _ids = corpus_embeddings
    assert embeddings.dtype == np.float32


def test_corpus_embeddings_chunk_ids_match_input_order(
    corpus_embeddings, corpus_chunks
) -> None:
    _e, ids = corpus_embeddings
    assert ids == [c.chunk_id for c in corpus_chunks]


def test_corpus_embeddings_are_l2_normalised(corpus_embeddings) -> None:
    # multi-qa-MiniLM-L6-cos-v1 outputs already-normalised vectors.
    embeddings, _ids = corpus_embeddings
    norms = np.linalg.norm(embeddings, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-3)


def test_corpus_embedding_runs_under_three_minutes(
    corpus_chunks, tmp_path: Path
) -> None:
    import time
    start = time.perf_counter()
    embed_chunks(corpus_chunks, cache_dir=tmp_path)
    assert (time.perf_counter() - start) < 180.0


def test_second_corpus_embedding_run_hits_cache(
    corpus_chunks, tmp_path: Path
) -> None:
    import time
    embed_chunks(corpus_chunks, cache_dir=tmp_path)  # warm-up
    start = time.perf_counter()
    embed_chunks(corpus_chunks, cache_dir=tmp_path)
    assert (time.perf_counter() - start) < 1.5  # cache hit, no re-embed


def test_cache_invalidation_with_different_model_name(
    corpus_chunks, tmp_path: Path, monkeypatch
) -> None:
    """Two different model names produce two different cache files."""
    # Stub _get_st_model to return a fake encoder so we don't download a
    # second real model. The fake's encode returns deterministic vectors
    # of dimension 16, distinct from the real 384-dim embeddings.
    class _FakeModel:
        def encode(self, texts, **kwargs):
            return np.array(
                [[hash((t, i)) % 100 / 100.0 for i in range(16)] for t in texts],
                dtype=np.float32,
            )

    real_get = None
    from src import ingestion

    def fake_get(name: str):
        if name == "multi-qa-MiniLM-L6-cos-v1":
            return real_get(name)
        return _FakeModel()

    real_get = ingestion._get_st_model
    monkeypatch.setattr(ingestion, "_get_st_model", fake_get)

    e1, _ = embed_chunks(corpus_chunks[:5], cache_dir=tmp_path,
                          model_name="multi-qa-MiniLM-L6-cos-v1")
    e2, _ = embed_chunks(corpus_chunks[:5], cache_dir=tmp_path,
                          model_name="fake-tiny-model")

    assert (tmp_path / "multi-qa-MiniLM-L6-cos-v1.npz").exists()
    assert (tmp_path / "fake-tiny-model.npz").exists()
    assert e1.shape != e2.shape  # 384 vs 16 — proves different cache content
