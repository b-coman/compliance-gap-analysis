"""Corpus ingestion — corpus loading, chunking, and embedding.

Loads the three-bucket corpus (regulation, Novara deployer policy,
Novara deployer-extras) from `corpus/manifest.json` into typed
Document records, then refines those into article/§/section-level
Chunks with per-chunk sentence breakdown.

Pre-conditions: `corpus/manifest.json` complete, hashes verified
(`scripts/validate_corpus.py`).

AI Act extraction observations: docs/ai-act-extraction-notes.md.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CORPUS_TAGS = ("REG", "DEP", "DEP_EXTRAS")

_STATIC_LABELS: dict[str, str] = {
    "regulation/eu-ai-act-2024-1689.txt": "EU AI Act (Regulation 2024/1689)",
    "deployer/novara-ai-policy-v3.1.txt": "Novara AI Policy v3.1",
    "deployer-extras/novara-talentlens-dpia.md": "Novara TalentLens DPIA",
    "deployer-extras/novara-talentlens-model-card.md": "Novara TalentLens Model Card",
    "deployer-extras/novara-talentlens-transparency-notice.md": "Novara TalentLens Transparency Notice",
    "deployer-extras/novara-talentlens-model-intake-assessment.md": "Novara TalentLens Model Intake Assessment",
    "deployer-extras/novara-2025-ai-governance-report.md": "Novara 2025 AI Governance Report",
}

# Regexes for ING-02 chunkers
_AI_ACT_NOISE = (
    re.compile(r"^.*PE-CONS\s*\d+/\d+.*$"),
    re.compile(r"^.*\bTREE\.\d+\.[A-Z]\b.*$"),  # matches mid-line too — e.g., "ANNEX I  TREE.2.B  EN" page headers
    re.compile(r"^\s*[A-Z]{2,4}\s*$"),
    re.compile(r"^\s*\d+\s*$"),
)
_AI_ACT_ARTICLE = re.compile(r"^\s{10,}Article\s+(\d+)\s*$")
_AI_ACT_ANNEX = re.compile(r"^\s*ANNEX\s+([IVX]+)\b")
_NUMBERED_PARA = re.compile(r"^\s*(\d+)\.\s+")
_NOVARA_POLICY_SECTION = re.compile(r"^(\d+(?:\.\d+)?)\s+([A-Z][a-zA-Z]{2,}.{4,})$")
_MD_HEADING = re.compile(r"^(#{1,3})\s+(.+)$")


# ----- ING-01 Document -----------------------------------------------------


@dataclass(frozen=True, slots=True)
class Document:
    chunk_id: str
    corpus_tag: str
    document_id: str
    section_reference: str
    source_url: str
    chunk_text: str
    file_path: str
    sha256_short: str


def _derive_corpus_tag(file_path: str) -> str:
    if file_path.startswith("regulation/"):
        return "REG"
    if file_path.startswith("deployer-extras/"):
        return "DEP_EXTRAS"
    if file_path.startswith("deployer/"):
        return "DEP"
    raise ValueError(f"unknown bucket for path: {file_path}")


def _derive_document_id(file_path: str) -> str:
    return Path(file_path).stem


def _derive_section_reference(file_path: str) -> str:
    if file_path in _STATIC_LABELS:
        return _STATIC_LABELS[file_path]

    stem = Path(file_path).stem

    if stem.startswith("uk-gdpr-art-"):
        return f"UK GDPR Article {stem.removeprefix('uk-gdpr-art-')}"

    return stem.replace("-", " ").title()


def _chunk_id(file_path: str) -> str:
    # Path with extension stripped. Globally unique by filesystem invariant.
    return str(Path(file_path).with_suffix(""))


def load_corpus(manifest_path: Path = Path("corpus/manifest.json")) -> list[Document]:
    """Load every text file in the manifest into a Document record.

    Skips manifest entries with `word_count == 0` (PDFs whose .txt
    siblings are the canonical retrieval source per v2 corpus spec § 2).
    Returns documents in manifest order; chunk_ids are stable across
    re-runs against the same manifest.
    """
    corpus_root = manifest_path.parent
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))

    documents: list[Document] = []
    for entry in entries:
        if entry["word_count"] == 0:
            # PDF entries are present for citation provenance only;
            # their .txt siblings are the canonical retrieval source.
            continue

        rel_path = entry["path"]
        text = (corpus_root / rel_path).read_text(encoding="utf-8")
        documents.append(Document(
            chunk_id=_chunk_id(rel_path),
            corpus_tag=_derive_corpus_tag(rel_path),
            document_id=_derive_document_id(rel_path),
            section_reference=_derive_section_reference(rel_path),
            source_url=entry["source_url"],
            chunk_text=text,
            file_path=rel_path,
            sha256_short=entry["sha256_short"],
        ))

    return documents


# ----- ING-02 Chunk + per-bucket chunkers ----------------------------------


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    parent_document_id: str
    corpus_tag: str
    document_id: str
    section_reference: str
    source_url: str
    chunk_text: str
    file_path: str
    sha256_short: str
    sentences: tuple[str, ...]


_NLP = None


def _get_sentencizer():
    """Lazy-init spaCy sentencizer per the week-7 RAG tutorial pattern."""
    global _NLP
    if _NLP is None:
        from spacy.lang.en import English
        _NLP = English()
        _NLP.add_pipe("sentencizer")
    return _NLP


def _sentences(text: str) -> tuple[str, ...]:
    if not text.strip():
        return ()
    nlp = _get_sentencizer()
    return tuple(s.text.strip() for s in nlp(text).sents if s.text.strip())


def _estimate_tokens(text: str) -> int:
    # Per docs/decisions.md §5: len // 3.5 is a reasonable English heuristic.
    return int(len(text) / 3.5)


def _strip_page_furniture(text: str) -> str:
    keep = []
    for line in text.splitlines():
        if any(p.match(line) for p in _AI_ACT_NOISE):
            continue
        keep.append(line)
    return "\n".join(keep)


def _section_chunk_id(parent_document_id: str, anchor: str) -> str:
    return f"{parent_document_id}#{anchor}"


def _split_long_article(body: str, max_tokens: int = 800) -> list[tuple[str, str]]:
    """Sub-split an over-long article body at numbered-paragraph boundaries.

    Returns [(anchor_suffix, sub_body), ...]. If the body is small enough or
    has no numbered boundaries, returns [("", body)]. Sub-anchors use
    sequential indexing ("para-1", "para-2", ...) rather than the matched
    paragraph number — annexes with Section A/B/C each carry their own
    "1.", "2." numbering, so the regex-matched number can collide; sequential
    keeps anchors unique within the parent body.
    """
    if _estimate_tokens(body) <= max_tokens:
        return [("", body)]
    lines = body.splitlines()
    paras: list[list[str]] = []
    current: list[str] = []
    has_started_paragraphing = False
    for line in lines:
        if _NUMBERED_PARA.match(line):
            if has_started_paragraphing:
                paras.append(current)
                current = [line]
            else:
                # First numbered paragraph absorbs preceding preamble lines
                # (article title, "Subject matter" line, etc.) — otherwise
                # the title becomes a tiny standalone chunk that wins retrieval
                # on title-keyword queries while carrying no obligation content.
                current.append(line)
                has_started_paragraphing = True
        else:
            current.append(line)
    if current:
        paras.append(current)
    if len(paras) <= 1:
        return [("", body)]
    return [(f"para-{i}", "\n".join(lns).strip())
            for i, lns in enumerate(paras, 1)
            if "\n".join(lns).strip()]


def _chunk_ai_act(text: str) -> list[tuple[str, str, str]]:
    """Split the AI Act text into [(anchor, label, body), ...].

    Drops page furniture, skips recitals (everything before Article 1),
    splits at Article and Annex boundaries, sub-splits over-long articles
    at numbered-paragraph boundaries.
    """
    cleaned = _strip_page_furniture(text)
    lines = cleaned.splitlines()

    # Find first "Article 1" — skip everything before (recitals + preamble).
    start = 0
    for i, line in enumerate(lines):
        m = _AI_ACT_ARTICLE.match(line)
        if m and m.group(1) == "1":
            start = i
            break
    else:
        return []

    # Walk articles and annexes.
    sections: list[tuple[str, str, list[str]]] = []  # (anchor, label, body lines)
    current_anchor = "article-1"
    current_label = "EU AI Act Article 1"
    current_body: list[str] = []
    seen_annexes: set[str] = set()

    for line in lines[start + 1:]:
        m_art = _AI_ACT_ARTICLE.match(line)
        m_anx = _AI_ACT_ANNEX.match(line)
        if m_art:
            sections.append((current_anchor, current_label, current_body))
            n = m_art.group(1)
            current_anchor = f"article-{n}"
            current_label = f"EU AI Act Article {n}"
            current_body = []
        elif m_anx and m_anx.group(1) not in seen_annexes:
            seen_annexes.add(m_anx.group(1))
            sections.append((current_anchor, current_label, current_body))
            roman = m_anx.group(1)
            current_anchor = f"annex-{roman.lower()}"
            current_label = f"EU AI Act Annex {roman}"
            current_body = []
        else:
            current_body.append(line)
    sections.append((current_anchor, current_label, current_body))

    # Materialise + sub-split long articles.
    out: list[tuple[str, str, str]] = []
    for anchor, label, body_lines in sections:
        body = "\n".join(body_lines).strip()
        if not body:
            continue
        sub = _split_long_article(body)
        if len(sub) == 1:
            out.append((anchor, label, body))
        else:
            for sub_anchor, sub_body in sub:
                out.append((f"{anchor}-{sub_anchor}", f"{label} ({sub_anchor})", sub_body))
    return out


def _chunk_gdpr_article(doc: Document) -> list[tuple[str, str, str]]:
    """Single chunk per GDPR per-article file. Article 5 is sub-split per
    principle paragraph if the file's structure supports it."""
    if doc.document_id == "uk-gdpr-art-5":
        # Art 5 has multiple principles; emit one chunk per non-empty paragraph.
        paras = [p.strip() for p in re.split(r"\n\s*\n", doc.chunk_text) if p.strip()]
        if len(paras) >= 2:
            return [(f"para-{i}", f"{doc.section_reference}({i})", p)
                    for i, p in enumerate(paras, 1)]
    return [("whole", doc.section_reference, doc.chunk_text.strip())]


def _chunk_novara_policy(doc: Document) -> list[tuple[str, str, str]]:
    """Split the Novara policy at numbered section/sub-section headings."""
    lines = doc.chunk_text.splitlines()
    sections: list[tuple[str, str, list[str]]] = []
    current_anchor = "preamble"
    current_label = f"{doc.section_reference} — Preamble"
    current: list[str] = []
    for line in lines:
        m = _NOVARA_POLICY_SECTION.match(line)
        if m:
            if current:
                sections.append((current_anchor, current_label, current))
            number, title = m.group(1), m.group(2).strip()
            current_anchor = f"section-{number.replace('.', '-')}"
            current_label = f"{doc.section_reference} §{number} {title}"
            current = []
        else:
            current.append(line)
    if current:
        sections.append((current_anchor, current_label, current))

    out: list[tuple[str, str, str]] = []
    for anchor, label, body_lines in sections:
        body = "\n".join(body_lines).strip()
        if len(body) >= 20:  # skip near-empty (whitespace-only) sections
            out.append((anchor, label, body))
    return out


def _chunk_novara_extras(doc: Document) -> list[tuple[str, str, str]]:
    """Split Novara markdown extras at ## and ### headings."""
    lines = doc.chunk_text.splitlines()
    sections: list[tuple[str, str, list[str]]] = []
    current_anchor = "preamble"
    current_label = f"{doc.section_reference} — Preamble"
    current: list[str] = []
    seen_section = False
    for line in lines:
        m = _MD_HEADING.match(line)
        # Treat ## and ### as section boundaries; skip the leading # title.
        if m and len(m.group(1)) >= 2:
            if current and (seen_section or _estimate_tokens("\n".join(current)) >= 50):
                sections.append((current_anchor, current_label, current))
            seen_section = True
            title = m.group(2).strip()
            anchor = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or f"section-{len(sections) + 1}"
            current_anchor = anchor
            current_label = f"{doc.section_reference} — {title}"
            current = []
        else:
            current.append(line)
    if current and (seen_section or _estimate_tokens("\n".join(current)) >= 50):
        sections.append((current_anchor, current_label, current))

    out: list[tuple[str, str, str]] = []
    for anchor, label, body_lines in sections:
        body = "\n".join(body_lines).strip()
        if body:
            out.append((anchor, label, body))
    return out


def _dispatch_chunker(doc: Document) -> list[tuple[str, str, str]]:
    if doc.file_path == "regulation/eu-ai-act-2024-1689.txt":
        return _chunk_ai_act(doc.chunk_text)
    if doc.corpus_tag == "REG":
        return _chunk_gdpr_article(doc)
    if doc.corpus_tag == "DEP":
        return _chunk_novara_policy(doc)
    if doc.corpus_tag == "DEP_EXTRAS":
        return _chunk_novara_extras(doc)
    raise ValueError(f"no chunker for {doc.file_path} (tag {doc.corpus_tag})")


def chunk_corpus(documents: list[Document]) -> list[Chunk]:
    """Refine file-level Documents into article/§/section-level Chunks
    with per-chunk sentence breakdown for FLEX-3 aggregation."""
    chunks: list[Chunk] = []
    for doc in documents:
        for anchor, label, body in _dispatch_chunker(doc):
            chunks.append(Chunk(
                chunk_id=_section_chunk_id(doc.chunk_id, anchor),
                parent_document_id=doc.chunk_id,
                corpus_tag=doc.corpus_tag,
                document_id=doc.document_id,
                section_reference=label,
                source_url=doc.source_url,
                chunk_text=body,
                file_path=doc.file_path,
                sha256_short=doc.sha256_short,
                sentences=_sentences(body),
            ))
    return chunks


# ----- ING-03 chunk embedding with on-disk cache --------------------------


_ST_MODELS: dict = {}  # model_name -> SentenceTransformer instance


def _get_st_model(model_name: str):
    """Lazy-load + cache SentenceTransformer instances at module level.

    Avoids the slow sentence_transformers import + model load when only
    ING-01/ING-02 is being exercised. Same singleton pattern as the
    spaCy sentencizer in ING-02.
    """
    if model_name not in _ST_MODELS:
        from sentence_transformers import SentenceTransformer
        _ST_MODELS[model_name] = SentenceTransformer(model_name)
    return _ST_MODELS[model_name]


def _cache_path(cache_dir: Path, model_name: str) -> Path:
    return cache_dir / f"{model_name}.npz"


def _load_cache(cache_path: Path) -> tuple[np.ndarray, list[str]] | None:
    """Return (embeddings, chunk_ids) from a cached .npz, or None on
    absent / corrupted file. Caller treats None as cache-miss.

    Catching Exception broadly is deliberate — a cache failure should never
    crash the system; we just re-embed. numpy raises an array of error
    types depending on what's wrong with the file (UnpicklingError,
    ValueError, OSError, BadZipFile…) and listing them all is fragile.
    """
    if not cache_path.exists():
        return None
    try:
        data = np.load(cache_path, allow_pickle=True)
        embeddings = data["embeddings"]
        chunk_ids = list(data["chunk_ids"])
        return embeddings, chunk_ids
    except Exception:
        return None


def _save_cache(
    cache_path: Path, embeddings: np.ndarray, chunk_ids: list[str]
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, embeddings=embeddings, chunk_ids=np.array(chunk_ids, dtype=object))


def _compute_embeddings(chunks: list[Chunk], model_name: str) -> np.ndarray:
    """Encode each chunk's chunk_text with the given model. Returns a
    float32 numpy array of shape (N, D). The `multi-qa-MiniLM-L6-cos-v1`
    model outputs already-L2-normalised vectors (the `-cos-v1` suffix);
    `normalize_embeddings=False` because re-normalising would be redundant."""
    if not chunks:
        return np.empty((0, 0), dtype=np.float32)
    model = _get_st_model(model_name)
    texts = [c.chunk_text for c in chunks]
    return model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=False,
        convert_to_numpy=True,
    ).astype(np.float32)


def embed_chunks(
    chunks: list[Chunk],
    model_name: str = "multi-qa-MiniLM-L6-cos-v1",
    cache_dir: Path = Path("embeddings"),
) -> tuple[np.ndarray, list[str]]:
    """Embed each chunk's text and persist to an on-disk cache.

    Hits `cache_dir/{model_name}.npz` when the cached chunk_ids exactly
    match the input chunks (same set, same order). Otherwise computes
    fresh embeddings and writes the cache. Model identity is encoded
    in the filename so FLEX-3 swaps (e.g., to `bge-large-en-v1.5`)
    create a new file alongside without risk of stale-vector reuse.

    Returns (embeddings, chunk_ids) where embeddings is shape (N, D)
    in float32 and chunk_ids[i] corresponds to row i.
    """
    input_ids = [c.chunk_id for c in chunks]
    cache_path = _cache_path(cache_dir, model_name)

    cached = _load_cache(cache_path)
    if cached is not None and cached[1] == input_ids:
        return cached

    embeddings = _compute_embeddings(chunks, model_name)
    _save_cache(cache_path, embeddings, input_ids)
    return embeddings, input_ids
