"""CHN-01..CHN-07 — five-step obligation-level reasoning chain.

This module is the *initial architecture as evaluated* for this project. It
implements the multi-step chain documented in compliance-gap-analysis-spec.md
§ Group: Reasoning Chain. Empirical evaluation (see docs/evaluation-findings.md
Stages 1, 5, 7) found three architectural failure modes that retrieval upgrade
does not fix: decomposition drift, miscalibrated silence detection, and
unreliable provenance labelling. The production demo path is the simplified
single-call architecture in src/simplified.py.

This chain is preserved in the codebase as the empirical baseline that
supports the report's Critical Analysis findings.

The unit of comparison is the atomic obligation, not the chunk (D-008).

  Step 1 (CHN-01)  decompose user query into focused sub-questions
                   (1 LLM call; cap at 4 sub-questions per decisions.md §3)
  Step 2 (CHN-02)  per-sub-question regulation retrieval (no LLM call)
  Step 3 (CHN-03)  atomic obligation extraction from retrieved regulation
                   chunks (1 LLM call per sub-question; cap ~5 obligations)
  Step 4 (CHN-04)  per-obligation matching with threshold-grounded silence
                   detection. Phase 1: silence by cosine similarity vs τ
                   (deterministic, no LLM). Phase 2: 4-state classifier
                   for surviving obligations (1 batched LLM call per
                   sub-question; per-obligation is the FLEX-1 escalation).
  Step 5 (CHN-05)  register synthesis from per-obligation rows (1 LLM
                   call). All fields except `gap_characterisation`
                   mechanically derived from chain state.

Cost per uncached query: ~6-8 LLM calls. Verbose mode (CHN-06) prints
intermediate state. End-to-end latency target on Gemma uncached: <90s
(CHN-07, measured via the live-API smoke test, not enforced in code).

Chain code types against `LLMClient` ABC (FLEX-6 strip-safety
discipline) — never against `RoutingClient` or a concrete adapter.

Reference: compliance-gap-analysis-spec.md § Group: Reasoning Chain.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

from src.ingestion import Chunk
from src.llm.client import LLMClient
from src.retrieval import ChunkEmbeddingRetriever
from src.schema import (
    CONFIDENCE_VALUES,
    EvidenceCitation,
    MATCH_STATUS_VALUES,
    RegisterRow,
    validate_register_row,
)


# ----- Chain-state types --------------------------------------------------


@dataclass(frozen=True, slots=True)
class Obligation:
    """Per-obligation chain state propagated through Steps 3-5.

    Constructed at CHN-03 with text + sub_question + regulation chunk_ids;
    extended by CHN-04 with match_status + per-corpus evidence citations;
    consumed by CHN-05 to materialise RegisterRow.

    Frozen — each step uses `dataclasses.replace()` to produce a new
    Obligation rather than mutating in place. Mutation-free state is
    easier to reason about and trivially serialisable for cache-replay.
    """
    text: str
    sub_question: str
    regulation_chunk_ids: tuple[str, ...]
    regulatory_provision: str
    match_status: str | None = None
    policy_evidence: tuple[EvidenceCitation, ...] = ()
    extras_evidence: tuple[EvidenceCitation, ...] = ()
    guidance_evidence: tuple[EvidenceCitation, ...] = ()
    max_sim_against_deployer: float | None = None


# ----- Module-level helpers (pure functions) ------------------------------


def _derive_confidence(ob: Obligation) -> str:
    """Confidence binning per `docs/decisions.md` §§1, 2.

    Silent rows are deterministic (no LLM judgment), so confidence is high.
    For other statuses, take the *minimum* cosine similarity across all
    cited evidence (worst-cited-chunk semantics — surfaces borderline
    citations honestly rather than smoothing them away).
    """
    if ob.match_status == "silent":
        return "high"
    cited_scores = [
        c.score for c in (
            list(ob.policy_evidence)
            + list(ob.extras_evidence)
            + list(ob.guidance_evidence)
        )
    ]
    if not cited_scores:
        return "low"
    min_score = min(cited_scores)
    if min_score >= 0.55:
        return "high"
    if min_score >= 0.45:
        return "medium"
    return "low"


def _derive_regulatory_provision(reg_chunks: list[tuple[Chunk, float]]) -> str:
    """Citation label for an obligation: highest-scoring chunk's section_reference.

    All obligations from the same sub-question share this label —
    defensible because they were extracted from the same chunk set.
    """
    if not reg_chunks:
        return "(no regulation chunks)"
    top_chunk, _score = reg_chunks[0]  # retriever returns descending order
    return top_chunk.section_reference


def _dedupe_chunks_by_id(chunks: list[Chunk]) -> list[Chunk]:
    seen: set[str] = set()
    out: list[Chunk] = []
    for c in chunks:
        if c.chunk_id not in seen:
            seen.add(c.chunk_id)
            out.append(c)
    return out


def _merge_classifications(
    surviving: list[Obligation],
    classifications: list[dict],
) -> list[Obligation]:
    """Apply the LLM classifier output back onto Obligation chain state.

    `classifications` is a list of dicts each shaped like:
        {"obligation": str, "match_status": str, "evidence_chunk_ids": list[str]}
    Position is the primary alignment (LLM should preserve order); we filter
    LLM-suggested evidence_chunk_ids against the citations actually retrieved
    so a hallucinated chunk_id never lands in the register.
    """
    classified: list[Obligation] = []
    for ob, cls in zip(surviving, classifications):
        status = cls.get("match_status", "partial")
        if status not in MATCH_STATUS_VALUES or status == "silent":
            # Silent at this stage would be illegitimate (Phase 1 caught the
            # real silences); collapse to "partial" rather than corrupt enum.
            status = "partial" if status == "silent" else status
            if status not in MATCH_STATUS_VALUES:
                status = "partial"
        cited_ids = set(cls.get("evidence_chunk_ids", []) or [])
        policy_kept = tuple(c for c in ob.policy_evidence if c.chunk_id in cited_ids)
        extras_kept = tuple(c for c in ob.extras_evidence if c.chunk_id in cited_ids)
        guidance_kept = tuple(c for c in ob.guidance_evidence if c.chunk_id in cited_ids)
        classified.append(dataclasses.replace(
            ob,
            match_status=status,
            policy_evidence=policy_kept,
            extras_evidence=extras_kept,
            guidance_evidence=guidance_kept,
        ))
    return classified


# ----- The chain ----------------------------------------------------------


class ComplianceGapChain:
    """Five-step obligation-level reasoning chain.

    Reads against a `ChunkEmbeddingRetriever` (RET-01) for retrieval and
    a `LLMClient` (LLM-01) for language-model calls. The LLMClient type
    is the chain's only LLM-side dependency — FLEX-6 strip-safety
    discipline means routing wrappers and concrete adapters are never
    referenced from here.
    """

    def __init__(
        self,
        retriever: ChunkEmbeddingRetriever,
        client: LLMClient,
        *,
        tau: float = 0.35,
        top_k: int = 5,
        max_sub_questions: int = 4,
        verbose: bool = False,
    ) -> None:
        self.retriever = retriever
        self.client = client
        self.tau = tau
        self.top_k = top_k
        self.max_sub_questions = max_sub_questions
        self.verbose = verbose

    # === public entry point ===

    def run(self, user_query: str) -> list[RegisterRow]:
        """End-to-end chain. Returns a list of validated RegisterRows."""
        sub_questions = self._decompose(user_query)
        all_obligations: list[Obligation] = []
        for sq in sub_questions:
            reg_chunks = self._retrieve_regulation(sq)
            if not reg_chunks:
                continue
            obligations = self._extract_obligations(sq, reg_chunks)
            if not obligations:
                continue
            classified = self._match(sq, obligations)
            all_obligations.extend(classified)
        rows = self._synthesise(all_obligations)
        for row in rows:
            validate_register_row(row)
        return rows

    # === step methods ===

    def _decompose(self, query: str) -> list[str]:
        sub_questions = self.client.decompose_query(
            query, max_sub_questions=self.max_sub_questions
        )
        if self.verbose:
            print(f"[CHN-01] decomposed into {len(sub_questions)} sub-questions")
            for i, sq in enumerate(sub_questions, 1):
                print(f"  {i}. {sq}")
        return sub_questions

    def _retrieve_regulation(self, sub_question: str) -> list[tuple[Chunk, float]]:
        results = self.retriever.retrieve(
            sub_question, top_k=self.top_k, corpus_filter="REG"
        )
        if self.verbose:
            print(
                f"[CHN-02] {len(results)} REG chunks for: "
                f"{sub_question[:60]}{'…' if len(sub_question) > 60 else ''}"
            )
            for chunk, score in results:
                print(f"  {score:.3f}  {chunk.section_reference}")
        return results

    def _extract_obligations(
        self,
        sub_question: str,
        reg_chunks: list[tuple[Chunk, float]],
    ) -> list[Obligation]:
        chunks_only = [c for c, _s in reg_chunks]
        obligation_texts = self.client.extract_obligations(sub_question, chunks_only)
        chunk_ids = tuple(c.chunk_id for c in chunks_only)
        provision = _derive_regulatory_provision(reg_chunks)
        obligations = [
            Obligation(
                text=text,
                sub_question=sub_question,
                regulation_chunk_ids=chunk_ids,
                regulatory_provision=provision,
            )
            for text in obligation_texts
        ]
        if self.verbose:
            print(f"[CHN-03] extracted {len(obligations)} obligations")
            for ob in obligations:
                preview = ob.text[:80] + ("…" if len(ob.text) > 80 else "")
                print(f"  - {preview}")
        return obligations

    def _retrieve_deployer_evidence(
        self, obligation_text: str
    ) -> tuple[float, dict[str, list[Chunk]], dict[str, tuple[EvidenceCitation, ...]]]:
        """Retrieve up to top_k chunks per deployer-side corpus.

        Returns: (max_sim, chunks_per_corpus, citations_per_corpus). Three
        retrievals are used (one per filter) per `docs/decisions.md` §6's
        "5 chunks per deployer-side corpus = up to 15 candidate evidence
        chunks per obligation".
        """
        per_corpus_chunks: dict[str, list[Chunk]] = {}
        per_corpus_citations: dict[str, tuple[EvidenceCitation, ...]] = {}
        all_scores: list[float] = []
        for corpus_label, tag in (
            ("policy", "DEP"),
            ("extras", "DEP_EXTRAS"),
            ("guidance", "OPS"),
        ):
            results = self.retriever.retrieve(
                obligation_text, top_k=self.top_k, corpus_filter=tag
            )
            per_corpus_chunks[corpus_label] = [c for c, _s in results]
            per_corpus_citations[corpus_label] = tuple(
                EvidenceCitation(
                    chunk_id=c.chunk_id,
                    section_reference=c.section_reference,
                    score=float(s),
                )
                for c, s in results
            )
            all_scores.extend(s for _c, s in results)
        max_sim = max(all_scores) if all_scores else 0.0
        return max_sim, per_corpus_chunks, per_corpus_citations

    def _match(
        self,
        sub_question: str,
        obligations: list[Obligation],
    ) -> list[Obligation]:
        # Phase 1 — deterministic silence detection. No LLM call.
        silent: list[Obligation] = []
        surviving_with_chunks: list[tuple[Obligation, dict[str, list[Chunk]]]] = []
        for ob in obligations:
            max_sim, chunks_per_corpus, citations_per_corpus = (
                self._retrieve_deployer_evidence(ob.text)
            )
            if max_sim < self.tau:
                silent.append(dataclasses.replace(
                    ob,
                    match_status="silent",
                    max_sim_against_deployer=max_sim,
                    # silent rows have empty evidence in all three corpora
                ))
            else:
                surviving_with_chunks.append((
                    dataclasses.replace(
                        ob,
                        max_sim_against_deployer=max_sim,
                        policy_evidence=citations_per_corpus["policy"],
                        extras_evidence=citations_per_corpus["extras"],
                        guidance_evidence=citations_per_corpus["guidance"],
                    ),
                    chunks_per_corpus,
                ))

        if self.verbose:
            print(
                f"[CHN-04 P1] {len(silent)} silent (max_sim < τ={self.tau}), "
                f"{len(surviving_with_chunks)} surviving"
            )
            for ob in silent:
                preview = ob.text[:60] + ("…" if len(ob.text) > 60 else "")
                print(f"  SILENT  max_sim={ob.max_sim_against_deployer:.3f}  {preview}")

        if not surviving_with_chunks:
            return silent

        # Phase 2 — batched 4-state classifier. 1 LLM call per sub-question.
        surviving = [ob for ob, _ in surviving_with_chunks]
        # Aggregate evidence chunks across all surviving obligations,
        # deduped by chunk_id. The LLM sees the full candidate pool and
        # picks which chunks support which obligation.
        aggregated_evidence = {
            "policy": _dedupe_chunks_by_id([
                c for _ob, ec in surviving_with_chunks for c in ec["policy"]
            ]),
            "extras": _dedupe_chunks_by_id([
                c for _ob, ec in surviving_with_chunks for c in ec["extras"]
            ]),
            "guidance": _dedupe_chunks_by_id([
                c for _ob, ec in surviving_with_chunks for c in ec["guidance"]
            ]),
        }
        classifications = self.client.classify_obligations(
            sub_question=sub_question,
            obligations=[ob.text for ob in surviving],
            evidence=aggregated_evidence,
        )
        classified = _merge_classifications(surviving, classifications)

        if self.verbose:
            print(f"[CHN-04 P2] classified {len(classified)} surviving obligations")
            for ob in classified:
                preview = ob.text[:60] + ("…" if len(ob.text) > 60 else "")
                status = (ob.match_status or "?").upper()
                print(f"  {status:14s}  {preview}")

        return silent + classified

    def _synthesise(self, obligations: list[Obligation]) -> list[RegisterRow]:
        if not obligations:
            return []
        # Build LLM input — only the fields the LLM should see.
        rows_for_llm = [
            {
                "obligation": ob.text,
                "regulatory_provision": ob.regulatory_provision,
                "match_status": ob.match_status,
                "policy_evidence_count": len(ob.policy_evidence),
                "extras_evidence_count": len(ob.extras_evidence),
                "guidance_evidence_count": len(ob.guidance_evidence),
            }
            for ob in obligations
        ]
        enriched = self.client.synthesise_register(rows_for_llm)
        # Match enriched rows back to obligations by obligation text — robust
        # to LLM reordering or dropping items.
        gap_by_text: dict[str, str] = {
            (en.get("obligation", "") if isinstance(en, dict) else ""):
                (en.get("gap_characterisation", "") if isinstance(en, dict) else "")
            for en in enriched
        }
        register: list[RegisterRow] = []
        for ob in obligations:
            register.append(RegisterRow(
                regulatory_provision=ob.regulatory_provision,
                regulation_chunk_ids=ob.regulation_chunk_ids,
                obligation=ob.text,
                match_status=ob.match_status or "partial",
                policy_evidence=ob.policy_evidence,
                extras_evidence=ob.extras_evidence,
                guidance_evidence=ob.guidance_evidence,
                gap_characterisation=gap_by_text.get(ob.text, ""),
                confidence=_derive_confidence(ob),
            ))
        if self.verbose:
            print(f"[CHN-05] synthesised {len(register)} rows")
        return register


# ----- factory -----------------------------------------------------------


def build_chain(
    *,
    manifest_path: Path = Path("corpus/manifest.json"),
    cache_dir: Path = Path("llm_cache"),
    embeddings_dir: Path = Path("embeddings"),
    model_name: str = "multi-qa-MiniLM-L6-cos-v1",
    use_routing: bool = True,
    verbose: bool = False,
) -> ComplianceGapChain:
    """End-to-end factory: retriever + RoutingClient + cache → chain.

    `use_routing=True` wires the production primary→fallback path
    (GroqLlama70B + LocalGemma2B); `use_routing=False` returns just
    GroqLlama70B for FLEX-6 strip-down readiness.

    Mirrors `build_retriever()` from `src.retrieval`. The notebook /
    REPL / future entry points call this once at startup.
    """
    from src.llm.adapters import GroqLlama70B, LocalGemma2B
    from src.llm.cache import DiskCache
    from src.llm.routing import RoutingClient
    from src.retrieval import build_retriever

    retriever = build_retriever(
        manifest_path=manifest_path,
        model_name=model_name,
        cache_dir=embeddings_dir,
    )
    cache = DiskCache(cache_dir=cache_dir)

    if use_routing:
        primary = GroqLlama70B(cache=cache)
        fallback = LocalGemma2B(cache=cache)
        client: LLMClient = RoutingClient(primary, fallback)
    else:
        client = GroqLlama70B(cache=cache)

    return ComplianceGapChain(retriever, client, verbose=verbose)
