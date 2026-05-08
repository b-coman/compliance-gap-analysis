# Corpus Manifest — Compliance Gap Analysis Project

**Snapshot date:** 2 May 2026 (corpus frozen for evaluation runs)
**Purpose:** Reproducible source material for the RAG-based compliance gap analysis system. Canonical corpus on which all evaluation results are computed.

This manifest is intended for inclusion as appendix material in the project's written report to satisfy the brief's reproducibility requirement.

---

## Summary

| Bucket | Description | Files | Words | Bytes |
|---|---|---|---|---|
| `regulation/` | External regulation: EU AI Act (Reg. 2024/1689) + UK GDPR articles 5, 6, 9, 13, 14, 22, 28, 35 | 10 | 98,571 | 2,380,534 |
| `deployer/` | Fictive deployer master policy: Novara AI Governance Policy v3.1 | 2 | 3,994 | 197,432 |
| `deployer-extras/` | Fictive deployer supporting documents: 5 TalentLens-anchored documents (Model Card, DPIA, Transparency Notice, Annual Governance Report, Model Intake Assessment) | 5 | 8,343 | 55,431 |
| **Total** | | **17** | **110,908** | **2,633,397** |

## Folder structure

```
corpus/
├── manifest.json              (machine-readable manifest)
├── manifest.md                (this file)
├── regulation/                (REG)
│   ├── eu-ai-act-2024-1689.pdf       Reg (EU) 2024/1689 — Artificial Intelligence Act (419pp)
│   ├── eu-ai-act-2024-1689.txt       Extracted text (~94k words)
│   └── uk-gdpr-art-{N}.txt           Individual articles (8 files: 5, 6, 9, 13, 14, 22, 28, 35)
├── deployer/                  (DEP — Novara fictive master policy)
│   ├── novara-ai-policy-v3.1.pdf     Master policy
│   └── novara-ai-policy-v3.1.txt     Extracted text
└── deployer-extras/           (DEP_EXTRAS — Novara fictive supporting documents)
    ├── novara-talentlens-model-card.md
    ├── novara-talentlens-dpia.md
    ├── novara-talentlens-transparency-notice.md
    ├── novara-2025-ai-governance-report.md
    └── novara-talentlens-model-intake-assessment.md
```

The two PDFs (`eu-ai-act-2024-1689.pdf`, `novara-ai-policy-v3.1.pdf`) are present for citation provenance only; their `.txt` siblings are the canonical retrieval source. The ingestion pipeline marks PDFs with `word_count == 0` and skips them at load time.

## Regulation (REG)

| File | Date | Source | Bytes | Words | SHA-256 (16) |
|---|---|---|---|---|---|
| `regulation/eu-ai-act-2024-1689.pdf` | 2024-07-12 | [link](https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=OJ:L_202401689) | 1,606,488 | 0 | `0955714a0b0ab19a` |
| `regulation/eu-ai-act-2024-1689.txt` | — | [link](https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=OJ:L_202401689) | 740,312 | 93,565 | `5927aa360d1fa9c9` |
| `regulation/uk-gdpr-art-5.txt` | — | [link](https://gdpr-info.eu/art-5-gdpr/) | 2,804 | 334 | `24412bec299ded45` |
| `regulation/uk-gdpr-art-6.txt` | — | [link](https://gdpr-info.eu/art-6-gdpr/) | 5,417 | 769 | `cd6a32c3b1b0348e` |
| `regulation/uk-gdpr-art-9.txt` | — | [link](https://gdpr-info.eu/art-9-gdpr/) | 5,595 | 802 | `e91a3930a54bf9cd` |
| `regulation/uk-gdpr-art-13.txt` | — | [link](https://gdpr-info.eu/art-13-gdpr/) | 3,216 | 526 | `94945eae1c8bf262` |
| `regulation/uk-gdpr-art-14.txt` | — | [link](https://gdpr-info.eu/art-14-gdpr/) | 5,384 | 776 | `1e659e173eb4ff23` |
| `regulation/uk-gdpr-art-22.txt` | — | [link](https://gdpr-info.eu/art-22-gdpr/) | 1,376 | 223 | `a3b18e8e212ca58f` |
| `regulation/uk-gdpr-art-28.txt` | — | [link](https://gdpr-info.eu/art-28-gdpr/) | 5,511 | 874 | `e60d6e5bcfa447a1` |
| `regulation/uk-gdpr-art-35.txt` | — | [link](https://gdpr-info.eu/art-35-gdpr/) | 4,431 | 702 | `78e04cfcc8bb75fb` |

## Deployer master policy (DEP)

| File | Date | Source | Bytes | Words | SHA-256 (16) |
|---|---|---|---|---|---|
| `deployer/novara-ai-policy-v3.1.pdf` | 2025-03-01 (fictive) | — Fictive (Novara fabricated for project) — | 169,883 | 0 | `a80ae3a67ae5c08f` |
| `deployer/novara-ai-policy-v3.1.txt` | — | — Fictive (Novara fabricated for project) — | 27,549 | 3,994 | `215d8fbc00cbcbdb` |

## Deployer supporting documents (DEP_EXTRAS)

| File | Date | Source | Bytes | Words | SHA-256 (16) |
|---|---|---|---|---|---|
| `deployer-extras/novara-2025-ai-governance-report.md` | 2026-02-28 (fictive) | — Fictive (Novara fabricated for project) — | 13,889 | 2,113 | `a99b9c6343316b83` |
| `deployer-extras/novara-talentlens-dpia.md` | 2025-02-12 (fictive) | — Fictive (Novara fabricated for project) — | 13,033 | 2,013 | `4bbcbda267147404` |
| `deployer-extras/novara-talentlens-model-card.md` | 2025-01-15 (fictive) | — Fictive (Novara fabricated for project) — | 12,749 | 1,790 | `9aef163f6b38c027` |
| `deployer-extras/novara-talentlens-model-intake-assessment.md` | 2025-01-15 (fictive) | — Fictive (Novara fabricated for project) — | 9,744 | 1,489 | `e2f2d1809b704904` |
| `deployer-extras/novara-talentlens-transparency-notice.md` | 2025-03-01 (fictive) | — Fictive (Novara fabricated for project) — | 6,016 | 938 | `ebd621b0001753ce` |

## Notes on document selection and known caveats

**On the AI Act PDF.** Fetched as the EU Council consolidated version (PE-CONS 24/24), textually identical to Regulation (EU) 2024/1689 as published in the Official Journal of the European Union on 12 July 2024. EUR-Lex is the canonical legal publisher; cite EUR-Lex for academic submission.

**On UK GDPR text source.** Article texts are taken from gdpr-info.eu, which mirrors the EU GDPR text. UK GDPR articles share numbering and substantive text with EU GDPR for the cluster used here. UK-specific implementations are referenced separately via the Data Protection Act 2018 (not in this corpus).

**On GDPR coverage.** Eight articles are included: principles (Art 5), lawful basis (Arts 6, 9), information at collection (Arts 13, 14), automated decisions (Art 22), processors (Art 28), and DPIA (Art 35). This is a curated subset relevant to a CV-screening AI deployer's compliance surface, not full GDPR coverage. A production system would extend the subset as the query domain expanded.

**On Novara fictive documents.** All Novara-named documents are fabricated for academic purposes. Novara AI, Inc. is a fictional company. The supporting documents are designed to contain realistic compliance gaps for the gap-analysis system to surface. None of the people, products, customers, or events described are real.

## Refresh and versioning policy

This corpus is **frozen** as of 2 May 2026 for the duration of evaluation runs. Any updates to source documents (regulatory amendments, revisions to fictive Novara documents) require a new versioned snapshot with a fresh manifest. SHA-256 hashes recorded above are the integrity reference; re-hashing the live files should produce these exact values.

---

*Manifest: 17 files, 110,908 words, 2,633,397 bytes (~2.5 MB).*
