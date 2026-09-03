# Spillover —  public release

This repository supports the **** Method/Resource manuscript:

**GenomeML Report Card: an executable framework for estimand-matched evaluation of genome machine learning**

It also archives companion Spillover viral-genome analysis artifacts (models, embeddings, and post-hoc scripts) used in the Report Card supplementary integrity case and related analyses.

## GenomeML Report Card (`genome-ml-reportcard`)

Executable audit package for estimand-matched genome ML evaluation: records label-assignment units and evaluation blocks, audits sequence overlap / group recurrence / label geometry, and compares random vs pre-specified blocked evaluation under a common reporting schema.

### Install

```bash
pip install genome-ml-reportcard
genome-ml-reportcard --help
```

Development / paper bundle from this repo:

```bash
git clone https://github.com/Xindi-Wang1004/Spillover.git
cd Spillover
pip install -e reportcard_companion/audit_toolkit
```

PyPI: https://pypi.org/project/genome-ml-reportcard/

Source: `reportcard_companion/audit_toolkit/`. Manuscript draft: `reportcard_companion/GB_estimand_v1.md`.  
Software archive: Zenodo DOI [10.5281/zenodo.22226465](https://doi.org/10.5281/zenodo.22226465).  
Fine-tuned Spillover checkpoints (separate): Zenodo DOI [10.5281/zenodo.21809791](https://doi.org/10.5281/zenodo.21809791).  
Release tag: [`reportcard-v0.1.1`](https://github.com/Xindi-Wang1004/Spillover/releases/tag/reportcard-v0.1.1).

## Quick check (Spillover companion tree)

```bash
export SPILLOVER_ROOT="$(pwd)"
python3 verify_release.py
```

## Layout

| Directory | Contents |
|-----------|----------|
| `reportcard_companion/` | GenomeML Report Card package, manifests, and GB manuscript materials |
| `data/` | Overlap cohort (n=632), SpillOver rankings, E2E predictions, IG inputs |
| `embeddings/` | Frozen pooled genome embeddings (regression / host / multitask) |
| `models/` | Fine-tuned checkpoints + `checkpoint_manifest.json` |
| `code/analysis/` | Final post-hoc analysis bundles (P0/P1, S35–S37, S43, Picorna, CoV dense) |
| `code/` | Host-interaction enrichment & known-loci scripts |
| `tables/` | Whitelisted `bib_tables` + analysis outputs (single copy, de-duplicated) |
| `figures/` | Main-text figures (+ GA if present) |
| `manuscript/` | Final Word manuscript (if copied at build time) |
| `docs/` | Reproducibility notes for reviewers |

## External data

SpillOver composite scores: Grange et al., 2021 — https://spillover.ecohealthalliance.org/  
Viral accessions: NCBI Virus / GenBank (listed in tables).

See `docs/DATA_SOURCES.md` and `docs/REPRODUCIBILITY.md`.

## Citation

Please cite the  manuscript (GenomeML Report Card) and, where SpillOver scores are used, the SpillOver data resource.
