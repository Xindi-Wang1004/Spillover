# Spillover — public release

This repository archives companion analysis artifacts for **GenomeML Report Card** (an executable audit framework for biological generalization claims in genome machine learning), including viral-genome integrity-case materials used alongside the software package.

## GenomeML Report Card (`genome-ml-reportcard`)

Canonical software home: https://github.com/Xindi-Wang1004/GenomeML-ReportCard

```bash
pip install genome-ml-reportcard
genome-ml-reportcard --help
```

PyPI: https://pypi.org/project/genome-ml-reportcard/  
Software archive (Zenodo concept DOI): https://doi.org/10.5281/zenodo.22275801  
Fine-tuned Spillover checkpoints (separate): https://doi.org/10.5281/zenodo.21809791

A lightweight package mirror and frozen tables also live under `reportcard_companion/` in this repository.

## Quick check

```bash
export SPILLOVER_ROOT="$(pwd)"
python3 verify_release.py
```

## Layout

| Directory | Contents |
|-----------|----------|
| `reportcard_companion/` | Report Card companion package mirror, manifests, frozen tables |
| `data/` | Overlap cohort, SpillOver rankings, E2E predictions, IG inputs |
| `embeddings/` | Frozen pooled genome embeddings |
| `models/` | Fine-tuned checkpoints + `checkpoint_manifest.json` |
| `code/` | Analysis scripts |
| `tables/` | Whitelisted tables and analysis outputs |
| `figures/` | Figures |
| `docs/` | Reproducibility notes |

## External data

SpillOver composite scores: Grange et al., 2021 — https://spillover.ecohealthalliance.org/  
Viral accessions: NCBI Virus / GenBank (listed in tables).

## Citation

Please cite the GenomeML Report Card manuscript and, where SpillOver scores are used, the SpillOver data resource.
