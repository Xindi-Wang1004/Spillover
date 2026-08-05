# Data sources

## SpillOver composite scores

- Grange ZL et al. Ranking the risk of animal-to-human spillover for newly discovered viruses. *PNAS* 2021.
- Tool: https://spillover.ecohealthalliance.org/
- File in release: `data/spillover_rankings.csv`

## Overlap evaluation cohort

- n = 632 sequences with non-missing SpillOver Total; excluded from all model development.
- File: `data/overlap_cohort.csv`

## Viral sequences

- NCBI Virus / GenBank accessions as listed in supplementary tables.
- Local FASTA caches from the original pipeline are **not** bundled in full (size / redundancy).

## GISAID

- Influenza descriptive analyses used GISAID EpiFlu metadata under GISAID terms.
- Metadata not redistributed in this repository.

## Model weights

- Listed in `models/checkpoint_manifest.json` with source paths on the build server.
- Evo backbone pretrained on prokaryotic OpenGenome (Nguyen et al., 2024); viral fine-tuning via LoRA.
