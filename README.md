# Spillover — Science Bulletin public release

**Distributed, lineage-structured host-associated signals in viral genomes recovered from spillover-risk phenotypes**

Wang, Luo, Li, Hon — *Science Bulletin*.

Built from the internal analysis workspace used for the manuscript.  
**Not** a dump of the whole project — only models, data, code, and tables used in the **final** manuscript.

## Quick check

```bash
export SPILLOVER_ROOT="$(pwd)"
python3 verify_release.py
```

## Layout

| Directory | Contents |
|-----------|----------|
| `data/` | Overlap cohort (n=632), SpillOver rankings, E2E predictions, IG inputs |
| `embeddings/` | Frozen pooled genome embeddings (regression / host / multitask) |
| `models/` | Fine-tuned checkpoints + `checkpoint_manifest.json` |
| `code/analysis/` | Final post-hoc analysis bundles (P0/P1, S35–S37, S43, Picorna, CoV dense) |
| `code/` | Host-interaction enrichment & known-loci scripts |
| `tables/` | Whitelisted `bib_tables` + analysis outputs (single copy, de-duplicated) |
| `figures/` | Main-text Figures 1–4 (+ GA if present) |
| `manuscript/` | Final Word manuscript (if copied at build time) |
| `docs/` | Reproducibility notes for reviewers and *Science Bulletin* |


## External data

SpillOver composite scores: Grange et al., 2021 — https://spillover.ecohealthalliance.org/  
Viral accessions: NCBI Virus / GenBank (listed in tables).

See `docs/DATA_SOURCES.md` and `docs/REPRODUCIBILITY.md`.

## Citation

Please cite the manuscript and the SpillOver data resource.
