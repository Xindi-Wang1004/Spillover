# Reproducibility — Science Bulletin

## Build provenance

This tree was assembled from the internal analysis workspace used for the manuscript
(via `build_public_release.sh` in the release tooling).  
Build metadata (if present): `RELEASE_meta.json`.

## Environment

- Conda env: `evo_design` (for full GPU pipelines on the original compute host)
- Python ≥3.10; `pip install -r requirements.txt` suffices for **statistical re-runs** on frozen embeddings

Random seed **42** throughout (see `run_meta_*.json` in each `code/analysis/*` folder).

## Reproduction map (final manuscript)

| Manuscript item | Location | Re-run |
|-----------------|----------|--------|
| Table 1 (E2E ρ) | `data/infer_*_per_sequence_predictions.csv` + overlap cohort | Requires `models/` or stored predictions |
| Table 2 (linear probes) | `embeddings/` + `code/analysis/analysis_p0p1_*` | `python code/analysis/analysis_p0p1_circularity_baselines_wxd0804/run_all.py` |
| Fig 2 label permutation | `tables/Table5_label_permutation_negative_controls.csv` | From training pipeline outputs |
| Supp P0/P1 | `code/analysis/analysis_p0p1_circularity_baselines_wxd0804/` | `run_all.py` |
| Supp S35–S37 | `code/analysis/analysis_sb_followups_wxd0803/` | `run_all.py` |
| Supp S43 (ORF1ab position perm) | `code/analysis/analysis_orf1ab_posperm_wxd0804/` | `run_orf1ab_position_perm_wxd0804.py` |
| Picorna exemplar | `code/analysis/analysis_picorna_prespec_wxd0804/` | See `REPRO_REGISTRY` in folder |
| CoV dense sensitivity | `code/analysis/analysis_cov_dense_windows_wxd0804/` | See `run_meta_*.json` |
| IG / module tables S19–S34 | `tables/Table19–Table34*` | `code/run_attribution_enrichment_host_interaction.py` (GPU + `evo_data` paths) |

## Path note

Some analysis scripts retain absolute paths from the original workspace.  
For local use, set:

```bash
export SPILLOVER_ROOT="$(pwd)"
```

and edit path constants in the relevant `run_*.py` / `run_all.py` to point at this tree
(or the companion Zenodo checkpoint deposit).

## Science Bulletin data-availability statement (suggested)

> Analysis code, frozen embeddings, supplementary tables, and analysis outputs supporting the main and supplementary results are available at https://github.com/Xindi-Wang1004/Spillover. Fine-tuned model checkpoints are archived at Zenodo (DOI: 10.5281/zenodo.21809791). Third-party data (GISAID EpiFlu metadata) were used under their respective terms and are not redistributed.

## Contact

Corresponding authors: Junyu Luo (luo_junyu@gzlab.ac.cn); Yixue Li (li_yixue@gzlab.ac.cn); Chitin Hon (cthon@must.edu.mo).
