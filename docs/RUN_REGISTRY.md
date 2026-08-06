# Run registry (Science Bulletin)

Index of frozen analysis runs supporting the main and supplementary results.
Full notes: `REPRODUCIBILITY.md`. Seed **42** unless a local `run_meta_*.json` says otherwise.

| Analysis | Registry / metadata | Role in manuscript |
|----------|---------------------|--------------------|
| P0/P1 circularity & matched shallow baselines | `code/analysis/analysis_p0p1_circularity_baselines_wxd0804/run_meta_p0p1_wxd0804.json` | Total′ / host-axis sensitivity; S38–S40; organism-blocked Evo vs taxonomy/k-mer (S39) |
| Non–Human Virus bootstrap (underpowered) | `…/run_meta_nonHuman_bootstrap_wxd0804.json` | Caveat only; not a main claim |
| Follow-ups S35–S37 | `code/analysis/analysis_sb_followups_wxd0803/run_meta_wxd0803.json` | Composition, known-locus meta, relatedness controls |
| Host-interaction enrichment (CoV + cross-family) | `code/analysis/analysis_host_interaction_enrichment/*_run_meta.json` | Within-genome enrichment; module tables; structure mapping |
| Picorna (Rhinovirus C) prespecified contrast | `code/analysis/analysis_picorna_prespec_wxd0804/REPRO_REGISTRY_picorna_prespec_wxd0804.md` + `run_meta_*.json` | Capsid/entry OR exemplar (S42) |
| CoV dense-window sensitivity | `code/analysis/analysis_cov_dense_windows_wxd0804/run_meta_cov_dense_windows_wxd0804.json` | Dense-window companion (S41) |
| ORF1ab position permutation (S43) | `code/analysis/analysis_orf1ab_posperm_wxd0804/` | Top-window position-permutation audit |

## End-to-end predictions (held-out overlap, n = 632)

Stored under `data/`:

- `infer_regression_per_sequence_predictions.csv`
- `infer_classification_per_sequence_predictions.csv`
- `infer_multitask_per_sequence_predictions.csv`
- `overlap_cohort.csv`

## Model checkpoints

Large `.pth` weights are **not** in this git tree. Download from Zenodo  
DOI [10.5281/zenodo.21809791](https://doi.org/10.5281/zenodo.21809791); verify SHA256 in `models/checkpoint_manifest.json`.
