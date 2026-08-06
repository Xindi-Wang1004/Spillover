# Model checkpoints

Three genome-fusion LoRA fine-tunes used in the manuscript (regression / host-classification / multitask).

Each `best_*.pth` is ~13 GB (includes adapter + training state as saved by the project).

## Download (Zenodo)

Large `.pth` weights are **not** stored on GitHub (file-size limits). They are archived at:

- **DOI:** [10.5281/zenodo.21809791](https://doi.org/10.5281/zenodo.21809791)
- **Record:** https://zenodo.org/records/21809791
- **Concept DOI** (all versions): [10.5281/zenodo.21809790](https://doi.org/10.5281/zenodo.21809790)

Files in the Zenodo deposit:

| File | Role |
|------|------|
| `best_genome_fusion_regression_20260501_042424.pth` | Regression |
| `best_classifier_genome_fusion_20260501_042922.pth` | Host classification |
| `best_multitask_genome_fusion_20260501_043813.pth` | Multitask |
| `metrics_*.json` | Training metrics companions |
| `checkpoint_manifest.json` | Paths + SHA256 checksums |

Verify downloads against SHA256 in `checkpoint_manifest.json` (also in this git tree).

## In this repository

This `models/` directory keeps metrics JSON and `checkpoint_manifest.json` only.

Training scripts: `../code/train/`
