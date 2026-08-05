# Model checkpoints

Three genome-fusion LoRA fine-tunes used in the manuscript (regression / host-classification / multitask).

Each `best_*.pth` is ~13 GB (includes adapter + training state as saved by the project).

## GitHub note

These `.pth` files exceed GitHub file limits. They are kept in the local release tree and should be published via:
- GitHub Releases (asset upload), or Zenodo / Hugging Face, with SHA256 in `checkpoint_manifest.json`.

Training scripts: `../code/train/`

Metrics JSON files are small and always included in the git-facing package.
