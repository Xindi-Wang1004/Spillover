# genome-ml-reportcard

**GenomeML Report Card** — an executable audit for genome machine learning when labels are assigned at biological-group levels (species, lineage, cluster, …).

Records **label-assignment units** and **blocking units**, audits sequence/group overlap and label geometry, and compares random vs pre-specified group-blocked evaluation under a common JSON/Markdown schema.

## Install

PyPI (`genome-ml-reportcard`) is **not yet published** (verified 2026-09-01). Install from this repository or the tagged release wheel:

```bash
git clone https://github.com/Xindi-Wang1004/Spillover.git
pip install -e Spillover/transfer_GB/audit_toolkit
```

```bash
# Release wheel (v0.1.1)
pip install https://github.com/Xindi-Wang1004/Spillover/releases/download/reportcard-v0.1.1/genome_ml_reportcard-0.1.1-py3-none-any.whl
```

After PyPI publication:

```bash
pip install genome-ml-reportcard
```

Software archive: [10.5281/zenodo.22226465](https://doi.org/10.5281/zenodo.22226465).

## Quickstart

```bash
genome-ml-reportcard \
  --table manifest.tsv \
  --accession accession \
  --group species \
  --block species \
  --label ogt_c \
  --features X_kmer4.npy \
  --out report/audit.json
```

Key output fields:

| Field | Meaning |
|-------|---------|
| `geometry.random_cv_shared_block_fraction` | Share of test rows whose block also appears in train under random CV |
| `geometry.within_block_homogeneity` | ICC (continuous) or majority purity (binary) within blocks |
| `probe.random` / `probe.blocked` | Task-primary metric under random vs blocked CV |
| `probe.delta` | Within-task split-design contrast |
| `overlap` | Accession / block overlap when `--table-b` is provided |

## Schema

| column | meaning |
|--------|---------|
| `sequence_id` / `accession` | genome identifier |
| `label` | numeric or binary phenotype |
| `group` | label-assignment unit (Layer A) |
| `block` (optional) | deployment / blocking unit (Layer B; defaults to `group`) |

## Simulation

From this package directory:

```bash
python simulate_label_geometry.py
```

## CI

GitHub Actions: `.github/workflows/ci.yml` (smoke test + wheel build).

## Citation

Wang et al., *GenomeML Report Card: an executable framework for estimand-matched evaluation of genome machine learning* (Genome Biology submission).

## License

MIT — see `LICENSE`.
