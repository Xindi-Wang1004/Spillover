# genome-ml-reportcard

**GenomeML Report Card** — an executable audit for genome machine learning when labels are assigned at biological-group levels (species, lineage, cluster, …).

Records **label-assignment units** and **blocking units**, audits sequence/group overlap and label geometry, and compares random vs pre-specified group-blocked evaluation under a common JSON/Markdown schema.

## Install

```bash
pip install genome-ml-reportcard
```

Development / paper bundle:

```bash
git clone https://github.com/Xindi-Wang1004/Spillover.git
pip install -e Spillover/transfer_GB/audit_toolkit
```

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
