# Reproducibility lock (Genome Biology package)

**Document generated:** 2026-09-01 (CST)  
**GB v2 preregistry file mtime:** 2026-08-31 23:48:18 CST  
**Claim:** `docs/PREREGISTRY_MULTI_TASK_BENCHMARK_GBv2.md` was locked before execution of T03_REP / T06 / T08 (those runs completed 2026-09-01 ~00:05 CST).

## Public code pointer

| Resource | Value |
|----------|--------|
| Spillover public tree commit (at package build) | `38f08c2dc5a04645827a61054253517e56ecd920` |
| GitHub | https://github.com/Xindi-Wang1004/Spillover |
| Zenodo checkpoints | 10.5281/zenodo.21809791 |
| Engqvist OGT | 10.5281/zenodo.1175609 |

**Action before journal upload:** create a dated GitHub release / Zenodo version that pins this `transfer_GB/` snapshot and refresh hashes below if any manifest changes.

## SHA-256 manifest lock

See machine-readable `tables/Table_reproducibility_hashes.json`.

## Amendment timeline (excerpt)

| When | What |
|------|------|
| 2026-08-31 | CG v1 multi-task prereg locked; T01–T05 executed; T04→T04_ALT |
| 2026-08-31 23:48 | **GB v2 prereg locked** (inclusion: non-singleton primary rows; AUROC for binary) |
| 2026-09-01 | T03_REP, T06, T08 executed; Domain B ingested as T07; construction controls demoted |
| 2026-09-01 | Statistical-robustness suite (locked group-aware α; repeated split-design contrast; group-macro; LOGO) → `tables/Table_robustness_split_design.csv` |

## Probe contract

**Historical matrix (Table 1):** `SEED=42`, RidgeCV, fold-wise StandardScaler, K=5 OOF; binary primary AUROC; continuous primary Spearman ρ.

**Robustness suite (§2.4):** lock Ridge α once via group-aware GroupKFold MSE; reuse α across repeats; report genome-pooled and group-macro Δ with 2.5–97.5% percentile intervals (`scripts/run_robustness_suite.py`).
