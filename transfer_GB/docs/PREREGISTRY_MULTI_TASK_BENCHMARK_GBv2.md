# Multi-task benchmark — Genome Biology amendment v2

**Status:** LOCKED before new-task execution (T03_REP, T06, T07, T08).  
**Date locked:** 2026-08-31  
**Target journal:** *Genome Biology* (Method / Resource — estimand-matched genome ML evaluation)  
**Supersedes for GB track:** `docs/PREREGISTRY_MULTI_TASK_BENCHMARK.md` (CG v1; retained for history)

**Identity:** Community **benchmark + audit resource**. Not a viral biology discovery paper.

---

## 1. Primary question

When phenotypes are assigned at a biological-group level, does the **evaluation unit** match the **label-assignment unit**? When it does not, genome-level random splits estimate **seen-group prediction** rather than the intended **unseen-group generalization**. Can a standardized report card + label-geometry simulation quantify when the design-sensitive gap is large, small, or near zero by construction?

## 2. Main-matrix inclusion (GB v2)

| Criterion | Rule |
|-----------|------|
| Public data | Manifest + labels downloadable or mirrored |
| Label-assignment level | Documented |
| **Non-singleton primary rows** | ≥10 groups with ≥2 genomes/group (or documented lineage block with ≥10 multi-member clusters) |
| Features | 4-mer and/or frozen NT-500m; same probe contract |
| Splits | Random KFold + GroupKFold by prescribed block unit |
| Metrics | Spearman ρ for cross-task comparability; **AUROC (+AUPRC when binary)** for classification tasks |
| No post-hoc tuning | Thresholds locked before reading blocked scores |

**Construction controls (not primary evidence):** tasks where block units are all singletons (random ≈ blocked by construction). Report in Supplementary Table only.

## 3. Locked task panel (GB v2)

| ID | Task | Block unit | Role | Status at lock |
|----|------|------------|------|----------------|
| T01 | Domain C OGT (100 spp) | species | group-constant + replication | Done |
| T02 | Viral SpillOver | organism | extreme group-constant / contamination | Done |
| T03 | Mollentze InfectsHumans (full) | species | mixed replication; weak signal | Done |
| **T03_REP** | Mollentze n≥2 species only | species | non-singleton classification | **Pending** |
| T04_ALT | Pa ceftazidime | PATRIC strain | **construction control** (302/302 singleton) | Done → demote |
| T05 | Babayan host (species block) | species | **construction control** (440/440 singleton) | Done → demote |
| **T06** | Hu Pa ceftazidime | Mash/k-mer cluster | isolate labels + lineage structure | **Pending** |
| **T07** | Domain B OGT (20 spp) | species | second species-constant panel (BacDive deferred) | Done (ingest) |
| **T08** | Babayan host | Viral group | lineage-level block (non-singleton) | **Pending** |

**Amendment (BacDive / T07):** Public BacDive trait × RefSeq multi-genome panel deferred (download/API risk). Pre-registered fallback: Domain B OGT pilot (560 genomes / 20 species; already executed under `second_domain_bacteria/docs/PREREGISTRY.md`).

**Amendment (T08 vs T05_ALT):** PATRIC clinical flag deferred. Pre-registered substitute: Babayan reservoir-host labels with **Viral group** as GroupKFold unit (multi-species groups).

## 4. Simulation (locked before empirical re-run)

Module: `audit_toolkit/simulate_label_geometry.py`  
Factors: `n_groups`, `genomes_per_group`, `icc` (group-constant vs varying), within-/between-group feature correlation, `effect_size`, `label_noise`.  
Primary output: Δρ (random − blocked) heatmaps. Purpose: show gap is driven by label geometry, not report-card bias.

## 5. Probe contract

- `probe_lib.oof_ridge_spearman`, `SEED=42`, `RidgeCV`, fold-wise `StandardScaler`, K=5  
- Binary tasks: also report OOF AUROC from the same predictions  
- Probe analyses are **not** independent model validation

## 6. Decision rules

| Outcome | Criterion |
|---------|-----------|
| Framework supported | ≥4 primary non-singleton rows show geometry-consistent patterns (large gap under group-constant replication; small gap under weak signal; lineage block changes estimand vs species-singleton) |
| Negative / boundary | Δ < 0.10 rows reported honestly |
| No universal claim | Do not claim all genome ML tasks have large gaps |
| Construction controls | T04_ALT / T05 species-block excluded from “framework validated” numerator |

## 7. Outputs

| Artifact | Path |
|----------|------|
| This prereg | `docs/PREREGISTRY_MULTI_TASK_BENCHMARK_GBv2.md` |
| Registry | `tables/Table_multi_task_benchmark_registry.csv` |
| Matrix (primary + controls) | `tables/Table_multi_task_matrix_summary.csv` |
| Simulation | `tables/Table_simulation_label_geometry.csv` + figures |
| Per-task JSON | `multi_task_audit/results/<id>_report_card.json` |
| Robustness (post-matrix) | `tables/Table_robustness_split_design.csv`; `multi_task_audit/results/robustness/` |

## 8. Amendment timeline

| Date | Event |
|------|-------|
| 2026-08-31 | CG v1 prereg locked; T01–T05 executed; T04→T04_ALT (E. coli amp QC fail) |
| 2026-08-31 | **GB v2 locked** before T03_REP / T06 / T08 runs; T07 = Domain B fallback; T04_ALT/T05 demoted to construction controls |
| 2026-09-01 | **Amendment (statistical robustness):** After matrix execution, add group-aware locked-α repeated random vs blocked CV, genome-pooled + group-macro Δ with percentile CIs, and LOGO for small-n_groups tasks. Does not change inclusion rules or primary matrix task IDs. Script: `scripts/run_robustness_suite.py`. |

---

*Prior CG amendment retained:* 2026-08-31 — T04 E. coli ampicillin NCBI-mapped n=253, susceptible=43 (<50). Matrix row as T04_ALT Pa ceftazidime.
