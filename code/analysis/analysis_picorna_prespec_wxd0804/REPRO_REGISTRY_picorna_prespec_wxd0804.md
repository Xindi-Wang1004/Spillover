# Reproducibility Registry — Picornaviridae / Rhinovirus C second-family prespecification

**Frozen before analysis run:** 2026-08-04 (local draft lock; analysis run immediately after this file was written).  
**Status:** PRESPECIFIED — do not alter primary endpoint after unblinding results.

## Panel

- Species: Rhinovirus C (Picornaviridae)
- Source: `/home/wangxindi/evo/evo_data/ig_species_wxd0729/Rhinovirus_C/`
- Accessions (n=10): KF958310, KF958311, KJ675505, KJ675506, KJ675507, KP890662, KP890663, KY369879, KY369880, MH330336
- Annotation coverage table: `annotation_mask_coverage.csv` (frac_entry ≈ 0.35; frac_polymerase ≈ 0.19)

## Asset audit (pre-run)

- Sliding-window IG long tables present for **regression** and **multitask** (3 windows/genome; max_ig_len=4096).
- Dense stitched attribution tracks for these accessions: **NOT FOUND** (2026-08-04 search under `evo_data` / paper dirs).
- Therefore dense equal-partition W∈{5,7,10} **cannot** be executed without new GPU IG. Rebuilding “pseudo-dense” tracks from three overlapping 4096-nt windows is **not allowed** (would invent within-window spatial resolution).

## Prespecified primary analysis (locked)

Mirrors the **main-text CoV OR=36 design** (sparse sliding windows + peak-fragment class), not the CoV dense-track sensitivity table.

| Item | Locked choice |
|---|---|
| Model | **regression** (primary); multitask = descriptive concordance only |
| Windows | Existing 3 sliding windows per genome (`sliding_ig_*_windows_long.csv`) |
| High definition | Per genome, window with max `window_attr_sum` |
| Peak fragment | `top_fragment_start`–`top_fragment_end` already in table (250 nt) |
| Class map | Prespecified three classes: **entry_interface** (capsid/VP-region); **polymerase_replicase** (replication/3Dpol); **other** |
| Primary contrast | Within-genome Fisher exact, high vs rest, for each class |
| OR | Woolf–Haldane corrected ((a+0.5)(d+0.5)/((b+0.5)(c+0.5))) reported alongside raw Fisher OR |
| Multiplicity | BH-FDR across the **3 prespecified classes** within the primary model (regression) |
| Sign test (optional) | Binomial on high∈entry across genomes; null = mean `frac_entry` from annotation table (≈0.35); **never 0.5** |
| Reporting | Report regression primary whether significant or not; **do not** switch primary to multitask post hoc |

## Deferred (explicitly not primary)

- Dense-track equal partitions W=5 (primary sensitivity) / W=3,7,10 (robustness), matching CoV Table S-CoV: **deferred** until dense stitched IG tracks exist for this panel.
- When dense tracks become available, use the same rules as CoV dense sensitivity (peakfrag 250 nt; coverage-null sign test; W=5 primary among denser sizes) without changing this sparse primary.

## Outputs

- `Table_S_Picorna_per_accession_window_classes.csv`
- `Table_S_Picorna_enrichment_summary.csv`
- `Table_S_Picorna_multitask_concordance.csv` (descriptive)
- `run_meta_picorna_prespec_wxd0804.json`

## Decision rule (post-run, for manuscript placement only)

- If regression entry Fisher is significant after BH across 3 classes → consider promoting a short second-family sentence into §3.5 (editorial decision; not a license to change endpoints).
- If not significant → remain supplement-only / descriptive; do not re-label multitask as primary.
