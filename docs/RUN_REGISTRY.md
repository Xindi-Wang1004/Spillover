# Run registry (Spillover companion analyses)

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
| Host-only embedding export equivalence (A4 gate) | `../../sb0731/upgrade_post_sb_reject/analysis_A4_external_sequences/embeddings/Table_A4_equivalence_report.json` (+ `EQUIVALENCE_PASS.md`) — **pass** 2026-08-25 on `10.40.1.16` GPU3; 10/10 max\|Δ\|=0 incl. OW483755 / MK035747 | Locks path that produced `embeddings_classification_20260514_201157.csv` (per-accession max\|Δ\|); **required before A4** |
| A4 pilot-40 host-only export | `../../sb0731/upgrade_post_sb_reject/analysis_A4_external_sequences/embeddings/embeddings_A4_hostonly_pilot40_meta.json` (+ `PILOT40_E2E_REPORT.md`, `embeddings/pilot40_sanity.json`) | Same locked path as equivalence: host `10.40.1.16`, `CUDA_VISIBLE_DEVICES=3`, conda `evo_design`, wrapper `scripts/extract_with_numpy_compat.py` → `extract_genome_embeddings.py --model-tag classification`; input `manifest/pilot40_as_test.csv` (FASTA→Sequence, no revcomp); join labels from `Table_A4_pilot_subset.csv`; sanity 512-d / no-NaN / label-map **pass** 2026-08-25 |
| A4 full-841 host-only export | `../../sb0731/upgrade_post_sb_reject/analysis_A4_external_sequences/embeddings/embeddings_A4_hostonly_full841_meta.json` (+ `embeddings/Table_A4_oof_metrics_full841.json`, `embeddings/full841_sanity.json`) — **done** 2026-08-25 ~20:32 server; 841/841 sanity pass | Identical env/wrapper/GPU as pilot; input `manifest/full841_as_test.csv`; join `Table_A4_accession_manifest.csv`; organism-blocked OOF AUC≈0.52, ρ≈0.02 → `SCLS_cap_lineage_specific` |
| R3 second-backbone NT gap profile (GB) | `../../sb0731/upgrade_post_sb_reject/analysis_GB_fillins/R3_CONTRACT.md` (+ `results/Table_R3_gap_profile_NT.json`) | Same `probe_lib.py` as p0p1 (RidgeCV/seed42); NT window 6000/3000/256 PRESPECIFIED (≠ Evo); model 2.5B if VRAM≥60GB else 500m; download via hf-mirror |
| SpillOver score variance decomposition (GB §2.6) | `../../sb0731/001_transfer/02_upgrade_evidence/GB_fillins/results/Table_variance_decomposition_ICC.json` (source: `analysis_p0p1_circularity_baselines_wxd0804/Table_P0_overlap_with_ablated_scores.csv`) | ICC at 25 organism groups (≈0.998) and 7 families (Total 0.76 / Total′ 0.88); pairs with S38 organism-blocked ablation 0.311→0.082 |
| Group-level primary estimand + bootstrap (GB §2.4) | `../../sb0731/001_transfer/02_upgrade_evidence/GB_fillins/results/Table_group_bootstrap_primary_estimand.json` — **done** 2026-08-27 on `10.40.1.16`; conda `evo_design`; regression embeddings + server overlap632 + GenBank years | 25 organism groups; B=2000 bootstrap on group means; random / organism-blocked / family LOFO / temporal organism-disjoint |
| Internal dev vs eval leakage audit (GB) | `../../sb0731/001_transfer/02_upgrade_evidence/GB_fillins/results/Table_R2_internal_402_vs_632_summary.json` (+ `Table_R2_internal_402_vs_632_nn.csv`) — **done** 2026-08-27 on `10.40.1.16` | 7-mer MinHash (stride 31, 128 hashes); dev `processed_ref_train_with_spillover.csv` (n=511) vs eval overlap632 (n=632); 111 exact MD5 duplicates |
| Leakage training-fold qualification (GB) | `../../sb0731/001_transfer/02_upgrade_evidence/GB_fillins/results/Table_leakage_training_fold_qualify.json` — **done** 2026-08-27 | Claimed clean 402 vs on-disk ensambled 721 (196∩eval) / ref 511 (111∩eval); log `train_genomes=612` → **must retrain** |
| Clean train manifest (retrain gate) | `../../sb0731/001_transfer/02_upgrade_evidence/GB_fillins/manifest/Table_train_manifest_audit.json` (+ accession list); full CSV on GPU16 `…/analysis_GB_fillins_wxd0827/…/manifest/train_manifest_accession_disjoint_clean.csv` — **LOCKED** n=398, gates all pass | Accession∩eval=0; MD5∩eval=0; NN≥0.95/0.99 cross pairs=0; dropped 2 NN leaks (PP874408, MW727454) |
| Locked training config for clean retrain | `../../sb0731/001_transfer/02_upgrade_evidence/GB_fillins/LOCKED_TRAINING_CONFIG_retrain_v1.json` | Window 8192/4096/256 from nohup; LoRA r=8 α=32 from script defaults; MT α=0.5 from log; unresolved: why original train_n=612 |
| Clean retrain | `10.40.1.16:/home/wangxindi/evo-main/train_fullgenome_notrunction/clean_retrain_wxd0827/` (+ `GATE_PASS_train_manifest.json`, logs) — **done** 2026-08-28; reg/cls/mt 3×epoch; wrapper `train_with_numpy_compat.py` | Manifest n=398; checkpoints: `best_genome_fusion_20260827_070941.pth`, `best_classifier_genome_fusion_20260827_074305.pth`, `best_multitask_genome_fusion_20260827_071006.pth` |
| Clean recalc postprocess (GB) | `10.40.1.16:…/clean_recalc_wxd0828/` + local `02_upgrade_evidence/GB_fillins/results/Table_clean_recalc_bundle.json` — **done** 2026-08-30 | Equiv×3 PASS; emb×3+841; infer×3; Table1/2/3+Total′ bundle; group/R1/sanity; manuscript still frozen |
| Phase 8 attribution (clean CoV IG) | `…/attribution_clean/` + `Table_R4R5_genome_level_attribution_clean.json` — **done** 2026-08-30; ORF1ab best-window **6/10**, OR_MLE≈13.5 (p≈0.007) vs published low pool | Table19 10×3; clean reg ckpt |
| Cautionary rewrite workspace | `../../sb0731/001_transfer/transfer_GB/` — **canonical** `GB_cautionary_full_v1.md`; `00_to_upload` untouched | Title: organism-constant targets; R5 clean null filled |
| Second-domain bacterial OGT (GB cross-domain) | `../../sb0731/001_transfer/transfer_GB/second_domain_bacteria/` — **done** 2026-08-30; **prereg locked 2026-08-30 before probes** (`docs/PREREGISTRY.md`); doc chain `docs/DATA_MANIFEST.md`; QUICKSTART + `docs/ZENODO_PACKING_LIST.md`; extreme-OGT scan **abandoned** (`docs/OGT_EXTREME_SPECIES_SCAN.md`); n=560; GPU16 GPU5 NT ~2.8 h | Intentionally isomorphic mirror; k-mer 0.948→0.058 (Δ0.89); NT 0.947→−0.099 (Δ1.05); both **A_replicates** |
| Second-domain ANI95 sensitivity | `docs/PREREGISTRY_ANI95_SENSITIVITY.md` locked **before** run; Mash-compatible sourmash k=21, d≤0.05; `results/ani95_dedup_stats.json` + `second_domain_summary_kmer4_ani95.json` — **done** 2026-08-30 | Pathogen RefSeq panel collapses to ~1 genome/species (n_after≈23 before min=5 filter); probe-eligible n=0 → **S_underpowered** per prereg; documents within-species ≥95% ANI near-clones; does **not** retract primary A_replicates |
| Domain C large OGT (≥100 spp) | `../../sb0731/001_transfer/transfer_GB/third_domain_ogt_large/` — **prereg locked 2026-08-31** before probes; census 785 eligible → stratified 120 planned → evaluated **100 spp / 777 genomes**; k-mer4 **A_replicates** Δρ≈0.60; NT-500m **A_replicates** (random ρ≈0.92 → blocked ρ≈0.55; Δρ≈0.37) — **done** 2026-08-31 GPU16 | Primary cross-domain statistical support; Domain B = pilot |
| Viral Δρ bootstrap + org-label permutation | `../../sb0731/001_transfer/02_upgrade_evidence/GB_fillins/results/Table_delta_rho_bootstrap.json` + `Table_group_label_permutation.json` + `METHODS_NOTE_stats_pack.md` — **done** 2026-08-31 | Group-resample Δρ CI excludes 0 (Evo/NT); permutation null mean still high |
| Audit report-card toolkit | `../../sb0731/001_transfer/transfer_GB/audit_toolkit/report_card.py` | Overlap + random vs group-blocked Ridge card |
| R5 Rhinovirus C clean IG | `…/attribution_clean/rvc/` + `20_run_rvc_ig_clean.py` — **done** 2026-08-30 (~08:29–11:50 UTC); entry_interface top-window **2/10**, OR_MLE≈0.044 (BH-q≈0.0015); contaminated-era 10/10 OR_H≈21 **not retained** | Locked 10 RV-C accessions; clean reg ckpt; appendix/supplement only |
| Clean early group estimand (clean emb) | `../../sb0731/001_transfer/02_upgrade_evidence/GB_fillins/results/Table_group_bootstrap_primary_estimand.json` (+ `DIAG_group_level_sanity.json`, `Table_group_host_label_AUC.json`) — **preview** 2026-08-29; do not fill manuscript until full pipeline | Clean regression emb; group random ρ≈0.436 (CI crosses 0); sanity random≥organism-blocked PASS |
| Clean R1 frozen transfer preview | `../../sb0731/001_transfer/02_upgrade_evidence/GB_fillins/results/Table_R1_frozen_transfer_clean.json` — AUC≈0.507 | Clean cls 632→841; still chance-level |

## End-to-end predictions (held-out overlap, n = 632)

Stored under `data/`:

- `infer_regression_per_sequence_predictions.csv`
- `infer_classification_per_sequence_predictions.csv`
- `infer_multitask_per_sequence_predictions.csv`
- `overlap_cohort.csv`

## Model checkpoints

Large `.pth` weights are **not** in this git tree. Download from Zenodo  
DOI [10.5281/zenodo.21809791](https://doi.org/10.5281/zenodo.21809791); verify SHA256 in `models/checkpoint_manifest.json`.
