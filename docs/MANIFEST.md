# Release manifest (curated)

## Included analysis directories

- `analysis_p0p1_circularity_baselines_wxd0804` — Supp P0/P1
- `analysis_sb_followups_wxd0803` — Supp S35–S37
- `analysis_orf1ab_posperm_wxd0804` — Supp S43
- `analysis_picorna_prespec_wxd0804` — Rhinovirus C exemplar
- `analysis_cov_dense_windows_wxd0804` — CoV dense sensitivity
- `analysis_host_interaction_enrichment` — sliding-window / enrichment meta

## Included paper scripts

- `run_attribution_enrichment_host_interaction.py`
- `run_known_loci_retrospective_from16.py` (if present)
- `run_multi_family_sliding_ig_wxd0728.py` (if present)

## Excluded patterns

- `*讨论后修的版本*`
- `*reviewer_checklist*`
- `*pre_*` manuscript backups
- `*.log`, large `.npy`/`.npz` null stores
- Entire unfiltered `bib_tables/` (whitelist only in `tables/`)

## De-duplication rule

Each supplementary CSV lives **once** under `tables/`. Analysis folders keep **code + run_meta.json** only; large regeneratable outputs are omitted when the table copy exists.
