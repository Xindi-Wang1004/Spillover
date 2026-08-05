# ORF1ab position-permutation audit (2026-08-04)

- Input: `sliding_fragments_reannotated.csv` (n=10 genomes × 3 windows)
- Observed 2×2 (polymerase_replicase on peak fragment ≥50%): high 8/10, low 2/20 → Haldane OR = 25.160, Fisher p = 0.000291
- Raw OR = 36.0
- Position permutation: B=10000, seed=42; empirical two-sided p = 0.0043
- Null Haldane OR mean = 1.667; 2.5–97.5% = [0.191, 5.296]
- Top-window index counts: {'0': 1, '1': 8, '2': 1}
- Low windows with ≥50% ORF1ab peak-frag: 2/20 (mean frac_pol low = 0.144)
- **Gate: `keep_OR36_primary`**

Files: `Table_S_orf1ab_window_coordinate_map.csv`, `Table_S_orf1ab_position_perm_summary.csv`, `Table_S_orf1ab_position_perm_summary.json`.
