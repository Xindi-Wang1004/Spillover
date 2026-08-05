# Molecular-layer analyses (node16 / evo-main)

## Status

| Analysis | Status | Outputs |
|----------|--------|---------|
| 1 Attribution enrichment + permutation | **done** | Table27–31, Fig S9 |
| 2 Family-controlled k-mer + AA properties | **done** | Table34–36, Fig S11 |
| 3 Known host-adaptation loci | **done** (refresh after dense IG) | Table32–33, Fig S10 |
| 4 Lightweight PDB mapping | **done** | Table37, Fig S12 |
| 5 Cross-family synthesis | **done** | Table38, Fig S13, `MOLECULAR_LAYER_SYNTHESIS.md` |
| Dense 7-window genome-wide IG | **running** (cuda 1/2/4/5/6) | → Table39–40 |

## Primary claims ready for text

1. **Distributed / polygenic signal** — single-window ablation drops regression Spearman 0.678 → 0.593.
2. **CoV replication-complex weight** — high- vs low-attribution windows enrich polymerase/ORF1ab (OR=36, Fisher p=2.9×10⁻⁴), **not** spike/entry alone.
3. **Family-controlled motifs** — FDR<0.05 5-mers in 6 families; no single universal k-mer.
4. **Known loci** — strongest current support for ORF1ab C-terminal / RdRp-like intervals; classic RBD/HA RBS weak under current (often prior-window) tracks.
5. **Structure** — RdRp PDB anchors have attribution support; avoid ΔΔG / “destabilization causes spillover” wording.

## When dense IG finishes

```bash
conda activate evo_design
cd ~/evo-main
python paper/run_dense_stitched_enrichment.py --n-perm 2000
python paper/run_known_loci_retrospective.py --n-perm 1000
python paper/run_structure_mapping_light.py
python paper/run_cross_family_common_factors.py
```

Monitor: `tail -f /home/wangxindi/evo/evo_data/ig_sliding_dense/logs/dense_ig_cuda*.log`
