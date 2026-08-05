# Host-interaction attribution enrichment (Analysis 1)

**Host:** `wangxindi@10.40.1.16` (`~/evo-main`)  
**Script:** `paper/run_attribution_enrichment_host_interaction.py`  
**Run meta:** `paper/attribution_enrichment_run_meta.json`

## What this upgrades

Previous Supplementary IG+BLAST (four genomes / CoV spike-prior windows) showed that
high-attribution 250 nt fragments BLAST back to the expected virus. That only shows
the model attends to *viral* sequence, not that it attends to *host-interaction*
functional regions.

This analysis maps attribution to GenBank-derived functional masks and adds:

1. length-aware mass enrichment
2. circular-shift permutation over genomic positions
3. random fragment-placement nulls
4. high-attr vs low-attr window contrast (Fisher)

## Functional ontology

| Class | Contents |
|-------|----------|
| `entry_interface` | Spike / S / surface glycoprotein / RBD |
| `polymerase_replicase` | ORF1ab / polyprotein / nsp / RdRp / helicase |
| `accessory_immune` | ORF3/6/7/8/10 and immune-related accessories |
| `host_interaction_broad` | union of the three above |
| `structural_other` | E / M / N |

## Primary results (unbiased sliding IG; Table19; n=10 CoV × 3 windows)

### High-attr vs low-attr windows (Table31) — main inferential contrast

| Class | High in | High out | Low in | Low out | OR | Fisher p |
|-------|---------|----------|--------|---------|----|----------|
| entry_interface | 1 | 9 | 2 | 18 | 1.0 | 1.0 |
| **polymerase_replicase** | **8** | **2** | **2** | **18** | **36.0** | **2.9×10⁻⁴** |
| **host_interaction_broad** | **9** | **1** | **8** | **12** | **13.5** | **1.7×10⁻²** |

Interpretation: relative to low-attribution windows on the same genomes, top fragments
from high-attribution windows are strongly enriched in **replicase/ORF1ab** and the
broader host-interaction union, but **not** specifically in spike/entry.

This is consistent with Table22 (CDS-pooled IG mass dominated by ORF1ab) and argues
against a spike-only mechanism story.

### Random genomic placement (Table30) — length null

Because ORF1ab spans ~70% of CoV genomes, random 250 nt draws already hit polymerase
often. Observed top-250 placement is therefore **not** above a length-matched random
null for polymerase/entry (p ≳ 0.4–1.0). The informative control is the
**high vs low attribution contrast**, not raw occupancy vs genome length.

### Spike-prior windows (Table28/29) — within-window only

Existing `ig_blast_results_cov_subset` tracks used a spike-centered hint
(window ≈ 21452–25548). These tests are **within-window** only and cannot claim
genome-wide spike enrichment. Several accessions lack complete S features in cached
GenBank (entry fraction = 0); treat those rows cautiously.

Where S is annotated inside the window, entry mass enrichment is mixed (some
accessions p_shift < 0.05; median across accessions is not uniformly > 1). RBD-specific
signal is generally weak in these tracks.

## Outputs

| File | Role |
|------|------|
| `bib_tables/Table27_host_interaction_region_lengths.csv` | Region length fractions |
| `bib_tables/Table28_spike_window_attr_enrichment.csv` | Within spike-prior window enrichment |
| `bib_tables/Table29_spike_window_permutation.csv` | Circular-shift p-values |
| `bib_tables/Table30_sliding_fragment_placement_enrichment.csv` | Placement vs random null |
| `bib_tables/Table31_high_vs_low_attr_window_classes.csv` | **Primary** high vs low Fisher tests |
| `bib_figures/SupplementaryFigure_S9_host_interaction_enrichment.png` | Summary figure |
| `analysis_host_interaction_enrichment/sliding_fragments_reannotated.csv` | Overlap-based re-annotation |

## Suggested manuscript wording (conservative)

> Using unbiased sliding-window Integrated Gradients on a coronavirus subset, top
> fragments from high-attribution windows were enriched for replicase/ORF1ab relative
> to low-attribution windows on the same genomes (odds ratio = 36; Fisher p = 2.9×10⁻⁴),
> and for a broader host-interaction annotation union (OR = 13.5; p = 1.7×10⁻²), but not
> for spike/entry alone. Because ORF1ab occupies most of the CoV genome, occupancy
> against a length-matched random null was non-significant; the high-versus-low
> attribution contrast is therefore the primary enrichment test. These results support
> a distributed, replication-complex–weighted attribution pattern rather than a
> spike-only host-entry narrative, and supersede BLAST identity checks as evidence of
> functional localization.

## Re-run

```bash
conda activate evo_design
cd ~/evo-main
python paper/run_attribution_enrichment_host_interaction.py --n-perm 2000 --seed 42
```

## Next (in progress / recommended)

1. Dense sliding IG (`run_sliding_ig_dense_for_enrichment.py`, 7 windows) → genome-wide
   stitched tracks + circular-shift enrichment without spike prior.
2. Known host-adaptation loci retrospective check (Analysis 3).
3. Extend ontology + enrichment beyond Coronaviridae (family-stratified).
