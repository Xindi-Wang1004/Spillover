# GenomeML Report Card: an executable framework for estimand-matched evaluation of genome machine learning

Xindi Wang¹,²,³, Junyu Luo³,\*, Yixue Li³,⁴,⁵,⁶,\*, Chitin Hon¹,²,\*

¹ Faculty of Innovation Engineering, Macau University of Science and Technology, 999078, Macao Special Administrative Region of China
² Institute of Systems Engineering, Macau University of Science and Technology, 999078, Macao Special Administrative Region of China
³ Guangzhou National Laboratory, No. 9 XingDaoHuanBei Road, Guangzhou International Bio Island, Guangzhou 510005, Guangdong Province, China
⁴ GMU-GIBH Joint School of Life Sciences, The Guangdong-Hong Kong-Macau Joint Laboratory for Cell Fate Regulation and Diseases, Guangzhou Medical University, Guangzhou 511436, China
⁵ Key Laboratory of Systems Health Science of Zhejiang Province, School of Life Science, Hangzhou Institute for Advanced Study, University of Chinese Academy of Sciences, Hangzhou 310024, China
⁶ Shanghai Institute of Nutrition and Health, Chinese Academy of Sciences, Shanghai 200030, China

ORCID: Xindi Wang, 0009-0003-5042-9534; Junyu Luo, 0000-0003-2206-1931; Yixue Li, 0000-0002-1198-7176; Chitin Hon, 0000-0002-5658-3503

\* Corresponding authors: Junyu Luo (luo_junyu@gzlab.ac.cn); Yixue Li (li_yixue@gzlab.ac.cn); Chitin Hon (cthon@must.edu.mo)

---

## Abstract

Genome machine-learning datasets often contain multiple sequences from the same biological entities, including species, organisms, and viral lineages. Consequently, sequence-level random cross-validation may evaluate prediction for groups already represented in training rather than transfer to groups excluded from training. Here we present **GenomeML Report Card**, an executable framework that records label-assignment units and evaluation blocks, audits sequence overlap and group recurrence, and compares random and pre-specified blocked evaluation under a common reporting schema.

Simulations show that random-to-blocked performance differences depend on within-group label homogeneity and replication, rather than representing a universal measure of leakage. Across microbial trait prediction, viral host prediction, and antimicrobial-resistance tasks, random and blocked splits produced task-dependent contrasts. In a temporal species-disjoint bacterial growth-temperature holdout, external performance aligned more closely with species-blocked than random cross-validation. GenomeML Report Card provides versioned manifests, machine-readable audit reports, and tiered reporting items to support explicit claims about held-out-group generalization.

**Keywords:** genome representation learning; estimand-matched evaluation; group-level estimand; label-assignment unit; split-design performance gap; data-integrity audit; benchmark resource

**Graphical abstract.** Random genome splits are not intrinsically invalid—they answer a different estimand. An audit report card quantifies when they estimate seen-group recognition rather than block-defined held-out-group performance, and documents label geometry before interpreting split-design contrasts.

---

## 1 Introduction

### 1.1 Formal estimands and blocking layers

We distinguish two complementary notions that are often conflated in genome ML evaluation:

**Layer A — label-assignment groups.** Labels are inherited or nearly constant within a biological unit \(g\in\mathcal{G}_{\mathrm{label}}\) (species-constant OGT [14], organism-constant SpillOver ranks [2], species-level virus phenotypes [3,4]). When \(\mathcal{G}_{\mathrm{label}}\) has multi-member replication and a high **random-CV shared-block fraction** (Table 1a), sequence-level random CV predominantly estimates performance **conditional on partial representation of test groups in training**—not automatic equivalence to a formal \(\theta_{\mathrm{seen}}\), but the operational regime the audit diagnoses.

**Layer B — prediction-dependence / deployment blocks.** A pre-specified block unit \(B\) (k-mer cluster, viral lineage, temporal cohort) defines a held-out target even when labels vary within blocks. The logic is correlation, taxonomic recurrence, composition similarity, or population structure—not label copying. **AMR–composition-cluster (T06)** and **viral-host–lineage (T08)** [4] belong here.

Random splits are not intrinsically invalid; they answer a different question. **Group-blocked** splits (GroupKFold / LOGO on block \(B\)) target held-out performance **at that blocking level**—not automatically phylogenetic novelty, temporal deployment, or arbitrary out-of-distribution transfer. Block choice is a scientific/deployment decision that must be declared before interpreting scores.

Let \(A\) be a learning algorithm, \(\Pi_{\mathrm{random}}\) a sample-level random partition, \(\Pi_B\) a block-level partition on unit \(B\), and \(M\) a dataset-level metric (AUROC, Spearman \(\rho\)). For cohort \(D\) and fold assignment \(\pi\):

\[
\theta_{\mathrm{random}}(A,\Pi_{\mathrm{random}},M)
=
\mathbb{E}_{D,\pi}
\left[
M\big(\{(y_i,\hat y_i): i\in \mathrm{test}(\pi)\},\, A(D_{\mathrm{train}(\pi)})\big)
\right].
\]

\[
\theta_{\mathrm{blocked},B}(A,\Pi_B,M)
=
\mathbb{E}_{D,\pi_B}
\left[
M\big(\{(y_i,\hat y_i): i\in \mathrm{test}(\pi_B)\},\, A(D_{\mathrm{train}(\pi_B)})\big)
\right].
\]

Under Layer A, we use \(\theta_{\mathrm{seen}}\) as a conceptual shorthand for performance conditional on representation of the relevant label-assignment group in training; random sequence-level CV may approximate it when groups have multi-member replication and high train–test group recurrence, and the report card measures these conditions rather than assuming them. \(\theta_{\mathrm{unseen}}(B)\) (shorthand \(\theta_{\mathrm{blocked},B}\)) denotes the performance induced by the declared blocked partition—not a universal out-of-distribution estimand. The **split-design contrast** \(\Delta_B=\theta_{\mathrm{random}}-\theta_{\mathrm{blocked},B}\) is a **within-task diagnostic** on the task-primary metric; \(\Delta_B\) is not a universal biological effect size and must not be compared across tasks with different metrics (e.g. \(\Delta\rho\) vs \(\Delta\)AUROC).

Related concerns include phylogenetic non-independence [9,10] and pseudoreplication [15]. **Sequence contamination** (accession/exact-duplicate/near-duplicate train–test overlap [13]) is audited separately from **cross-fold group recurrence** and **evaluation-target mismatch** (Fig. 1).

### 1.2 Relation to existing practice

Group-aware evaluation is established (homology/identity-cluster splits [7,8], taxonomic hold-outs, phylogenetic CV [9,10]). **Our contribution is not “first to propose group split.”** We provide an **executable audit-and-reporting framework** that requires explicit label-assignment and blocking units, separates sequence contamination from group recurrence, compares random and blocked estimands side-by-side, and ships machine-readable manifests. Related genome language models such as DNABERT [5], Nucleotide Transformer [6], and Evo [1] motivate the need for estimand-matched evaluation when labels are assigned at biological-group levels.

### 1.3 Resource and evidence overview

We release **GenomeML Report Card** (`genome-ml-reportcard` CLI; GitHub repository), a label-geometry simulator, version-locked benchmark manifests, and a reproducibility hash table. Primary evidence follows four chains: (i) **concept**—Layer A vs Layer B; (ii) **tool**—automated overlap, geometry, and split comparison; (iii) **simulation**—contrast depends on homogeneity and replication; (iv) **empirical audit**—multi-task panel (Fig. 3) plus temporal species-disjoint external holdout on **OGT–species (T01)** [14]. Supplementary materials include audit demonstrations on public datasets linked to published benchmarks, model-class robustness, and a SpillOver viral integrity audit case [2] (Supplementary Fig. S_integrity).

---

## 2 Results

### 2.1 GenomeML Report Card

Given a manifest with genomes, labels, a label-assignment column, and a blocking column, GenomeML Report Card reports: (1) accession/exact-sequence and near-neighbor overlap; (2) within-block label homogeneity and random-CV shared-block fraction; (3) random vs group-blocked probe scores on the task-primary metric; and (4) machine-readable JSON/Markdown audit output (Fig. 1).

Minimal manifest fields are `sequence_id` (or `accession`), `label`, `label_group`, and `evaluation_block`; `accession` and `sequence` are optional but recommended for contamination audits. `label_group` may equal `evaluation_block` (Layer A) but need not (Layer B); the framework requires both to be declared explicitly.

Install with `pip install genome-ml-reportcard` (PyPI) or `pip install -e audit_toolkit/` from the paper repository. Minimal usage:

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

Key output fields include overlap counts (with `--table-b`), label-geometry fields (`within_block_homogeneity`, `random_cv_shared_block_fraction`, singleton-block fraction), random vs blocked probe scores on the task-primary metric, and the split-design contrast \(\Delta_B\).

### 2.2 Simulation establishes when contrasts arise

With a locked Ridge out-of-fold probe (Fig. 2), the split-design gap under group-constant labels (ICC = 1) rises with genomes per group (mean \(\Delta\rho\approx0.77\) at 5 genomes/group), is near zero when ICC = 0 at the same replication, and is near zero for singleton groups under ICC = 1. The framework therefore diagnoses geometry-dependent contrasts rather than inventing gaps independent of label structure.

### 2.3 Label geometry and multi-task audit matrix

Split-design contrasts require multi-member blocks, sufficient within-block label homogeneity, and cross-fold block recurrence under random CV (Table 1a).

| Task | Layer | Label unit | Block unit | Within-block homogeneity | Random CV shared-block |
|---|---|---|---|---:|---:|
| OGT–species (T01) | A | species | species | ICC = 1.0 | 1.00 |
| Viral-host–lineage (T08) | B | species | Viral group | purity = 0.75 | 1.00 |
| AMR–composition-cluster (T06) | B | isolate | k-mer cluster | purity = 0.74 | 0.99 |
| Virus phenotype–species (T03_REP) | A | species | species | purity = 1.0 | 0.96 |

**OGT–species (T01)** [14] and **virus phenotype–species (T03_REP)** [3] are Layer A (label inheritance; ICC/purity = 1.0). **Viral-host–lineage (T08)** [4] is Layer B: reservoir labels are species-constant, but Viral-group blocks are label-heterogeneous (majority purity ≈ 0.75; 12 groups—exploratory scale, not broad biological replication). **AMR–composition-cluster (T06)** is Layer B: isolate-level labels with clusters defined from sequence-composition features (Supplementary Table S_T06)—an **illustrative operational block** that represents a declared stress-test target rather than a validated phylogenetic or clinical deployment partition.

**Primary evidence** uses repeated random vs group-blocked CV with group-aware locked Ridge \(\alpha\) (Table 1; Fig. 3). Historical single-seed estimates are Supplementary only.

| Task | Layer | Block | Metric | Random mean | Blocked mean | Mean \(\Delta\) | SAI‡ | Repeats |
|---|---|---|---|---:|---:|---:|---|---:|
| OGT–species (T01) | A | species | \(\rho\) | 0.869 | 0.382 | 0.486 | [0.41, 0.57] | 15 |
| Viral-host–lineage (T08) | B | Viral group | AUROC | 0.768 | 0.528 | 0.240 | [0.16, 0.33] | 40 |
| AMR–composition-cluster (T06) | B | k-mer cluster | AUROC | 0.580 | 0.510 | 0.071 | [0.02, 0.13] | 30 |
| Virus phenotype–species (T03_REP) | A | species | AUROC | 0.700 | 0.621 | 0.079 | [0.05, 0.11] | 20 |

‡Split-assignment interval: 2.5–97.5% across repeated fold assignments; quantifies partition sensitivity on the observed cohort, not population-level inferential uncertainty (Methods 4.2).

**Interpretation.** Large contrasts appear under Layer A species-constant traits with cross-fold species recurrence (T01). In tasks with lower within-block label homogeneity, weaker predictive signal, or limited effective group replication, the random-to-blocked contrast was correspondingly small (T03_REP, T06). **Viral-host–lineage (T08)** supports that lineage-blocked and random splits target different estimands within this small-group setting; it does not support broad viral host-transfer claims. Blocking units are not interchangeable—species, viral-group, and composition-cluster blocks estimate different held-out targets (Supplementary Table S_blocking).

Fig. 3 displays paired random vs blocked scores **per task in separate panels** with task-native metrics; \(\Delta_B\) is a within-task diagnostic and must not be read as a cross-task leakage league table.

### 2.4 External deployment validation

To test whether a declared block approximates deployment-relevant generalization, we evaluated an external holdout on **OGT–species (T01)** [14] with the same k-mer Ridge probe (Table 2; Supplementary Fig. S_external). All species whose first RefSeq `seq_rel_date` is ≥2015 were withheld from training (524 test genomes, 63 species). External Spearman ρ≈**0.29** (species-level bootstrap 95% CI 0.07–0.49; genome-level 0.23–0.36), versus random CV ρ≈0.86 and species-blocked CV ρ≈0.34—in this temporal species-disjoint panel, external performance was substantially closer to species-blocked than to random cross-validation. `seq_rel_date` is a database-entry proxy for entry of taxa into the reference panel, and temporal shifts in assembly quality or sampling cannot be fully excluded. Geographic holdouts on Domain B OGT (T07; supplementary) show that country blocking without species exclusion does not imply novel-species transfer under species-constant labels.

| Panel | Holdout | n_test | Random CV | Species-blocked CV | External |
|---|---|---:|---:|---:|---:|
| OGT–species (T01) | species first sequenced ≥2015 | 524 | 0.86 | 0.34 | **0.29** |

### 2.5 Reporting schema and software availability

We propose tiered reporting items for claims of block-defined held-out-group generalization (Fig. 4): **minimum** fields (label unit, block unit, overlap, random + blocked scores, version-locked manifests); **strongly recommended** (within-block homogeneity, group-balanced secondary metrics); **conditional** (near-neighbor screen for fine-tuned checkpoints). Model-class robustness, split-assignment sensitivity, audit demonstrations on public benchmark datasets (Supplementary Table S_audit), and the SpillOver viral integrity audit [2] (Supplementary Fig. S_integrity) are in Supplementary Materials.

---

## 3 Discussion

Genome ML training–test partitions must match the claimed generalization target. Sequence overlap, cross-fold group recurrence, and estimand mismatch are distinct audit targets and should be reported separately. Under Layer A, when label-assignment groups recur across random folds with high homogeneity, random splits predominantly measure performance conditional on groups already partially represented in training—not held-out-group transfer without an explicit block declaration. Under Layer B, blocks define deployment-relevant holdout units even when labels vary within blocks. **Group-blocked CV is not automatically “true OOD”**; it estimates performance at the declared block level, which may or may not match a given deployment setting.

The split-design contrast \(\Delta_B\) is a **within-task diagnostic**, not a universal leakage score or cross-task league table. Repeated split-assignment sensitivity, external holdout on **OGT–species (T01)**, and multi-model checks (Supplementary) show that large T01 contrasts are not single-split artifacts; in tasks with lower homogeneity, weaker signal, or limited replication, contrasts were correspondingly small. **The tool does not choose the “correct” block**—it requires users to declare the deployment estimand and audit consistency with the split design. The report card is model-agnostic; the present empirical panel uses standardized k-mer Ridge probes to isolate split-design effects under a common modeling contract.

Limitations: **Viral-host–lineage (T08)** has only 12 viral groups with ≈75% within-group label purity—illustrative, not scalable biological replication. **AMR–composition-cluster (T06)** uses composition clusters as operational blocks. Frozen k-mer probes are not end-to-end fine-tuning validation. External temporal validation currently covers T01 only. PyPI/Zenodo release pins should be refreshed on the submission tag.

---

## 4 Methods

Version-locked analysis plan: `docs/PREREGISTRY_MULTI_TASK_BENCHMARK_GBv2.md` (locked 2026-08-31 CST, before T03_REP/T06/T08 execution). Post-matrix robustness and multi-model analyses are logged as amendments in the same file.

### 4.1 Estimands and primary evaluation

\(\theta_{\mathrm{random}}\), \(\theta_{\mathrm{blocked},B}\), and \(\Delta_B\) as in §1.1. Primary summaries (Tables 1, 1a) use repeated CV with group-aware locked Ridge \(\alpha\). **Group-balanced AUROC:** genome-pooled AUROC with sample weight \(w_i=1/n_{g(i)}\). **Group-macro Spearman:** equal weight per group on group-mean predictions (Layer A continuous traits). Frozen k-mer Ridge probes are not end-to-end fine-tuning validation; a Nucleotide Transformer embedding companion for OGT–species is reported in Supplementary materials [6]. Attribution analyses using integrated gradients [11,12] on an Evo-related checkpoint [1] are Supplementary only.

### 4.2 Split-assignment sensitivity

Ridge \(\alpha\) locked once via group-aware GroupKFold MSE; reused across 15–40 repeated random vs group-fold assignments per task. **Split-assignment interval (SAI)** = 2.5–97.5% percentile across repeats on the observed cohort. LOGO diagnostics for T08: Supplementary Fig. S_T08_logo.

### 4.3 Label-geometry diagnostics

ICC (continuous), majority-label purity (binary), median block size, singleton-block fraction, and random-CV shared-block fraction computed uniformly across primary tasks (Table 1a).

### 4.4 Multi-model robustness (supplementary)

Ridge (locked α), logistic regression, and HistGradientBoosting on shared k-mer features for T01, T08, T03_REP.

### 4.5 External holdouts

**OGT–species (T01):** species whose first `seq_rel_date` ≥2015 withheld from training. **Domain B OGT (T07; supplementary):** geographic holdout on BioSample country with and without species-disjoint training exclusion.

### 4.6 Simulation

Label-geometry simulator; factors ICC, genomes/group, within-group feature correlation (`SEED=42`).

### 4.7 Multi-task audit panel

Inclusion: public data; documented label-assignment level; primary rows require non-singleton blocks. **Primary panel:** OGT–species (T01) [14]; viral-host–lineage (T08) [4]; AMR–composition-cluster (T06); virus phenotype–species (T03_REP) [3]. **Supplementary:** Domain B OGT (T07); SpillOver integrity audit (T02) [2]; construction controls T04_ALT/T05. Babayan labels: binary mammal vs other; Orphan excluded. Near-neighbor screens use a 7-mer MinHash Jaccard candidate screen [7,8] (k = 7, stride = 31, n_hash = 128, seed = 42; train–evaluation pairs with J ≥ 0.95 flagged, also reported at J ≥ 0.99). The screen is an inexpensive candidate-detection procedure rather than a definitive homology or contamination call; flagged pairs are reported for optional sequence-alignment or ANI confirmation.

### 4.8 Reproducibility

SHA-256 hashes of manifests and analysis plans (`tables/Table_reproducibility_hashes.json`). Code: https://github.com/Xindi-Wang1004/Spillover (`transfer_GB/` snapshot). Zenodo checkpoints DOI 10.5281/zenodo.21809791. Refresh hashes on submission tag.

---

## 5 Data and resource availability

Code and GenomeML Report Card: https://github.com/Xindi-Wang1004/Spillover (`transfer_GB/` snapshot). Install via `pip install genome-ml-reportcard` (or `pip install -e transfer_GB/audit_toolkit` from the repository). Version-locked manifests, simulation outputs, and hash table accompany the release. Spillover checkpoints: Zenodo DOI 10.5281/zenodo.21809791; software/code archive DOI assigned at the `reportcard-v0.1.1` GitHub/Zenodo release.

---

## Figure captions

**Figure 1. GenomeML Report Card and estimands.** (a) Pipeline from genomes+labels+label/block units to overlap audits, split contrast, and machine-readable reports; three distinct targets (sequence contamination, cross-fold group recurrence, estimand mismatch). (b) Layer A label-assignment groups vs Layer B deployment blocks; \(\theta_{\mathrm{random}}\) (≈ \(\theta_{\mathrm{seen}}\) under Layer A recurrence) vs \(\theta_{\mathrm{unseen}}(B)\); within-task contrast \(\Delta_B\).

**Figure 2. Label-geometry simulation.** Split-design contrast vs replication and ICC.

**Figure 3. Multi-task audit matrix (repeated CV).** Paired random vs group-blocked scores per task in separate panels with task-native metrics (Spearman ρ or AUROC). Within-task diagnostic only; metrics are not comparable across panels.

**Figure 4. Tiered reporting items for held-out-block claims.** Minimum, strongly recommended, and conditional/recommended fields.

**Supplementary Figure S_integrity.** SpillOver organism-constant labels [2]; contamination rebuild illustrates the risk of interpreting seen-group performance as evidence of held-out-group transfer [13].

**Supplementary Figure S_robustness.** Split-design contrasts with split-assignment intervals.

**Supplementary Figure S_T08_logo.** Viral-host–lineage (T08) leave-one-group-out AUROC by viral group.

**Supplementary Figure S_external.** OGT–species (T01) temporal species-disjoint external holdout.

## Table captions

**Table 1.** Primary multi-task audit matrix (repeated CV; task-primary metrics).

**Table 1a.** Label geometry diagnostics (Layer A/B, homogeneity, random-CV block recurrence).

**Table 2.** External deployment holdout (OGT–species temporal panel).

**Supplementary Table S_audit.** Audit demonstrations on public datasets linked to published genome-ML benchmarks (common k-mer Ridge probe; single-seed estimates for cross-study comparability only—not reproduction of original study pipelines or reported performance).

**Supplementary Table S_T06.** k-mer composition-cluster within vs between-cluster cosine similarity (T06).

**Supplementary Table S.** Construction controls; T07 sensitivity; model robustness; LOGO details; literature design audit; reproducibility hashes.

---

## References

[1] Nguyen E, Poli M, Durrant MG, Kang B, Katrekar D, Li DB, et al. Sequence modeling and design from molecular to genome scale with Evo. Science. 2024 Nov 15;386(6723):eado9336. doi:10.1126/science.ado9336

[2] Grange ZL, Goldstein T, Johnson CK, Anthony S, Gilardi K, Daszak P, et al. Ranking the risk of animal-to-human spillover for newly discovered viruses. Proc Natl Acad Sci USA. 2021 Apr 13;118(15):e2002324118. doi:10.1073/pnas.2002324118

[3] Mollentze N, Babayan SA, Streicker DG. Identifying and prioritizing potential human-infecting viruses from their genome sequences. PLoS Biol. 2021 Sep;19(9):e3001390. doi:10.1371/journal.pbio.3001390

[4] Babayan SA, Orton RJ, Streicker DG. Predicting reservoir hosts and arthropod vectors from evolutionary signatures in RNA virus genomes. Science. 2018 Nov 2;362(6414):577–80. doi:10.1126/science.aap9072

[5] Ji Y, Zhou Z, Liu H, Davuluri RV. DNABERT: pre-trained Bidirectional Encoder Representations from Transformers model for DNA-language in genome. Bioinformatics. 2021 Aug 1;37(15):2112–20. doi:10.1093/bioinformatics/btab083

[6] Dalla-Torre H, Gonzalez L, Mendoza-Revilla J, Lopez Carranza N, Grzywaczewski AH, Oteri F, et al. Nucleotide Transformer: building and evaluating robust foundation models for human genomics. Nat Methods. 2025 Feb;22(2):274–84. doi:10.1038/s41592-024-02523-z

[7] Fu L, Niu B, Zhu Z, Wu S, Li W. CD-HIT: accelerated for clustering the next-generation sequencing data. Bioinformatics. 2012 Dec 1;28(23):3150–2. doi:10.1093/bioinformatics/bts565

[8] Ondov BD, Treangen TJ, Melsted P, Mallonee AB, Bergman NH, Koren S, et al. Mash: fast genome and metagenome distance estimation using MinHash. Genome Biol. 2016 Jun 20;17:132. doi:10.1186/s13059-016-0997-x

[9] Felsenstein J. Phylogenies and the comparative method. Am Nat. 1985 Jan;125(1):1–15. doi:10.1086/284325

[10] Ives AR, Garland T Jr. Phylogenetic logistic regression for binary dependent variables. Syst Biol. 2010 Feb;59(1):9–26. doi:10.1093/sysbio/syp074

[11] Sundararajan M, Taly A, Yan Q. Axiomatic attribution for deep networks. In: Proceedings of the 34th International Conference on Machine Learning (ICML). Sydney, Australia: PMLR; 2017. p. 3319–28.

[12] Novakovsky G, Dexter N, Libbrecht MW, Wasserman WW, Mostafavi S. Obtaining genetics insights from deep learning via explainable artificial intelligence. Nat Rev Genet. 2023 Feb;24(2):125–37. doi:10.1038/s41576-022-00532-2

[13] Kaufman S, Rosset S, Perlich C, Stitelman O. Leakage in data mining: formulation, detection, and avoidance. ACM Trans Knowl Discov Data. 2012 Dec 18;6(4):15:1–15:21. doi:10.1145/2382577.2382579

[14] Engqvist MKM. Correlating enzyme annotations with a large set of microbial growth temperatures reveals metabolic adaptations to growth at diverse temperatures. BMC Bioinformatics. 2018;19(Suppl 1):32. doi:10.1186/s12859-018-2020-1

[15] Hurlbert SH. Pseudoreplication and the design of ecological field experiments. Ecol Monogr. 1984 Jun;54(2):187–211. doi:10.2307/1942661

---

## Conflict of interest

The authors declare that they have no conflict of interest.

## Funding

This work was supported in part by the Major Project of Guangzhou National Laboratory (Nos. SRPG22-007 and GZNL2025A0009), the National Key Research and Development Program of China (2025YFE0126600), the Startup Program of Guangzhou National Laboratory (YW-YFYJ0101), and the Science and Technology Development Fund of Macau SAR (0002/2024/RDP).

## Author contributions

Xindi Wang: conceptualization, methodology, coding, writing, and visualization. Junyu Luo: supervision and project administration. Yixue Li: supervision, project administration, and funding acquisition. Chitin Hon: supervision, project administration, and funding acquisition.

## Declaration of generative AI and AI-assisted technologies

During the preparation of this work the authors used AI-assisted tools for language editing, structural polishing, and drafting support; the authors reviewed and edited the content and take full responsibility for the published article.
