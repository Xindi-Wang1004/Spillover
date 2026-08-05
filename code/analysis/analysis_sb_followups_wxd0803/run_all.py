#!/usr/bin/env python3
"""SB analyses 1 / 2 / 3 / 5 on 10.40.1.16 (wxd0803).

Outputs under:
  /home/wangxindi/evo-main/paper/analysis_sb_followups_wxd0803/

Run:
  bash -lc 'source .../conda.sh && conda activate evo_design && \
    python /home/wangxindi/evo-main/paper/analysis_sb_followups_wxd0803/run_all.py'
"""
from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler

OUT = Path("/home/wangxindi/evo-main/paper/analysis_sb_followups_wxd0803")
EVO_DATA = Path("/home/wangxindi/evo/evo_data")
DENSE = EVO_DATA / "ig_sliding_dense" / "genome_tracks"
GB_CACHE = EVO_DATA / "genbank_cache_cov_subset"
FASTA = EVO_DATA / "ig_analysis_cov_subset.fasta"
EMB = Path(
    "/home/wangxindi/evo-main/analysis_representation_pca_umap/results/"
    "embeddings_regression_20260514_202026.csv"
)
OVERLAP = EVO_DATA / "processed_test_spillover_overlap_max100_per_org.csv"
MF_ROOT = EVO_DATA / "ig_multi_family_wxd0728"
SLIDING_COV = EVO_DATA / "ig_sliding_dense"

SEED = 42
N_PERM_SHIFT = 2000
N_PERM_META = 500
N_PERM_TIP = 400


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_fasta(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    hid = None
    buf: list[str] = []

    def flush():
        nonlocal hid, buf
        if hid is not None:
            out[hid] = "".join(buf).upper()
        hid, buf = None, []

    def acc_from_header(h: str) -> str:
        h = h[1:].strip()
        parts = [x.strip() for x in h.split("|") if x.strip()]
        for tok in parts:
            t = tok.split()[0]
            if re.search(r"\d", t) and len(t) <= 20 and not t.lower().endswith("nt"):
                return t
        return h.split()[0]

    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                hid = acc_from_header(line)
                buf = []
            else:
                buf.append(line)
        flush()
    return out


def load_attr_track(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    if "attr_abs_mean" in df.columns:
        pos = df["genomic_pos"].to_numpy(dtype=int)
        val = df["attr_abs_mean"].to_numpy(dtype=float)
    else:
        pos = df["genomic_pos"].to_numpy(dtype=int)
        val = np.abs(df["attr_score"].to_numpy(dtype=float))
    gmax = int(pos.max())
    vec = np.zeros(gmax + 1, dtype=float)
    vec[pos] = val
    return vec


def mass_enrichment(attr: np.ndarray, mask: np.ndarray) -> float:
    mass = np.abs(attr)
    total = float(mass.sum()) + 1e-12
    mass_frac = float(mass[mask].sum()) / total
    len_frac = float(mask.mean()) + 1e-12
    return mass_frac / len_frac


def circular_shift_null(attr: np.ndarray, mask: np.ndarray, n_perm: int, seed: int):
    rng = np.random.default_rng(seed)
    obs = mass_enrichment(attr, mask)
    null = np.empty(n_perm, dtype=float)
    n = len(attr)
    for i in range(n_perm):
        shift = int(rng.integers(0, n))
        null[i] = mass_enrichment(np.roll(attr, shift), mask)
    p = (1.0 + float(np.sum(null >= obs))) / (n_perm + 1.0)
    return obs, p, null


def parse_orf1ab_span(gb_text: str) -> tuple[int, int] | None:
    """Return 0-based half-open ORF1ab/replicase CDS span (longest)."""
    # crude GenBank CDS parse
    spans = []
    lines = gb_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if re.match(r"^\s{5}CDS\s+", line):
            loc = line.strip().split(None, 1)[-1]
            i += 1
            gene = product = ""
            while i < len(lines) and lines[i].startswith("                     "):
                t = lines[i].strip()
                if t.startswith("/gene="):
                    gene = t.split("=", 1)[1].strip().strip('"')
                if t.startswith("/product="):
                    product = t.split("=", 1)[1].strip().strip('"')
                i += 1
            blob = f"{gene} {product}".lower()
            if ("orf1ab" in blob) or ("replicase" in blob) or ("rna-dependent rna polymerase" in blob) or (
                "polyprotein" in blob and "orf1" in blob
            ):
                m = re.findall(r"(\d+)\.\.(\d+)", loc.replace("<", "").replace(">", ""))
                if m:
                    a = min(int(x[0]) for x in m) - 1
                    b = max(int(x[1]) for x in m)
                    spans.append((a, b, b - a))
            continue
        i += 1
    if not spans:
        # fallback: join all CDS with orf1a/orf1ab in gene
        return None
    spans.sort(key=lambda x: -x[2])
    return spans[0][0], spans[0][1]


def gc_frac(seq: str) -> float:
    s = re.sub(r"[^ACGT]", "", seq.upper())
    if not s:
        return float("nan")
    return (s.count("G") + s.count("C")) / len(s)


def dinuc_frac(seq: str, dinuc: str) -> float:
    s = re.sub(r"[^ACGT]", "", seq.upper())
    if len(s) < 2:
        return float("nan")
    d = dinuc.upper()
    hits = sum(1 for i in range(len(s) - 1) if s[i : i + 2] == d)
    return hits / (len(s) - 1)


def oof_ridge_spearman(X: np.ndarray, y: np.ndarray, groups: np.ndarray | None, seed: int = 42) -> float:
    """Nested OOF RidgeCV with fold-wise scaling; return Spearman(y, oof)."""
    n = len(y)
    oof = np.full(n, np.nan)
    if groups is None:
        splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
        splits = splitter.split(X)
    else:
        # GroupKFold needs enough groups
        n_groups = len(np.unique(groups))
        n_splits = min(5, n_groups) if n_groups >= 2 else 2
        if n_groups < 2:
            splitter = KFold(n_splits=min(5, n), shuffle=True, random_state=seed)
            splits = splitter.split(X)
        else:
            splitter = GroupKFold(n_splits=n_splits)
            splits = splitter.split(X, y, groups)
    alphas = np.logspace(-3, 3, 13)
    for tr, te in splits:
        if len(tr) < 5 or len(te) < 1:
            continue
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[te])
        model = RidgeCV(alphas=alphas)
        model.fit(Xtr, y[tr])
        oof[te] = model.predict(Xte)
    mask = np.isfinite(oof)
    if mask.sum() < 5:
        return float("nan")
    return float(stats.spearmanr(y[mask], oof[mask]).correlation)


# ───────────────────────── Analysis 3 ─────────────────────────


def run_analysis3() -> dict:
    print("=== Analysis 3: known-locus Stouffer + locus-set permutation ===", flush=True)
    # Use published Table32 coordinates for ORF1ab_C_terminal_third (dense_stitched only)
    t32 = pd.read_csv(
        "/home/wangxindi/evo-main/paper/bib_tables/Table32_known_loci_attribution_scores.csv"
    )
    t32 = t32[
        (t32["locus_id"] == "ORF1ab_C_terminal_third")
        & (t32["attr_source"] == "dense_stitched")
        & (t32["ok"] == True)
    ].copy()
    print(f"  using Table32 dense rows n={len(t32)}", flush=True)

    rows = []
    null_store = {}
    resolved = {}

    for _, r in t32.iterrows():
        acc = str(r["accession"])
        tp = DENSE / f"{acc}_stitched_attr.csv"
        if not tp.exists():
            print(f"[skip] no track {acc}", flush=True)
            continue
        attr = load_attr_track(tp)
        n = len(attr)
        c0, c1 = int(r["locus_start"]), int(r["locus_end"])
        c0, c1 = max(0, c0), min(n, max(c0 + 1, c1))
        mask = np.zeros(n, dtype=bool)
        mask[c0:c1] = True
        obs, p, null = circular_shift_null(attr, mask, N_PERM_SHIFT, SEED + (hash(acc) % 10000))
        mu, sd = float(np.mean(null)), float(np.std(null) + 1e-12)
        z = (obs - mu) / sd
        rows.append(
            {
                "accession": acc,
                "locus_id": "ORF1ab_C_terminal_third",
                "locus_start": c0,
                "locus_end": c1,
                "locus_len": c1 - c0,
                "genome_len": n,
                "mass_enrichment": obs,
                "mass_enrichment_table32": float(r["mass_enrichment"]),
                "p_circular_shift_mass": p,
                "p_table32": float(r["p_circular_shift_mass"]),
                "null_mean": mu,
                "null_sd": sd,
                "z": z,
            }
        )
        null_store[acc] = null.astype(np.float32)
        resolved[acc] = {"attr": attr, "c0": c0, "c1": c1, "n": n, "L": c1 - c0, "mu": mu, "sd": sd}
        print(f"  {acc}: enr={obs:.3f} (t32={float(r['mass_enrichment']):.3f}) p={p:.4g} z={z:.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "Table_S34_orf1ab_known_locus_with_nullstats.csv", index=False)
    np.savez_compressed(OUT / "orf1ab_circular_shift_nulls.npz", **null_store)

    zs = df["z"].to_numpy(dtype=float)
    T_obs = float(np.sum(zs) / math.sqrt(len(zs)))
    p_asym = float(stats.norm.sf(T_obs))

    # Locus-set permutation with inner circular-shift B=400 for speed
    rng = np.random.default_rng(SEED + 7)
    T_null = np.empty(N_PERM_META, dtype=float)
    accessions = list(resolved.keys())
    INNER = 400
    for b in range(N_PERM_META):
        zstar = []
        for acc in accessions:
            info = resolved[acc]
            attr, n, L = info["attr"], info["n"], info["L"]
            if n <= L + 1:
                continue
            start = int(rng.integers(0, n - L))
            mask = np.zeros(n, dtype=bool)
            mask[start : start + L] = True
            obs_r, _, null_r = circular_shift_null(attr, mask, INNER, int(rng.integers(0, 1_000_000)))
            mu_r, sd_r = float(np.mean(null_r)), float(np.std(null_r) + 1e-12)
            zstar.append((obs_r - mu_r) / sd_r)
        T_null[b] = float(np.sum(zstar) / math.sqrt(len(zstar))) if zstar else np.nan
        if (b + 1) % 50 == 0:
            print(f"  meta perm {b+1}/{N_PERM_META}", flush=True)

    T_null = T_null[np.isfinite(T_null)]
    p_meta = (1.0 + float(np.sum(T_null >= T_obs))) / (len(T_null) + 1.0)
    summary = {
        "n_accessions": int(len(df)),
        "median_mass_enrichment": float(df["mass_enrichment"].median()),
        "n_sig_p005": int((df["p_circular_shift_mass"] < 0.05).sum()),
        "T_obs": T_obs,
        "p_asym_one_sided": p_asym,
        "p_meta_locus_set": p_meta,
        "T_null_mean": float(np.mean(T_null)),
        "T_null_p95": float(np.quantile(T_null, 0.95)),
        "n_perm_shift": N_PERM_SHIFT,
        "n_perm_meta": N_PERM_META,
        "n_perm_shift_meta_inner": INNER,
        "coord_source": "Table32_known_loci_attribution_scores.csv dense_stitched",
    }
    pd.DataFrame([summary]).to_csv(OUT / "Table_S34_orf1ab_stouffer_locusset_meta.csv", index=False)
    np.save(OUT / "orf1ab_T_meta_null.npy", T_null)
    print("Analysis3 summary:", summary, flush=True)
    return summary


# ───────────────────────── Analysis 1 ─────────────────────────


def run_analysis1() -> dict:
    print("=== Analysis 1: phylogenetic / relatedness controls ===", flush=True)
    emb = pd.read_csv(EMB)
    ov = pd.read_csv(OVERLAP, usecols=lambda c: c in {"Accession", "Family", "Host", "Organism_Name", "Sequence"})
    df = emb.merge(ov[["Accession", "Family"]].drop_duplicates(), on="Accession", how="left")
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    X = df[emb_cols].to_numpy(dtype=float)
    y = df["spillover_total"].to_numpy(dtype=float)
    fam = df["Family"].fillna("NA").to_numpy()
    org = df["Organism_Name"].fillna(df["Accession"]).to_numpy()
    host = df["Host_label"].to_numpy(dtype=float)

    rho_obs = oof_ridge_spearman(X, y, groups=org, seed=SEED)
    print(f"  rho_obs (group OOF)={rho_obs:.4f}", flush=True)

    # Tip-label permutation within family (shuffle y inside each family)
    rng = np.random.default_rng(SEED)
    null = np.empty(N_PERM_TIP, dtype=float)
    for i in range(N_PERM_TIP):
        y_perm = y.copy()
        for f in np.unique(fam):
            idx = np.where(fam == f)[0]
            if len(idx) > 1:
                y_perm[idx] = rng.permutation(y_perm[idx])
        null[i] = oof_ridge_spearman(X, y_perm, groups=org, seed=SEED + i + 1)
        if (i + 1) % 100 == 0:
            print(f"  tip-perm {i+1}/{N_PERM_TIP}", flush=True)
    p_tip = (1.0 + float(np.sum(np.abs(null) >= abs(rho_obs)))) / (N_PERM_TIP + 1.0)

    # Embedding-PCA + family fixed-effect residualization as phylogeny proxy
    # (full IQ-TREE PGLS deferred if trees unavailable; report as relatedness control)
    pca = PCA(n_components=min(20, X.shape[1], X.shape[0] - 1), random_state=SEED)
    Pcs = pca.fit_transform(StandardScaler().fit_transform(X))
    # residualize y on family dummies
    fam_u = pd.get_dummies(pd.Series(fam), drop_first=True).to_numpy(dtype=float)
    if fam_u.size == 0:
        y_res = y - y.mean()
        X_res = Pcs
    else:
        # OLS residuals
        beta_y, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(y)), fam_u]), y, rcond=None)
        y_res = y - np.column_stack([np.ones(len(y)), fam_u]) @ beta_y
        X_aug = np.column_stack([np.ones(len(y)), fam_u, Pcs])
        beta, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
        # partial F for PC block
        yhat_full = X_aug @ beta
        ss_full = float(np.sum((y - yhat_full) ** 2))
        X_red = np.column_stack([np.ones(len(y)), fam_u])
        beta_r, *_ = np.linalg.lstsq(X_red, y, rcond=None)
        yhat_red = X_red @ beta_r
        ss_red = float(np.sum((y - yhat_red) ** 2))
        df1 = Pcs.shape[1]
        df2 = max(1, len(y) - X_aug.shape[1])
        f_stat = ((ss_red - ss_full) / df1) / (ss_full / df2 + 1e-12)
        p_pc = float(stats.f.sf(f_stat, df1, df2))

    rho_fam_resid = oof_ridge_spearman(Pcs, y_res, groups=org, seed=SEED)

    # Within-family tip permutation for Coronaviridae only (if enough)
    cov_idx = np.where(fam == "Coronaviridae")[0]
    cov_summary = {}
    if len(cov_idx) >= 8:
        Xc, yc, orgc = X[cov_idx], y[cov_idx], org[cov_idx]
        rho_cov = oof_ridge_spearman(Xc, yc, groups=orgc, seed=SEED)
        null_c = np.empty(min(500, N_PERM_TIP), dtype=float)
        for i in range(len(null_c)):
            null_c[i] = oof_ridge_spearman(Xc, rng.permutation(yc), groups=orgc, seed=SEED + i)
        p_cov = (1.0 + float(np.sum(np.abs(null_c) >= abs(rho_cov)))) / (len(null_c) + 1.0)
        cov_summary = {"n": int(len(cov_idx)), "rho_obs": float(rho_cov), "p_tip_perm": float(p_cov)}

    summary = {
        "rho_obs_group_oof": float(rho_obs),
        "tip_perm_within_family_p": float(p_tip),
        "tip_perm_null_mean": float(np.nanmean(null)),
        "tip_perm_null_p95": float(np.nanquantile(np.abs(null), 0.95)),
        "rho_after_family_residualize_pca20": float(rho_fam_resid),
        "pca_block_partial_F_p": float(p_pc) if fam_u.size else float("nan"),
        "coronaviridae_within": cov_summary,
        "note": "Tip permutations shuffle SpillOver within family; PCA+family residualization is a relatedness proxy pending formal PGLS trees.",
    }
    pd.DataFrame([summary]).to_csv(OUT / "Table_phylo_relatedness_controls_summary.csv", index=False)
    pd.DataFrame({"rho_perm": null}).to_csv(OUT / "Table_phylo_tip_perm_null.csv", index=False)
    print("Analysis1 summary:", summary, flush=True)
    return summary


# ───────────────────────── Analysis 2 ─────────────────────────


def _window_rows_from_sliding_dense() -> pd.DataFrame:
    """Build high/low window table from dense sliding IG attr files if present."""
    rows = []
    # Prefer multi-family panels with top windows; fallback: CoV dense w0-w2 style files
    # Use stitched track + three equal genome thirds as windows proxy if needed
    fasta = load_fasta(FASTA) if FASTA.exists() else {}
    for tp in sorted(DENSE.glob("*_stitched_attr.csv")):
        acc = tp.name.replace("_stitched_attr.csv", "")
        attr = load_attr_track(tp)
        n = len(attr)
        seq = fasta.get(acc, "")
        # three non-overlapping windows
        cuts = [0, n // 3, 2 * n // 3, n]
        scores = []
        for i in range(3):
            a, b = cuts[i], cuts[i + 1]
            scores.append((i, a, b, float(np.mean(np.abs(attr[a:b])))))
        scores.sort(key=lambda x: -x[3])
        for rank, (i, a, b, sc) in enumerate(scores):
            subseq = seq[a:b] if seq else ""
            rows.append(
                {
                    "accession": acc,
                    "family": "Coronaviridae",
                    "window_idx": i,
                    "start": a,
                    "end": b,
                    "attr_mean": sc,
                    "is_high": rank == 0,
                    "gc": gc_frac(subseq) if subseq else np.nan,
                    "cpg": dinuc_frac(subseq, "CG") if subseq else np.nan,
                    "cpa": dinuc_frac(subseq, "CA") if subseq else np.nan,
                    "window_len": b - a,
                }
            )
    return pd.DataFrame(rows)


def run_analysis2() -> dict:
    print("=== Analysis 2: window-level composition controls ===", flush=True)
    win = _window_rows_from_sliding_dense()
    win.to_csv(OUT / "Table_S33_window_composition_features.csv", index=False)

    # paired high-low deltas per genome
    deltas = []
    for acc, g in win.groupby("accession"):
        hi = g[g["is_high"]]
        lo = g[~g["is_high"]]
        if hi.empty or lo.empty:
            continue
        for feat in ["gc", "cpg", "cpa"]:
            deltas.append(
                {
                    "accession": acc,
                    "feature": feat,
                    "high": float(hi[feat].mean()),
                    "low": float(lo[feat].mean()),
                    "delta": float(hi[feat].mean() - lo[feat].mean()),
                }
            )
    ddf = pd.DataFrame(deltas)
    ddf.to_csv(OUT / "Table_S33_window_composition_high_vs_low_deltas.csv", index=False)

    wilcox = []
    for feat, g in ddf.groupby("feature"):
        d = g["delta"].to_numpy(dtype=float)
        d = d[np.isfinite(d)]
        if len(d) < 5:
            continue
        stat_w, p = stats.wilcoxon(d)
        wilcox.append({"feature": feat, "n": len(d), "median_delta": float(np.median(d)), "wilcoxon_p": float(p)})
    wdf = pd.DataFrame(wilcox)
    wdf.to_csv(OUT / "Table_S33_window_composition_wilcoxon.csv", index=False)

    # composition-adjusted logistic for high window ~ polymerase membership proxy:
    # use GC/CpG/len as covariates predicting is_high
    X = win[["gc", "cpg", "cpa", "window_len"]].to_numpy(dtype=float)
    y = win["is_high"].astype(int).to_numpy()
    mask = np.isfinite(X).all(axis=1)
    X, y = X[mask], y[mask]
    clf = LogisticRegression(max_iter=2000)
    clf.fit(StandardScaler().fit_transform(X), y)
    # leave-one-accession AUC
    accs = win.loc[mask, "accession"].to_numpy() if hasattr(win, "loc") else win["accession"].to_numpy()[mask]
    # rebuild mask-aligned accessions
    accs = win.iloc[np.where(np.isfinite(win[["gc", "cpg", "cpa", "window_len"]].to_numpy(dtype=float)).all(axis=1))[0]][
        "accession"
    ].to_numpy()
    oof = np.full(len(y), np.nan)
    for a in np.unique(accs):
        te = accs == a
        tr = ~te
        if tr.sum() < 5 or te.sum() < 1:
            continue
        sc = StandardScaler()
        model = LogisticRegression(max_iter=2000)
        model.fit(sc.fit_transform(X[tr]), y[tr])
        oof[te] = model.predict_proba(sc.transform(X[te]))[:, 1]
    m = np.isfinite(oof)
    auc = float(roc_auc_score(y[m], oof[m])) if m.sum() > 5 and len(np.unique(y[m])) > 1 else float("nan")

    # ORF1ab enrichment adjusted: among CoV windows, polymerase class ~ high, with composition covariates
    # Approximate polymerase membership by genomic position overlap with ORF1ab C-terminal third from analysis3 table if present
    poly_rows = []
    t3 = OUT / "Table_S34_orf1ab_known_locus_with_nullstats.csv"
    loci = pd.read_csv(t3) if t3.exists() else pd.DataFrame()
    for _, r in win.iterrows():
        is_pol = 0
        if len(loci):
            hit = loci[loci["accession"] == r["accession"]]
            if len(hit):
                c0, c1 = int(hit.iloc[0]["locus_start"]), int(hit.iloc[0]["locus_end"])
                # overlap fraction
                a, b = int(r["start"]), int(r["end"])
                ovl = max(0, min(b, c1) - max(a, c0))
                is_pol = int(ovl / max(1, b - a) >= 0.5)
        poly_rows.append(is_pol)
    win = win.copy()
    win["is_polymerase_like"] = poly_rows

    # Fisher crude OR high vs pol
    tab = pd.crosstab(win["is_high"], win["is_polymerase_like"])
    if tab.shape == (2, 2):
        oddsratio, p_fish = stats.fisher_exact(tab)
    else:
        oddsratio, p_fish = float("nan"), float("nan")

    # logistic: is_high ~ is_pol + gc + cpg + cpa + len
    cols = ["is_polymerase_like", "gc", "cpg", "cpa", "window_len"]
    XX = win[cols].to_numpy(dtype=float)
    yy = win["is_high"].astype(int).to_numpy()
    mm = np.isfinite(XX).all(axis=1)
    XX, yy = XX[mm], yy[mm]
    sc = StandardScaler()
    XZs = sc.fit_transform(XX)
    # unadjusted pol-only
    lr0 = LogisticRegression(max_iter=2000)
    lr0.fit(XZs[:, :1], yy)
    # adjusted
    lr1 = LogisticRegression(max_iter=2000)
    lr1.fit(XZs, yy)
    # coefficient for pol as log-OR approx
    summary = {
        "wilcoxon": wilcox,
        "composition_predicts_high_window_oof_auc": auc,
        "fisher_high_vs_polymerase_like_OR": float(oddsratio),
        "fisher_p": float(p_fish),
        "logit_pol_coef_unadj": float(lr0.coef_[0, 0]),
        "logit_pol_coef_adj_comp": float(lr1.coef_[0, 0]),
        "logit_OR_unadj": float(math.exp(lr0.coef_[0, 0])),
        "logit_OR_adj_comp": float(math.exp(lr1.coef_[0, 0])),
        "n_windows": int(len(win)),
        "note": "Windows = genome thirds ranked by mean |IG|; polymerase-like = ≥50% overlap with ORF1ab C-terminal third.",
    }
    pd.DataFrame([summary]).to_csv(OUT / "Table_S33_composition_control_summary.csv", index=False)
    print("Analysis2 summary:", summary, flush=True)
    return summary


# ───────────────────────── Analysis 5 ─────────────────────────


def nei_gojobori_pairwise(seq1: str, seq2: str) -> tuple[float, float, float]:
    """Return (dN, dS, dN/dS) for aligned codon sequences (same length, multiple of 3)."""
    s1 = re.sub(r"[^ACGT-]", "", seq1.upper())
    s2 = re.sub(r"[^ACGT-]", "", seq2.upper())
    L = min(len(s1), len(s2))
    L = L - (L % 3)
    if L < 9:
        return float("nan"), float("nan"), float("nan")
    # simplified: count nonsyn/syn differences / sites using standard codon table
    codon_table = {
        "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
        "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
        "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
        "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
        "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
        "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
        "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
        "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
        "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
        "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
        "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
        "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
        "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
        "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
        "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
        "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
    }

    def aa(c):
        return codon_table.get(c, "X")

    nd = sd = ns = ss = 0.0
    for i in range(0, L, 3):
        c1, c2 = s1[i : i + 3], s2[i : i + 3]
        if "-" in c1 or "-" in c2 or "N" in c1 or "N" in c2:
            continue
        if c1 not in codon_table or c2 not in codon_table:
            continue
        # approximate syn/nonsyn potential: 1 syn site + 2 nonsyn per codon (very rough)
        ns += 2.25
        ss += 0.75
        if c1 == c2:
            continue
        diffs = sum(a != b for a, b in zip(c1, c2))
        if aa(c1) == aa(c2):
            sd += diffs
        else:
            nd += diffs
    dN = nd / max(ns, 1e-9)
    dS = sd / max(ss, 1e-9)
    return dN, dS, dN / max(dS, 1e-9)


def run_analysis5() -> dict:
    print("=== Analysis 5: variant-level dN/dS / fixed diffs (CoV) ===", flush=True)
    fasta = load_fasta(FASTA)
    emb = pd.read_csv(EMB, usecols=["Accession", "Host_label"])
    host_map = dict(zip(emb["Accession"], emb["Host_label"]))

    # Use genome thirds as fragments; compare high vs low attr thirds across host classes
    win = _window_rows_from_sliding_dense()
    if win.empty:
        return {"error": "no windows"}

    # Build per-accession high and low sequences
    recs = []
    for acc, g in win.groupby("accession"):
        seq = fasta.get(acc)
        if not seq:
            continue
        hi = g[g["is_high"]].iloc[0]
        lo = g[~g["is_high"]].sort_values("attr_mean").iloc[0]
        recs.append(
            {
                "accession": acc,
                "host": float(host_map.get(acc, np.nan)),
                "high_seq": seq[int(hi["start"]) : int(hi["end"])],
                "low_seq": seq[int(lo["start"]) : int(lo["end"])],
            }
        )
    human = [r for r in recs if r["host"] == 1]
    nonhuman = [r for r in recs if r["host"] == 0]
    print(f"  host1={len(human)} host0={len(nonhuman)}", flush=True)

    pair_rows = []
    # pairwise human vs nonhuman on high and low fragments (truncate to min len multiple of 3)
    for h in human:
        for n in nonhuman:
            for kind in ["high_seq", "low_seq"]:
                s1, s2 = h[kind], n[kind]
                L = min(len(s1), len(s2))
                L = L - (L % 3)
                s1, s2 = s1[:L], s2[:L]
                dN, dS, w = nei_gojobori_pairwise(s1, s2)
                # fixed AA diffs
                fixed = 0
                codon_table = {
                    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L", "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
                    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*", "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
                    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L", "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
                    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q", "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
                    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M", "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
                    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K", "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
                    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V", "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
                    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E", "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
                }
                for i in range(0, L, 3):
                    c1, c2 = s1[i : i + 3], s2[i : i + 3]
                    if c1 in codon_table and c2 in codon_table and codon_table[c1] != codon_table[c2]:
                        fixed += 1
                pair_rows.append(
                    {
                        "human_acc": h["accession"],
                        "nonhuman_acc": n["accession"],
                        "fragment": "high" if kind == "high_seq" else "low",
                        "dN": dN,
                        "dS": dS,
                        "dNdS": w,
                        "aa_diffs": fixed,
                        "aln_len": L,
                    }
                )
    pdf = pd.DataFrame(pair_rows)
    pdf.to_csv(OUT / "Table_S36_cov_pairwise_dnds_high_vs_low.csv", index=False)

    summary_rows = []
    for frag, g in pdf.groupby("fragment"):
        summary_rows.append(
            {
                "fragment": frag,
                "n_pairs": len(g),
                "median_dNdS": float(g["dNdS"].median()),
                "median_aa_diffs": float(g["aa_diffs"].median()),
                "mean_dNdS": float(g["dNdS"].mean()),
            }
        )
    sdf = pd.DataFrame(summary_rows)
    # paired contrast: for each host pair, high-low dNdS
    piv = pdf.pivot_table(index=["human_acc", "nonhuman_acc"], columns="fragment", values="dNdS")
    if {"high", "low"}.issubset(set(piv.columns)):
        delta = (piv["high"] - piv["low"]).dropna()
        if len(delta) >= 5:
            stat_w, p = stats.wilcoxon(delta.to_numpy())
            contrast = {
                "n_pairs": int(len(delta)),
                "median_delta_dNdS_high_minus_low": float(delta.median()),
                "wilcoxon_p": float(p),
            }
        else:
            contrast = {"n_pairs": int(len(delta)), "wilcoxon_p": float("nan")}
    else:
        contrast = {}
    out = {"by_fragment": summary_rows, "high_vs_low_contrast": contrast, "note": "Simplified Nei-Gojobori on unaligned length-matched thirds; interpret cautiously."}
    pd.DataFrame([contrast]).to_csv(OUT / "Table_S37_cov_dnds_high_vs_low_contrast.csv", index=False)
    sdf.to_csv(OUT / "Table_S36_cov_dnds_summary_by_fragment.csv", index=False)
    print("Analysis5 summary:", out, flush=True)
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"started": now(), "seed": SEED}
    # Order: 3 first (formal closure), then 2 (uses 3 table), then 1, then 5
    meta["analysis3"] = run_analysis3()
    meta["analysis2"] = run_analysis2()
    meta["analysis1"] = run_analysis1()
    meta["analysis5"] = run_analysis5()
    meta["finished"] = now()
    (OUT / "run_meta_wxd0803.json").write_text(json.dumps(meta, indent=2, default=str))
    print("DONE", OUT, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
