#!/usr/bin/env python3
"""Bootstrap CI for Human Virus? == No subset (n≈56). Supplement only."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

OUT = Path("/home/wangxindi/evo-main/paper/analysis_p0p1_circularity_baselines_wxd0804")
OVERLAP = OUT / "Table_P0_overlap_with_ablated_scores.csv"
SEED = 42
B = 2000
ALPHAS = np.logspace(-3, 3, 13)


def emb_cols(df):
    return [c for c in df.columns if c.startswith("emb_")]


def oof_pred(X, y, groups, n_splits=5):
    groups = np.asarray(groups)
    n_groups = pd.Series(groups).nunique()
    splits = min(n_splits, int(n_groups))
    if splits < 2:
        return np.full(len(y), np.nan), int(n_groups), splits
    cv = GroupKFold(n_splits=splits)
    pred = np.full(len(y), np.nan)
    for tr, te in cv.split(X, y, groups):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[te])
        model = RidgeCV(alphas=ALPHAS)
        model.fit(Xtr, y[tr])
        pred[te] = model.predict(Xte)
    return pred, int(n_groups), splits


def spearman_safe(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5:
        return np.nan
    r = spearmanr(a[m], b[m]).correlation
    return float(r) if np.isfinite(r) else np.nan


def bootstrap_organism(y, pred, groups, B=B, seed=SEED):
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    rhos = []
    for _ in range(B):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in samp])
        rhos.append(spearman_safe(y[idx], pred[idx]))
    rhos = np.asarray(rhos, dtype=float)
    rhos = rhos[np.isfinite(rhos)]
    return {
        "rho_point": spearman_safe(y, pred),
        "n_boot": int(len(rhos)),
        "ci95_low": float(np.percentile(rhos, 2.5)) if len(rhos) else np.nan,
        "ci95_high": float(np.percentile(rhos, 97.5)) if len(rhos) else np.nan,
        "boot_median": float(np.median(rhos)) if len(rhos) else np.nan,
    }


def main():
    m = pd.read_csv(OVERLAP)
    if "human_virus" not in m.columns:
        raise SystemExit(f"missing human_virus in {OVERLAP}; cols sample={list(m.columns)[:20]}")
    submask = m["human_virus"].fillna(0).to_numpy() == 0
    sub = m.loc[submask].copy()
    print(
        "eligible",
        int(submask.sum()),
        "n_org",
        sub["Organism_Name"].nunique(),
        "human_virus counts",
        m["human_virus"].value_counts(dropna=False).to_dict(),
        flush=True,
    )

    X = sub[emb_cols(sub)].to_numpy(dtype=float)
    groups = sub["Organism_Name"].astype(str).to_numpy()
    target_map = {
        "spillover_total": "spillover_total",
        "spillover_host_score": "spillover_host_score",
        "total_ablated_circular": (
            "total_ablated_circular"
            if "total_ablated_circular" in sub.columns
            else "total_ablated_circular_factors"
        ),
    }
    rows = []
    for name, col in target_map.items():
        y = sub[col].to_numpy(dtype=float)
        pred, n_groups, n_splits = oof_pred(X, y, groups)
        boot = bootstrap_organism(y, pred, groups)
        row = {
            "target": name,
            "cv": "GroupKFold_organism",
            "n_seq": int(len(sub)),
            "n_groups": n_groups,
            "n_splits": n_splits,
            **boot,
        }
        rows.append(row)
        print(row, flush=True)

    pd.DataFrame(rows).to_csv(OUT / "Table_P0_nonHuman_bootstrap_CI.csv", index=False)
    (OUT / "run_meta_nonHuman_bootstrap_wxd0804.json").write_text(
        json.dumps({"filter": "human_virus==0", "B": B, "rows": rows}, indent=2)
    )
    print("DONE", OUT / "Table_P0_nonHuman_bootstrap_CI.csv", flush=True)


if __name__ == "__main__":
    main()
