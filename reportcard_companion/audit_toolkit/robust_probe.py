#!/usr/bin/env python3
"""Group-aware nested probes + repeated split-design contrasts.

Does NOT replace legacy oof_ridge_spearman (bit-compatible historical contract).
Use this module for GB statistical-robustness analyses.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler

SEED = 42
ALPHAS = np.logspace(-3, 3, 7)  # compact grid for nested search


def _is_binary(y: np.ndarray) -> bool:
    u = np.unique(y[np.isfinite(y)])
    return set(u.tolist()).issubset({0.0, 1.0}) and len(u) == 2


def _pick_alpha_group_aware(Xtr, ytr, gtr, alphas=ALPHAS, n_inner=3, seed=SEED):
    """Nested GroupKFold (or KFold if no groups) MSE minimization on train fold only."""
    gtr = None if gtr is None else np.asarray(gtr)
    best_a, best_mse = alphas[len(alphas) // 2], np.inf
    if gtr is not None:
        n_g = pd.Series(gtr).nunique()
        k = min(n_inner, n_g)
        if k < 2:
            return float(best_a)
        splitter = GroupKFold(n_splits=k).split(Xtr, ytr, gtr)
    else:
        splitter = KFold(n_splits=min(n_inner, len(ytr)), shuffle=True, random_state=seed).split(Xtr, ytr)
    folds = list(splitter)
    for a in alphas:
        mses = []
        for tr, te in folds:
            if len(tr) < 5 or len(te) < 1:
                continue
            sc = StandardScaler()
            model = Ridge(alpha=float(a))
            model.fit(sc.fit_transform(Xtr[tr]), ytr[tr])
            pred = model.predict(sc.transform(Xtr[te]))
            mses.append(float(np.mean((pred - ytr[te]) ** 2)))
        if mses and np.mean(mses) < best_mse:
            best_mse = float(np.mean(mses))
            best_a = float(a)
    return float(best_a)


def lock_alpha_once(X, y, groups, *, alphas=ALPHAS, n_splits=5, seed=SEED) -> float:
    """Lock Ridge α once via group-aware GroupKFold MSE on the full cohort.

    Preferred for GB robustness suite (fast, avoids nested retuning each repeat).
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    groups = np.asarray(groups)
    mask = np.isfinite(y) & np.isfinite(X).all(1)
    return _pick_alpha_group_aware(X[mask], y[mask], groups[mask], alphas=alphas, n_inner=n_splits, seed=seed)


def _fit_predict_fold(X, y, tr, te, groups=None, seed=SEED, alpha: Optional[float] = None):
    if alpha is None:
        gtr = None if groups is None else groups[tr]
        alpha = _pick_alpha_group_aware(X[tr], y[tr], gtr, seed=seed)
    sc = StandardScaler()
    model = Ridge(alpha=float(alpha))
    model.fit(sc.fit_transform(X[tr]), y[tr])
    return model.predict(sc.transform(X[te])), float(alpha)


def _score_genome(y, pred) -> dict[str, float]:
    ok = np.isfinite(y) & np.isfinite(pred)
    out = {"n": float(ok.sum()), "rho": float("nan"), "auc": float("nan")}
    if ok.sum() < 10:
        return out
    out["rho"] = float(spearmanr(y[ok], pred[ok])[0])
    if _is_binary(y[ok]) and len(np.unique(y[ok])) == 2:
        out["auc"] = float(roc_auc_score(y[ok], pred[ok]))
    return out


def _score_group_macro(y, pred, groups) -> dict[str, float]:
    """Group-aware secondary metrics.

    - Spearman rho: equal weight per group (mean label vs mean prediction).
    - AUROC: inverse-group-size weighted genome-pooled AUROC (group-balanced),
      not majority-vote group-mean labels (avoids ambiguous pseudo-binary labels
      when blocks are label-heterogeneous).
    """
    df = pd.DataFrame({"y": y, "pred": pred, "g": groups})
    df = df[np.isfinite(df.y) & np.isfinite(df.pred)]
    out = {"n_groups": float(df.g.nunique()), "rho": float("nan"), "auc": float("nan")}
    if len(df) < 10 or df.g.nunique() < 3:
        return out
    agg = df.groupby("g").agg(y=("y", "mean"), pred=("pred", "mean"))
    out["rho"] = float(spearmanr(agg.y, agg.pred)[0])
    if _is_binary(df.y.to_numpy()) and len(np.unique(df.y)) == 2:
        sizes = df.groupby("g")["y"].transform("size").to_numpy()
        weights = 1.0 / sizes
        out["auc"] = float(roc_auc_score(df.y.to_numpy(), df.pred.to_numpy(), sample_weight=weights))
    return out


def oof_once(
    X,
    y,
    groups,
    *,
    blocked: bool,
    n_splits=5,
    seed=SEED,
    group_fold_seed=None,
    alpha: Optional[float] = None,
):
    """One OOF pass; if blocked, optionally reshuffle group→fold via group_fold_seed.

    If ``alpha`` is set, reuse that Ridge α (GB suite protocol). Else nest α on each outer train.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    mask = np.isfinite(y) & np.isfinite(X).all(1)
    X, y = X[mask], y[mask]
    groups_arr = None if groups is None else np.asarray(groups)[mask]
    n = len(y)
    pred = np.full(n, np.nan)
    alphas = []

    if blocked:
        assert groups_arr is not None
        uniq = np.array(sorted(pd.Series(groups_arr).unique().tolist()), dtype=object)
        k = min(n_splits, len(uniq))
        if k < 2:
            return {"error": "too_few_groups"}
        rng = np.random.default_rng(SEED if group_fold_seed is None else group_fold_seed)
        order = rng.permutation(len(uniq))
        fold_of = {uniq[idx]: int(j % k) for j, idx in enumerate(order)}
        folds = np.array([fold_of[g] for g in groups_arr])
        for f in range(k):
            te = np.where(folds == f)[0]
            tr = np.where(folds != f)[0]
            if len(tr) < 5 or len(te) < 1:
                continue
            p, a = _fit_predict_fold(X, y, tr, te, groups=groups_arr, seed=seed + f, alpha=alpha)
            pred[te] = p
            alphas.append(a)
        scheme = f"repeated_GroupFold_k{k}"
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for f, (tr, te) in enumerate(cv.split(X, y)):
            p, a = _fit_predict_fold(X, y, tr, te, groups=None, seed=seed + f, alpha=alpha)
            pred[te] = p
            alphas.append(a)
        scheme = f"shuffle_KFold_k{n_splits}"

    genome = _score_genome(y, pred)
    gmacro = _score_group_macro(y, pred, groups_arr) if groups_arr is not None else {}
    primary = "auc" if _is_binary(y) and np.isfinite(genome.get("auc", np.nan)) else "rho"
    return {
        "scheme": scheme,
        "primary": primary,
        "genome": genome,
        "group_macro": gmacro,
        "alphas": alphas,
        "y": y,
        "pred": pred,
        "groups": groups_arr,
        "mask": mask,
        "locked_alpha": alpha,
    }


def repeated_split_design_contrast(
    X,
    y,
    groups,
    *,
    n_repeats: int = 20,
    n_splits: int = 5,
    seed: int = SEED,
    lock_alpha: bool = True,
) -> dict[str, Any]:
    """For each repeat r: random CV vs blocked CV; collect deltas.

    Default ``lock_alpha=True``: pick α once with group-aware CV, then reuse (GB suite).
    Set ``lock_alpha=False`` for fully nested per-outer-train α (slow).
    """
    alpha = lock_alpha_once(X, y, groups, n_splits=n_splits, seed=seed) if lock_alpha else None
    rows = []
    for r in range(n_repeats):
        rnd = oof_once(
            X, y, groups, blocked=False, n_splits=n_splits, seed=seed + 1000 + r, alpha=alpha
        )
        blk = oof_once(
            X,
            y,
            groups,
            blocked=True,
            n_splits=n_splits,
            seed=seed + 1000 + r,
            group_fold_seed=seed + r,
            alpha=alpha,
        )
        if "error" in rnd or "error" in blk:
            continue
        prim = rnd["primary"]
        rg = float(rnd["genome"][prim])
        bg = float(blk["genome"][prim])
        rm = float(rnd["group_macro"].get(prim, np.nan)) if rnd["group_macro"] else float("nan")
        bm = float(blk["group_macro"].get(prim, np.nan)) if blk["group_macro"] else float("nan")
        rows.append(
            {
                "repeat": r,
                "primary": prim,
                "random_genome": rg,
                "blocked_genome": bg,
                "delta_genome": rg - bg,
                "random_group_macro": rm,
                "blocked_group_macro": bm,
                "delta_group_macro": rm - bm if np.isfinite(rm) and np.isfinite(bm) else float("nan"),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return {"n_repeats": 0, "error": "no_successful_repeats"}

    def _summ(col):
        v = df[col].to_numpy(float)
        v = v[np.isfinite(v)]
        if len(v) == 0:
            return {"mean": float("nan"), "std": float("nan"), "ci95": [float("nan"), float("nan")], "n": 0}
        return {
            "mean": float(np.mean(v)),
            "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
            "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
            "n": int(len(v)),
        }

    return {
        "n_repeats": int(len(df)),
        "primary": str(df["primary"].iloc[0]),
        "tuning": (
            "alpha_locked_once_group_aware_GroupKFold_MSE"
            if lock_alpha
            else "nested_group_aware_Ridge_alpha_on_outer_train"
        ),
        "locked_alpha": float(alpha) if alpha is not None else None,
        "random_genome": _summ("random_genome"),
        "blocked_genome": _summ("blocked_genome"),
        "delta_genome": _summ("delta_genome"),
        "random_group_macro": _summ("random_group_macro"),
        "blocked_group_macro": _summ("blocked_group_macro"),
        "delta_group_macro": _summ("delta_group_macro"),
        "repeats": rows,
    }


def logo_influence(X, y, groups, *, seed: int = SEED, alpha: Optional[float] = None) -> dict[str, Any]:
    """Leave-one-group-out OOF; per-group held-out scores + pooled."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    groups = np.asarray(groups)
    if alpha is None:
        alpha = lock_alpha_once(X, y, groups, seed=seed)
    uniq = sorted(pd.Series(groups).unique().tolist())
    pred = np.full(len(y), np.nan)
    per = []
    for i, g in enumerate(uniq):
        te = np.where(groups == g)[0]
        tr = np.where(groups != g)[0]
        if len(tr) < 5 or len(te) < 1:
            per.append({"group": str(g), "status": "too_small"})
            continue
        p, a = _fit_predict_fold(X, y, tr, te, groups=groups, seed=seed + i, alpha=alpha)
        pred[te] = p
        yg, pg = y[te], p
        entry = {"group": str(g), "n_test": int(len(te)), "alpha": a, "status": "ok"}
        if _is_binary(y) and len(np.unique(yg)) == 2 and len(yg) >= 2:
            entry["note"] = "single_group_held_out"
        if np.nanstd(yg) > 0 and len(yg) >= 3:
            entry["rho_within"] = float(spearmanr(yg, pg)[0])
        per.append(entry)
    genome = _score_genome(y, pred)
    gmacro = _score_group_macro(y, pred, groups)
    prim = "auc" if _is_binary(y) and np.isfinite(genome["auc"]) else "rho"
    base = genome[prim]
    influences = []
    for g in uniq:
        m = groups != g
        sc = _score_genome(y[m], pred[m])
        influences.append(
            {
                "drop_group": str(g),
                "score": sc[prim],
                "delta_from_full": float(sc[prim] - base) if np.isfinite(sc[prim]) else float("nan"),
            }
        )
    return {
        "primary": prim,
        "pooled_genome": genome,
        "pooled_group_macro": gmacro,
        "per_group": per,
        "drop_one_group_influence": influences,
        "locked_alpha": float(alpha),
        "tuning": "alpha_locked_once_group_aware_GroupKFold_MSE",
    }
