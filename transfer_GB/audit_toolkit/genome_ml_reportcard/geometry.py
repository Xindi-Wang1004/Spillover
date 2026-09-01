"""Label-geometry and overlap diagnostics for GenomeML Report Card."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def _is_binary(y: np.ndarray) -> bool:
    u = np.unique(y[np.isfinite(y)])
    return len(u) <= 2 and set(np.round(u).tolist()).issubset({0.0, 1.0})


def majority_purity(y: np.ndarray) -> float:
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    if len(y) == 0:
        return float("nan")
    vals, counts = np.unique(y, return_counts=True)
    return float(counts.max() / counts.sum())


def icc_oneway(y: np.ndarray, groups: np.ndarray) -> float:
    """One-way random-effects ICC (between / total)."""
    df = pd.DataFrame({"y": y, "g": groups}).dropna()
    if df.empty or df["g"].nunique() < 2:
        return float("nan")
    grand = df["y"].mean()
    ss_total = ((df["y"] - grand) ** 2).sum()
    if ss_total <= 0:
        return 1.0 if df.groupby("g")["y"].nunique().max() <= 1 else float("nan")
    means = df.groupby("g")["y"].mean()
    sizes = df.groupby("g")["y"].size()
    ss_between = ((means - grand) ** 2 * sizes).sum()
    return float(np.clip(ss_between / ss_total, 0.0, 1.0))


def within_block_homogeneity(y: np.ndarray, blocks: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, float)
    blocks = np.asarray(blocks)
    mask = np.isfinite(y)
    y, blocks = y[mask], blocks[mask]
    if _is_binary(y):
        purities = [majority_purity(y[blocks == b]) for b in np.unique(blocks)]
        return {
            "metric_type": "majority_purity",
            "within_block_homogeneity": float(np.nanmean(purities)),
            "label_icc_or_purity": float(majority_purity(y)),  # overall for binary is less informative
        }
    return {
        "metric_type": "ICC",
        "within_block_homogeneity": icc_oneway(y, blocks),
        "label_icc_or_purity": icc_oneway(y, blocks),
    }


def block_size_stats(blocks: np.ndarray) -> dict[str, Any]:
    sizes = pd.Series(blocks).astype(str).value_counts()
    return {
        "n_blocks": int(len(sizes)),
        "median_block_size": float(sizes.median()) if len(sizes) else float("nan"),
        "pct_singleton_blocks": float((sizes == 1).mean()) if len(sizes) else float("nan"),
        "group_size_min": int(sizes.min()) if len(sizes) else 0,
        "group_size_max": int(sizes.max()) if len(sizes) else 0,
        "frac_singleton_groups": float((sizes == 1).mean()) if len(sizes) else float("nan"),
    }


def random_cv_shared_block_fraction(blocks: np.ndarray, n_splits: int = 5, seed: int = 42) -> float:
    """Fraction of test rows whose block also appears in the train fold (mean over folds)."""
    blocks = np.asarray(blocks).astype(str)
    n = len(blocks)
    if n < n_splits * 2:
        return float("nan")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fracs = []
    for tr, te in kf.split(np.arange(n)):
        train_blocks = set(blocks[tr])
        shared = sum(1 for b in blocks[te] if b in train_blocks)
        fracs.append(shared / max(len(te), 1))
    return float(np.mean(fracs))


def geometry_report(
    y: np.ndarray,
    label_groups: np.ndarray,
    blocks: np.ndarray,
    *,
    n_splits: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    y = np.asarray(y, float)
    label_groups = np.asarray(label_groups).astype(str)
    blocks = np.asarray(blocks).astype(str)
    out: dict[str, Any] = {
        "n_rows": int(np.isfinite(y).sum()),
        "n_label_groups": int(pd.Series(label_groups).nunique()),
        "label_unit_stats": block_size_stats(label_groups),
        "block_stats": block_size_stats(blocks),
        "random_cv_shared_block_fraction": random_cv_shared_block_fraction(blocks, n_splits=n_splits, seed=seed),
    }
    out.update(within_block_homogeneity(y, blocks))
    # For Layer A diagnostics, also report label-unit homogeneity
    lab_h = within_block_homogeneity(y, label_groups)
    out["within_label_unit_homogeneity"] = lab_h["within_block_homogeneity"]
    out["label_unit_metric_type"] = lab_h["metric_type"]
    return out


def overlap_audit(
    a: pd.DataFrame,
    b: pd.DataFrame,
    *,
    id_col: str = "sequence_id",
    block_col: str | None = "group",
) -> dict[str, Any]:
    ids_a = set(a[id_col].astype(str))
    ids_b = set(b[id_col].astype(str))
    out: dict[str, Any] = {
        "n_a": len(a),
        "n_b": len(b),
        "accession_overlap": len(ids_a & ids_b),
        "accession_overlap_frac_b": float(len(ids_a & ids_b) / max(len(ids_b), 1)),
    }
    if block_col and block_col in a.columns and block_col in b.columns:
        ga, gb = set(a[block_col].astype(str)), set(b[block_col].astype(str))
        out["block_overlap"] = len(ga & gb)
        out["block_overlap_frac_b"] = float(len(ga & gb) / max(len(gb), 1))
    return out
