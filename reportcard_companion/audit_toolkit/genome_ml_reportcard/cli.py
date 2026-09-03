#!/usr/bin/env python3
"""CLI entry: genome-ml-reportcard"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler

from genome_ml_reportcard import SCHEMA_COLUMNS, __version__
from genome_ml_reportcard.geometry import geometry_report, overlap_audit
from genome_ml_reportcard.report import write_markdown_report

SEED = 42


def _is_binary(y: np.ndarray) -> bool:
    u = np.unique(y[np.isfinite(y)])
    return len(u) == 2 and set(np.round(u).tolist()).issubset({0.0, 1.0})


def oof_ridge(X, y, groups=None, n_splits=5, seed=SEED):
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    mask = np.isfinite(y) & np.isfinite(X).all(1)
    X, y = X[mask], y[mask]
    groups_arr = None if groups is None else np.asarray(groups)[mask]
    if groups_arr is not None:
        n_g = int(pd.Series(groups_arr).nunique())
        splits = max(2, min(n_splits, n_g))
        splitter = GroupKFold(n_splits=splits).split(X, y, groups_arr)
        scheme = f"GroupKFold_k{splits}"
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X, y)
        scheme = f"shuffle_KFold_k{n_splits}"
    pred = np.full(len(y), np.nan)
    for tr, te in splitter:
        sc = StandardScaler()
        m = RidgeCV(alphas=np.logspace(-3, 3, 13))
        m.fit(sc.fit_transform(X[tr]), y[tr])
        pred[te] = m.predict(sc.transform(X[te]))
    ok = np.isfinite(pred)
    out = {"n": int(ok.sum()), "scheme": scheme, "rho": float("nan"), "auc": float("nan")}
    if ok.sum() >= 10:
        out["rho"] = float(spearmanr(y[ok], pred[ok])[0])
        if _is_binary(y[ok]):
            try:
                out["auc"] = float(roc_auc_score(y[ok], pred[ok]))
            except ValueError:
                out["auc"] = float("nan")
    return out


def probe_gap(X, y, blocks, *, n_splits=5, seed=SEED):
    rnd = oof_ridge(X, y, groups=None, n_splits=n_splits, seed=seed)
    blk = oof_ridge(X, y, groups=blocks, n_splits=n_splits, seed=seed)
    primary = "auc" if _is_binary(y) and np.isfinite(rnd.get("auc", np.nan)) else "rho"
    delta = float(rnd[primary] - blk[primary]) if np.isfinite(rnd[primary]) and np.isfinite(blk[primary]) else float("nan")
    sizes = pd.Series(blocks).astype(str).value_counts()
    warnings = []
    if (sizes == 1).all():
        warnings.append("ALL_SINGLETON_GROUPS: random≈blocked by construction")
    if len(sizes) < 10:
        warnings.append("FEW_GROUPS: n_blocks<10; interpret contrasts cautiously")
    return {
        "seed": seed,
        "schema": list(SCHEMA_COLUMNS),
        "primary_metric": primary,
        "random": rnd,
        "blocked": blk,
        "delta": delta,
        "delta_rho": float(rnd["rho"] - blk["rho"]) if np.isfinite(rnd["rho"]) and np.isfinite(blk["rho"]) else float("nan"),
        "n": int(len(y)),
        "n_groups": int(len(sizes)),
        "group_size_min": int(sizes.min()),
        "group_size_max": int(sizes.max()),
        "frac_singleton_groups": float((sizes == 1).mean()),
        "warnings": warnings,
    }


def _pick_col(df: pd.DataFrame, requested: str, aliases: list[str]) -> str:
    if requested in df.columns:
        return requested
    for a in aliases:
        if a in df.columns:
            return a
    raise SystemExit(f"Column not found: {requested!r} (aliases tried: {aliases})")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="genome-ml-reportcard",
        description=(
            f"GenomeML Report Card v{__version__}: audit label-assignment units, "
            "blocking units, sequence/group overlap, and random vs blocked split contrast."
        ),
    )
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    ap.add_argument("--table", required=True, type=Path, help="Manifest TSV/CSV")
    ap.add_argument("--table-b", type=Path, default=None, help="Optional second split for overlap audit")
    ap.add_argument("--accession", default="sequence_id", help="ID column")
    ap.add_argument("--group", default="group", help="Label-assignment unit column (Layer A)")
    ap.add_argument(
        "--block",
        default=None,
        help="Blocking / deployment unit column (Layer B). Defaults to --group.",
    )
    ap.add_argument("--label", default="label", help="Phenotype column")
    ap.add_argument("--features", type=Path, default=None, help="Optional .npy feature matrix (rows=table)")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", type=Path, required=True, help="JSON report path")
    ap.add_argument("--md-out", type=Path, default=None, help="Markdown report path (default: --out with .md)")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    df = pd.read_csv(args.table, sep=None, engine="python")
    acc = _pick_col(df, args.accession, ["sequence_id", "accession", "genome_id"])
    grp = _pick_col(df, args.group, ["group", "species", "organism"])
    lab = _pick_col(df, args.label, ["label", "ogt_c", "y"])
    if args.block:
        blk_col = _pick_col(df, args.block, [args.block, "block", "cluster", "Viral group"])
    else:
        blk_col = grp

    y = pd.to_numeric(df[lab], errors="coerce").to_numpy(float)
    label_groups = df[grp].astype(str).to_numpy()
    blocks = df[blk_col].astype(str).to_numpy()

    report = {
        "tool": "genome-ml-reportcard",
        "version": __version__,
        "n_rows": int(len(df)),
        "n_groups": int(pd.Series(label_groups).nunique()),
        "columns": {
            "accession": acc,
            "label_assignment_unit": grp,
            "blocking_unit": blk_col,
            "label": lab,
        },
        "geometry": geometry_report(y, label_groups, blocks, n_splits=args.n_splits, seed=args.seed),
    }

    if args.table_b is not None:
        b = pd.read_csv(args.table_b, sep=None, engine="python")
        # normalize id/block names if possible
        b_acc = acc if acc in b.columns else _pick_col(b, args.accession, ["sequence_id", "accession", "genome_id"])
        b_blk = blk_col if blk_col in b.columns else (grp if grp in b.columns else None)
        a_norm = pd.DataFrame({"sequence_id": df[acc].astype(str)})
        b_norm = pd.DataFrame({"sequence_id": b[b_acc].astype(str)})
        if b_blk:
            a_norm["group"] = df[blk_col].astype(str)
            b_norm["group"] = b[b_blk].astype(str)
        report["overlap"] = overlap_audit(a_norm, b_norm, id_col="sequence_id", block_col="group" if b_blk else None)

    if args.features is not None:
        X = np.load(args.features)
        if len(X) != len(df):
            raise SystemExit(f"features {len(X)} != table {len(df)}")
        report["probe"] = probe_gap(X, y, blocks, n_splits=args.n_splits, seed=args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    md_path = args.md_out or args.out.with_suffix(".md")
    write_markdown_report(report, md_path)
    print("wrote", args.out)
    print("wrote", md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
