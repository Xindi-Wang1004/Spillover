#!/usr/bin/env python3
"""Minimal group-label audit report card.

Given an accession table with a group column and numeric labels, report:
  1) Overlap audit between two splits (accession / optional MD5)
  2) Random vs group-blocked Ridge probe (Spearman ρ) + Δρ

Example:
  python -m audit_toolkit.report_card \\
    --table manifest.tsv --accession accession --group species --label ogt_c \\
    --features X.npy --out report.json

Or feature-free overlap-only:
  python -m audit_toolkit.report_card --table A.csv --table-b B.csv \\
    --accession accession --group organism --overlap-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "third_domain_ogt_large" / "lib"))
try:
    from probe_lib import SEED, oof_ridge_spearman
except ImportError:
    sys.path.insert(0, str(ROOT.parent.parent / "upgrade_post_sb_reject" / "analysis_GB_fillins"))
    from probe_lib import SEED, oof_ridge_spearman  # type: ignore


def md5_file(path: Path, max_bytes: int = 0) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        if max_bytes > 0:
            h.update(f.read(max_bytes))
        else:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def overlap_audit(a: pd.DataFrame, b: pd.DataFrame, accession: str, group: str | None, fasta_col: str | None):
    sa = set(a[accession].astype(str))
    sb = set(b[accession].astype(str))
    inter = sa & sb
    out = {
        "n_a": len(sa),
        "n_b": len(sb),
        "accession_overlap": len(inter),
        "accession_overlap_ids": sorted(inter)[:50],
        "claim_sequence_disjoint": len(inter) == 0,
    }
    if group and group in a.columns and group in b.columns:
        ga = set(a[group].astype(str))
        gb = set(b[group].astype(str))
        go = ga & gb
        out["n_groups_a"] = len(ga)
        out["n_groups_b"] = len(gb)
        out["group_overlap"] = len(go)
        out["group_overlap_ids"] = sorted(go)[:50]
        out["claim_group_disjoint"] = len(go) == 0
    if fasta_col and fasta_col in a.columns and fasta_col in b.columns and len(inter) == 0:
        # optional MD5 on files that exist
        md5a = {}
        md5b = {}
        for _, r in a.iterrows():
            p = Path(str(r[fasta_col]))
            if p.is_file():
                md5a[md5_file(p, max_bytes=1_000_000)] = str(r[accession])
        for _, r in b.iterrows():
            p = Path(str(r[fasta_col]))
            if p.is_file():
                md5b[md5_file(p, max_bytes=1_000_000)] = str(r[accession])
        shared = set(md5a) & set(md5b)
        out["md5_prefix1MB_overlap"] = len(shared)
        out["claim_md5_disjoint"] = len(shared) == 0
    return out


def probe_gap(X: np.ndarray, y: np.ndarray, groups: np.ndarray):
    rnd = oof_ridge_spearman(X, y, groups=None)
    blk = oof_ridge_spearman(X, y, groups=groups)
    delta = float(rnd["rho"] - blk["rho"]) if np.isfinite(rnd["rho"]) and np.isfinite(blk["rho"]) else float("nan")
    return {
        "seed": SEED,
        "random": rnd,
        "blocked": blk,
        "delta_rho": delta,
        "n": int(len(y)),
        "n_groups": int(pd.Series(groups).nunique()),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Group-label audit report card")
    ap.add_argument("--table", required=True, help="Primary table (CSV/TSV)")
    ap.add_argument("--table-b", default=None, help="Second split for overlap audit")
    ap.add_argument("--accession", default="accession")
    ap.add_argument("--group", default="species")
    ap.add_argument("--label", default="ogt_c")
    ap.add_argument("--fasta-col", default=None)
    ap.add_argument("--features", default=None, help="numpy .npy feature matrix aligned to --table rows")
    ap.add_argument("--overlap-only", action="store_true")
    ap.add_argument("--out", default="audit_report_card.json")
    args = ap.parse_args(argv)

    sep = "\t" if args.table.endswith(".tsv") else ","
    a = pd.read_csv(args.table, sep=sep)
    report = {"seed": SEED, "table": args.table}

    if args.table_b:
        sep_b = "\t" if args.table_b.endswith(".tsv") else ","
        b = pd.read_csv(args.table_b, sep=sep_b)
        report["overlap"] = overlap_audit(a, b, args.accession, args.group, args.fasta_col)

    if not args.overlap_only:
        if args.features is None:
            if args.table_b:
                pass  # overlap-only path acceptable
            else:
                ap.error("--features required unless --overlap-only with --table-b")
        else:
            X = np.load(args.features)
            if len(X) != len(a):
                raise SystemExit(f"features n={len(X)} != table n={len(a)}")
            y = a[args.label].to_numpy(float)
            g = a[args.group].astype(str).to_numpy()
            report["probe"] = probe_gap(X, y, g)

    out = Path(args.out)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
