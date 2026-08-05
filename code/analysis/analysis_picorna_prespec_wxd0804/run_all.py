#!/usr/bin/env python3
"""Prespecified Picornaviridae / Rhinovirus C second-family enrichment.

PRIMARY (frozen 2026-08-04): regression sparse 3-window peak-fragment Fisher
(mirrors CoV OR=36 design). Multitask = descriptive only.
Dense W=5 deferred (no dense tracks). See REPRO_REGISTRY_picorna_prespec_wxd0804.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE = Path("/home/wangxindi/evo/evo_data/ig_species_wxd0729/Rhinovirus_C")
OUT = Path("/home/wangxindi/evo-main/paper/analysis_picorna_prespec_wxd0804")
OUT.mkdir(parents=True, exist_ok=True)

CLASSES = ["entry_interface", "polymerase_replicase", "other"]  # locked order
PRIMARY_MODEL = "regression"
DESC_MODEL = "multitask"


def fisher_block(df: pd.DataFrame, cls: str) -> dict:
    """High = max window_attr_sum within accession; class from top_fragment_class."""
    a = b = c = d = 0
    per = []
    for acc, g in df.groupby("accession"):
        g = g.sort_values("window_attr_sum", ascending=False)
        top_idx = g.iloc[0]["window_index"]
        for _, r in g.iterrows():
            high = r["window_index"] == top_idx
            in_cls = str(r["top_fragment_class"]) == cls
            if high and in_cls:
                a += 1
            elif high and not in_cls:
                b += 1
            elif (not high) and in_cls:
                c += 1
            else:
                d += 1
        top = g.iloc[0]
        per.append(
            {
                "accession": acc,
                "top_window_index": int(top["window_index"]),
                "top_fragment_class": top["top_fragment_class"],
                "top_in_class": str(top["top_fragment_class"]) == cls,
                "window_attr_sum": float(top["window_attr_sum"]),
            }
        )
    table = np.array([[a, b], [c, d]], dtype=int)
    odds, p = stats.fisher_exact(table)
    aa, bb, cc, dd = table.astype(float).ravel()
    or_h = ((aa + 0.5) * (dd + 0.5)) / ((bb + 0.5) * (cc + 0.5))
    return {
        "class": cls,
        "high_in": a,
        "high_out": b,
        "low_in": c,
        "low_out": d,
        "odds_ratio": float(odds) if np.isfinite(odds) else np.nan,
        "odds_ratio_haldane": float(or_h),
        "p_fisher": float(p),
        "n_genomes": int(df["accession"].nunique()),
        "n_top_in_class": int(sum(1 for r in per if r["top_in_class"])),
        "per_accession": per,
        "contingency": [a, b, c, d],
    }


def bh_fdr(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    q = np.empty(m, dtype=float)
    cum = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        cum = min(cum, p[i] * m / (rank + 1))
        q[i] = cum
    return q.tolist()


def analyze_model(path: Path, model: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df = pd.read_csv(path)
    assert df["model"].nunique() == 1 or (df["model"] == model).all() or "model" in df.columns
    if "model" in df.columns:
        df = df[df["model"] == model].copy() if (df["model"] == model).any() else df.copy()
    rows = []
    per_rows = []
    for cls in CLASSES:
        block = fisher_block(df, cls)
        rows.append({k: v for k, v in block.items() if k != "per_accession"})
        for r in block["per_accession"]:
            per_rows.append({"model": model, "tested_class": cls, **r})
    summ = pd.DataFrame(rows)
    summ["q_bh_across_3_classes"] = bh_fdr(summ["p_fisher"].tolist())
    summ["model"] = model
    summ["role"] = "primary" if model == PRIMARY_MODEL else "descriptive_concordance"
    # sign test for entry
    cov = pd.read_csv(BASE / "annotation_mask_coverage.csv")
    p0 = float(cov["frac_entry"].mean())
    entry_row = summ[summ["class"] == "entry_interface"].iloc[0]
    k = int(entry_row["n_top_in_class"])
    n = int(entry_row["n_genomes"])
    sign_p = float(stats.binomtest(k, n, p=min(max(p0, 1e-6), 1 - 1e-6), alternative="greater").pvalue)
    sign_p_anti = float(stats.binomtest(k, n, p=0.5, alternative="greater").pvalue)
    meta = {
        "model": model,
        "mean_frac_entry": p0,
        "sign_test_entry_p_vs_coverage": sign_p,
        "sign_test_entry_p_vs_0.5_anticonservative": sign_p_anti,
        "n_top_entry": k,
        "n_genomes": n,
    }
    return summ, pd.DataFrame(per_rows), meta


def main():
    reg_sum, reg_per, reg_meta = analyze_model(
        BASE / "sliding_ig_regression_windows_long.csv", PRIMARY_MODEL
    )
    mt_sum, mt_per, mt_meta = analyze_model(
        BASE / "sliding_ig_multitask_windows_long.csv", DESC_MODEL
    )

    # per-accession primary table (top class only, regression)
    reg = pd.read_csv(BASE / "sliding_ig_regression_windows_long.csv")
    tops = []
    for acc, g in reg.groupby("accession"):
        top = g.sort_values("window_attr_sum", ascending=False).iloc[0]
        tops.append(
            {
                "accession": acc,
                "model": PRIMARY_MODEL,
                "top_window_index": int(top["window_index"]),
                "window_start": int(top["window_start"]),
                "window_end": int(top["window_end"]),
                "top_fragment_start": int(top["top_fragment_start"]),
                "top_fragment_end": int(top["top_fragment_end"]),
                "top_fragment_class": top["top_fragment_class"],
                "window_attr_sum": float(top["window_attr_sum"]),
            }
        )
    top_df = pd.DataFrame(tops)

    summary = pd.concat([reg_sum, mt_sum], ignore_index=True)
    summary.to_csv(OUT / "Table_S_Picorna_enrichment_summary.csv", index=False)
    top_df.to_csv(OUT / "Table_S_Picorna_per_accession_window_classes.csv", index=False)

    # concordance: same top class?
    mt = pd.read_csv(BASE / "sliding_ig_multitask_windows_long.csv")
    mt_tops = []
    for acc, g in mt.groupby("accession"):
        top = g.sort_values("window_attr_sum", ascending=False).iloc[0]
        mt_tops.append({"accession": acc, "mt_top_class": top["top_fragment_class"]})
    conc = top_df.merge(pd.DataFrame(mt_tops), on="accession")
    conc["concordant_top_class"] = conc["top_fragment_class"] == conc["mt_top_class"]
    conc.to_csv(OUT / "Table_S_Picorna_multitask_concordance.csv", index=False)

    primary_entry = reg_sum[reg_sum["class"] == "entry_interface"].iloc[0].to_dict()
    meta = {
        "frozen_registry": "REPRO_REGISTRY_picorna_prespec_wxd0804.md",
        "frozen_date": "2026-08-04",
        "primary_model": PRIMARY_MODEL,
        "descriptive_model": DESC_MODEL,
        "classes_prespecified": CLASSES,
        "window_design": "sparse_sliding_3_windows_peakfrag_mirrors_CoV_OR36",
        "dense_W5_status": "deferred_no_dense_tracks",
        "regression_meta": reg_meta,
        "multitask_meta": mt_meta,
        "primary_entry_interface": primary_entry,
        "concordance_frac": float(conc["concordant_top_class"].mean()),
        "n_accessions": int(top_df["accession"].nunique()),
        "note": (
            "Dense equal-partition W=5 not run: no dense stitched tracks for this panel. "
            "Primary mirrors main-text CoV sparse-window design. Report regression as-is; "
            "do not switch primary to multitask."
        ),
    }
    (OUT / "run_meta_picorna_prespec_wxd0804.json").write_text(json.dumps(meta, indent=2, default=str))

    print("=== PRIMARY regression ===", flush=True)
    print(reg_sum.to_string(index=False), flush=True)
    print("sign/coverage", reg_meta, flush=True)
    print("=== DESCRIPTIVE multitask ===", flush=True)
    print(mt_sum.to_string(index=False), flush=True)
    print("concordance", meta["concordance_frac"], flush=True)
    print("DONE", OUT, flush=True)


if __name__ == "__main__":
    main()
