#!/usr/bin/env python3
"""P0 circularity + P1 matched shallow baselines on n=632 overlap cohort.

Does NOT rewrite manuscript narrative. Outputs numbers only.

P0:
  Parse SpilloverRankings 'Risk Levels' factor blocks; recompute Total after
  removing human-infection / transmission / epidemic circularity factors
  (Budeski-style). Report OOF Spearman ρ of frozen Evo embeddings vs
  original and ablated scores; also residualize vs binary Host_label.

P1:
  Matched GroupKFold(organism) RidgeCV OOF ρ for:
    - Evo embeddings
    - taxonomy + length + GC
    - 3-mer frequency features
  Same seed/scaler conventions as paper follow-ups.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler

OUT = Path("/home/wangxindi/evo-main/paper/analysis_p0p1_circularity_baselines_wxd0804")
OUT.mkdir(parents=True, exist_ok=True)

EMB = Path("/home/wangxindi/evo-main/analysis_representation_pca_umap/results/embeddings_regression_20260514_202026.csv")
OV = Path("/home/wangxindi/evo/evo_data/processed_test_spillover_overlap_max100_per_org.csv")
RANK = Path("/home/wangxindi/evo-main/evo_data/SpilloverRankings.csv")
SEED = 42
ALPHAS = np.logspace(-3, 3, 13)

# Circularity-prone factor name substrings (case-insensitive). Aligns with
# Budeski & Lipsitch-type post-hoc human infection / transmission knowledge.
CIRCULAR_NAME_PATTERNS = [
    r"infectivity in humans",
    r"known to infect humans in the viral family",
    r"known human pathogens in the viral family",
    r"animal to human transmission",
    r"human to human transmission",
    r"duration of virus species infection in humans",
    r"epidemicity of the virus species",
    r"pandemic virus",
]

# Host-axis factors (Grange SI)
HOST_AXIS_PATTERNS = [
    r"host plasticity - no\. of species",
    r"host plasticity - no\. of orders",
    r"geography of the host",
    r"genetic relatedness between the host species and humans",
]


def parse_risk_levels(text: str) -> list[dict]:
    if not isinstance(text, str) or not text.strip():
        return []
    blocks = re.split(r"\n(?=Risk Name:)", text.strip())
    out = []
    for b in blocks:
        name_m = re.search(r"Risk Name:\s*(.+)", b)
        w_m = re.search(r"Weighted Score:\s*([0-9.]+)", b)
        r_m = re.search(r"Risk Score:\s*([0-9.]+)", b)
        i_m = re.search(r"Impact Score:\s*([0-9.]+)", b)
        if not name_m or not w_m:
            continue
        out.append(
            {
                "name": name_m.group(1).strip(),
                "weighted": float(w_m.group(1)),
                "risk_score": float(r_m.group(1)) if r_m else np.nan,
                "impact": float(i_m.group(1)) if i_m else np.nan,
            }
        )
    return out


def is_circular(name: str) -> bool:
    n = name.lower()
    return any(re.search(p, n) for p in CIRCULAR_NAME_PATTERNS)


def is_host_axis(name: str) -> bool:
    n = name.lower()
    return any(re.search(p, n) for p in HOST_AXIS_PATTERNS)


def score_row(factors: list[dict]) -> dict:
    total = sum(f["weighted"] for f in factors)
    circ = sum(f["weighted"] for f in factors if is_circular(f["name"]))
    host = sum(f["weighted"] for f in factors if is_host_axis(f["name"]))
    return {
        "n_factors": len(factors),
        "total_recon": total,
        "circular_mass": circ,
        "total_ablated_circular": total - circ,
        "host_axis_recon": host,
        "n_circular_factors": sum(1 for f in factors if is_circular(f["name"])),
    }


def oof_ridge_spearman(X: np.ndarray, y: np.ndarray, groups: np.ndarray | None, n_splits=5):
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(y)
    X, y = X[mask], y[mask]
    if groups is not None:
        groups = np.asarray(groups)[mask]
        # drop groups with <2 if needed for GroupKFold
        vc = pd.Series(groups).value_counts()
        keep = np.array([vc.get(g, 0) >= 1 for g in groups])
        # GroupKFold needs n_splits <= n_groups
        n_groups = pd.Series(groups).nunique()
        splits = min(n_splits, n_groups)
        if splits < 2:
            return {"rho": np.nan, "n": int(mask.sum()), "note": "too few groups"}
        cv = GroupKFold(n_splits=splits)
        splitter = cv.split(X, y, groups)
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        splitter = cv.split(X, y)

    pred = np.full(len(y), np.nan)
    for tr, te in splitter:
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[te])
        model = RidgeCV(alphas=ALPHAS)
        model.fit(Xtr, y[tr])
        pred[te] = model.predict(Xte)
    ok = np.isfinite(pred)
    if ok.sum() < 10:
        return {"rho": np.nan, "n": int(ok.sum())}
    rho, p = spearmanr(y[ok], pred[ok])
    return {"rho": float(rho), "p": float(p), "n": int(ok.sum())}


def residualize(y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """OLS residual of y on Z (with intercept)."""
    y = np.asarray(y, dtype=float)
    Z = np.asarray(Z, dtype=float)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    A = np.column_stack([np.ones(len(y)), Z])
    mask = np.isfinite(y) & np.isfinite(A).all(axis=1)
    resid = np.full(len(y), np.nan)
    coef, *_ = np.linalg.lstsq(A[mask], y[mask], rcond=None)
    resid[mask] = y[mask] - A[mask] @ coef
    return resid


def gc_content(seq: str) -> float:
    s = (seq or "").upper()
    if not s:
        return np.nan
    return (s.count("G") + s.count("C")) / len(s)


def kmer_counts(seq: str, k=3) -> np.ndarray:
    s = re.sub(r"[^ACGT]", "", (seq or "").upper())
    alphabet = "ACGT"
    idx = {a + b + c: i for i, (a, b, c) in enumerate(
        (x, y, z) for x in alphabet for y in alphabet for z in alphabet
    )} if k == 3 else None
    if k != 3:
        raise ValueError("only k=3 implemented")
    v = np.zeros(4**k, dtype=float)
    if len(s) < k:
        return v
    for i in range(len(s) - k + 1):
        mer = s[i : i + k]
        if mer in idx:
            v[idx[mer]] += 1
    tot = v.sum()
    if tot > 0:
        v /= tot
    return v


def main():
    print("=== load ===", flush=True)
    emb = pd.read_csv(EMB)
    ov = pd.read_csv(OV, usecols=["Accession", "Organism_Name", "Family", "Length", "Sequence", "Host"])
    rank = pd.read_csv(RANK)

    emb_cols = [c for c in emb.columns if c.startswith("emb_")]
    X_emb = emb[emb_cols].to_numpy(dtype=float)
    groups = emb["Organism_Name"].to_numpy()

    # --- parse factor-level scores from Risk Levels ---
    print("=== parse Risk Levels ===", flush=True)
    parsed = []
    factor_catalog = {}
    for _, row in rank.iterrows():
        factors = parse_risk_levels(row.get("Risk Levels", ""))
        for f in factors:
            factor_catalog[f["name"]] = factor_catalog.get(f["name"], 0) + 1
        sc = score_row(factors)
        sc["Virus Name"] = row["Virus Name"]
        sc["Human Virus?"] = row.get("Human Virus?")
        sc["Zoonotic Virus?"] = row.get("Zoonotic Virus?")
        sc["Human Transmission?"] = row.get("Human Transmission?")
        sc["Host Score_table"] = row.get(" Host Score") if " Host Score" in rank.columns else row.get("Host Score")
        # column might have leading space
        for cand in [" Host Score", "Host Score"]:
            if cand in rank.columns:
                sc["Host Score_table"] = row[cand]
                break
        sc["Total Score_table"] = row.get("Total Score")
        sc["Virus Score_table"] = row.get("Virus Score")
        sc["Environment Score_table"] = row.get("Environment Score")
        parsed.append(sc)
    pdf = pd.DataFrame(parsed)

    # catalog
    cat_rows = []
    for name, cnt in sorted(factor_catalog.items()):
        cat_rows.append(
            {
                "factor_name": name,
                "n_rows_with_factor": cnt,
                "is_circular_flag": is_circular(name),
                "is_host_axis": is_host_axis(name),
            }
        )
    cat = pd.DataFrame(cat_rows)
    cat.to_csv(OUT / "Table_P0_spillover_factor_catalog.csv", index=False)
    print("n unique factors", len(cat), "circular flagged", int(cat["is_circular_flag"].sum()), flush=True)

    # aggregate by Virus Name (median) then merge to emb on Organism_Name == Virus Name
    agg = (
        pdf.groupby("Virus Name", as_index=False)
        .agg(
            {
                "total_recon": "median",
                "total_ablated_circular": "median",
                "circular_mass": "median",
                "host_axis_recon": "median",
                "n_factors": "median",
                "n_circular_factors": "median",
                "Host Score_table": "median",
                "Total Score_table": "median",
                "Virus Score_table": "median",
                "Environment Score_table": "median",
            }
        )
    )
    # flags: any Yes
    flags = (
        pdf.groupby("Virus Name")
        .agg(
            human_virus=("Human Virus?", lambda s: int((s.astype(str).str.lower() == "yes").any())),
            zoonotic=("Zoonotic Virus?", lambda s: int((s.astype(str).str.lower() == "yes").any())),
            human_transmission=("Human Transmission?", lambda s: int((s.astype(str).str.lower() == "yes").any())),
        )
        .reset_index()
    )
    agg = agg.merge(flags, on="Virus Name", how="left")

    m = emb.merge(agg, left_on="Organism_Name", right_on="Virus Name", how="left")
    print("merge coverage", float(m["Virus Name"].notna().mean()), flush=True)
    m.to_csv(OUT / "Table_P0_overlap_with_ablated_scores.csv", index=False)

    # recon fidelity vs published scores
    fidelity = {
        "corr_total_recon_vs_table": float(spearmanr(m["total_recon"], m["Total Score_table"], nan_policy="omit").correlation),
        "corr_total_recon_vs_emb_spillover_total": float(spearmanr(m["total_recon"], m["spillover_total"], nan_policy="omit").correlation),
        "corr_host_axis_vs_emb_host": float(spearmanr(m["host_axis_recon"], m["spillover_host_score"], nan_policy="omit").correlation),
        "corr_ablated_vs_total": float(spearmanr(m["total_ablated_circular"], m["spillover_total"], nan_policy="omit").correlation),
        "median_circular_mass_fraction": float(np.nanmedian(m["circular_mass"] / m["total_recon"].replace(0, np.nan))),
    }
    print("fidelity", fidelity, flush=True)

    # --- P0 OOF ρ ---
    print("=== P0 OOF ρ ===", flush=True)
    targets = {
        "spillover_total": m["spillover_total"].to_numpy(),
        "spillover_host_score": m["spillover_host_score"].to_numpy(),
        "spillover_virus_score": m["spillover_virus_score"].to_numpy(),
        "spillover_env_score": m["spillover_env_score"].to_numpy(),
        "total_recon_from_factors": m["total_recon"].to_numpy(),
        "total_ablated_circular_factors": m["total_ablated_circular"].to_numpy(),
        "host_axis_recon": m["host_axis_recon"].to_numpy(),
    }
    # residualize totals against binary host label
    hl = m["Host_label"].to_numpy(dtype=float)
    targets["spillover_total_resid_Host_label"] = residualize(m["spillover_total"].to_numpy(), hl)
    targets["spillover_host_resid_Host_label"] = residualize(m["spillover_host_score"].to_numpy(), hl)
    targets["total_ablated_resid_Host_label"] = residualize(m["total_ablated_circular"].to_numpy(), hl)

    p0_rows = []
    for name, y in targets.items():
        for scheme, g in [("shuffle_KFold", None), ("GroupKFold_organism", groups)]:
            res = oof_ridge_spearman(X_emb, y, g)
            p0_rows.append({"target": name, "cv": scheme, **res})
            print(f"  {name} | {scheme} | rho={res.get('rho')} n={res.get('n')}", flush=True)

    # subset: Human Virus? == No
    sub = m["human_virus"].fillna(0).to_numpy() == 0
    for name, y in [
        ("spillover_total_nonHumanVirus", m["spillover_total"].to_numpy()),
        ("total_ablated_nonHumanVirus", m["total_ablated_circular"].to_numpy()),
        ("spillover_host_nonHumanVirus", m["spillover_host_score"].to_numpy()),
    ]:
        yy = y.copy()
        yy[~sub] = np.nan
        res = oof_ridge_spearman(X_emb, yy, groups)
        p0_rows.append({"target": name, "cv": "GroupKFold_organism", **res, "n_eligible": int(sub.sum())})
        print(f"  {name} | rho={res.get('rho')} eligible={sub.sum()}", flush=True)

    p0 = pd.DataFrame(p0_rows)
    p0.to_csv(OUT / "Table_P0_circularity_oof_rho.csv", index=False)

    # --- P1 matched baselines ---
    print("=== P1 baselines ===", flush=True)
    # attach sequences
    seqmap = ov.drop_duplicates("Accession").set_index("Accession")
    m["Sequence"] = m["Accession"].map(seqmap["Sequence"])
    m["Length"] = m["Accession"].map(seqmap["Length"])
    m["Family"] = m["Accession"].map(seqmap["Family"])
    m["GC"] = m["Sequence"].map(gc_content)

    # taxonomy one-hot
    fam = pd.get_dummies(m["Family"].fillna("NA"), prefix="fam")
    X_tax = np.column_stack([fam.to_numpy(dtype=float), m["Length"].to_numpy(dtype=float), m["GC"].to_numpy(dtype=float)])

    print("  computing 3-mers...", flush=True)
    X_kmer = np.vstack([kmer_counts(s, 3) for s in m["Sequence"].tolist()])

    y_total = m["spillover_total"].to_numpy()
    p1_rows = []
    for feat_name, X in [
        ("evo_embedding", X_emb),
        ("taxonomy_length_GC", X_tax),
        ("kmer3_freq", X_kmer),
    ]:
        for scheme, g in [("shuffle_KFold", None), ("GroupKFold_organism", groups)]:
            res = oof_ridge_spearman(X, y_total, g)
            p1_rows.append({"features": feat_name, "target": "spillover_total", "cv": scheme, **res})
            print(f"  {feat_name} | {scheme} | rho={res.get('rho')}", flush=True)
        # also vs host score
        res_h = oof_ridge_spearman(X, m["spillover_host_score"].to_numpy(), groups)
        p1_rows.append({"features": feat_name, "target": "spillover_host_score", "cv": "GroupKFold_organism", **res_h})
        # vs ablated total
        res_a = oof_ridge_spearman(X, m["total_ablated_circular"].to_numpy(), groups)
        p1_rows.append({"features": feat_name, "target": "total_ablated_circular_factors", "cv": "GroupKFold_organism", **res_a})

    p1 = pd.DataFrame(p1_rows)
    p1.to_csv(OUT / "Table_P1_matched_baselines_oof_rho.csv", index=False)

    # CoV expansion readiness note (no IG yet)
    cov = ov[ov["Family"] == "Coronaviridae"].copy()
    existing = [
        "EF065509", "GU190215", "KF268336", "KF268337", "MF370205", "MG762674",
        "MT549854", "MW218395", "NC_048211", "ON648240", "OQ297728",
    ]
    cov["in_current_panel"] = cov["Accession"].isin(existing)
    # stratified sample candidates: up to 30 unique organism, prefer length>20000
    cand = cov[~cov["in_current_panel"]].copy()
    cand = cand.sort_values("Length", ascending=False)
    # one per organism first
    cand_u = cand.drop_duplicates("Organism_Name").head(40)
    expand = pd.concat([
        cov[cov["in_current_panel"]].drop_duplicates("Accession"),
        cand_u,
    ], ignore_index=True)
    expand.to_csv(OUT / "Table_P1_cov_panel_expansion_candidates.csv", index=False)

    meta = {
        "seed": SEED,
        "n_emb": int(len(emb)),
        "circular_patterns": CIRCULAR_NAME_PATTERNS,
        "host_axis_patterns": HOST_AXIS_PATTERNS,
        "fidelity": fidelity,
        "n_circular_factor_types": int(cat["is_circular_flag"].sum()),
        "circular_factor_names": cat.loc[cat["is_circular_flag"], "factor_name"].tolist(),
        "host_axis_factor_names": cat.loc[cat["is_host_axis"], "factor_name"].tolist(),
        "p0_highlights": p0.to_dict(orient="records"),
        "p1_highlights": p1.to_dict(orient="records"),
        "cov_current_panel_n": int(cov["in_current_panel"].sum()),
        "cov_overlap_n": int(len(cov)),
        "cov_expansion_candidate_rows": int(len(expand)),
        "note": (
            "Ablation removes weighted factor mass for human-infection/transmission/epidemic "
            "factors parsed from SpilloverRankings Risk Levels text. Not a retrain of SpillOver; "
            "readout-only sensitivity on frozen embeddings. Narrative unchanged."
        ),
    }
    (OUT / "run_meta_p0p1_wxd0804.json").write_text(json.dumps(meta, indent=2))
    print("DONE", OUT, flush=True)


if __name__ == "__main__":
    main()
