#!/usr/bin/env python3
"""Dense-track denser-window ORF1ab/polymerase enrichment on the same 11 CoV accessions.

Preferred CoV route (no new organisms): partition each dense stitched track into
W equal windows (W in {3,5,7,10}), define high = top-1 window by mean |attr|,
label polymerase_replicase if >=50% nt overlap with GenBank ORF1ab/replicase
mask (join-aware), Fisher OR + binomial + Stouffer on per-genome ORF enrichment.

Also reports a parallel contrast using ORF1ab C-terminal third (Table32) as a
stricter RdRp-like mask. Does not change manuscript narrative.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

OUT = Path("/home/wangxindi/evo-main/paper/analysis_cov_dense_windows_wxd0804")
OUT.mkdir(parents=True, exist_ok=True)
TRACK_DIR = Path("/home/wangxindi/evo/evo_data/ig_sliding_dense/genome_tracks")
GB_CACHE = Path("/home/wangxindi/evo/evo_data/genbank_cache_cov_subset")
T32 = Path("/home/wangxindi/evo-main/paper/bib_tables/Table32_known_loci_attribution_scores.csv")
T27 = Path("/home/wangxindi/evo-main/paper/bib_tables/Table27_host_interaction_region_lengths.csv")

ACCESSIONS = [
    "EF065509", "GU190215", "KF268336", "KF268337", "MF370205", "MG762674",
    "MT549854", "MW218395", "NC_048211", "ON648240", "OQ297728",
]

POL_KEYS = (
    "orf1ab", "orf1a", "pp1ab", "pp1a", "replicase", "polyprotein 1ab",
    "1ab polyprotein", "orf1ab polyprotein", "nonstructural polyprotein",
    "rna-dependent rna polymerase", "rdRp".lower(),
)


def load_track(acc: str) -> pd.DataFrame:
    p = TRACK_DIR / f"{acc}_stitched_attr.csv"
    df = pd.read_csv(p)
    if "attr_abs_mean" not in df.columns:
        raise RuntimeError(f"missing attr_abs_mean in {p}: {list(df.columns)}")
    if "genomic_pos" not in df.columns:
        df["genomic_pos"] = np.arange(len(df))
    df = df.sort_values("genomic_pos").reset_index(drop=True)
    return df


def parse_gb_pol_spans(path: Path) -> list[tuple[int, int]]:
    """Collect polymerase/replicase CDS intervals (0-based half-open), join-aware."""
    if not path.exists():
        return []
    text = path.read_text(errors="ignore")
    spans: list[tuple[int, int]] = []

    # join(...) CDS blocks
    for m in re.finditer(
        r"CDS\s+join\(([^\)]+)\)(.*?)(?=\n\s{5}\S|\nORIGIN|\nFEATURES|\Z)",
        text,
        flags=re.S,
    ):
        coords, block = m.group(1), m.group(2).lower()
        if not any(k in block for k in POL_KEYS):
            # also accept gene=orf1 / polyprotein in join without keyword sometimes
            if "polyprotein" not in block and "orf1" not in block and "pp1" not in block:
                continue
        for a, b in re.findall(r"<?(\d+)\.\.>?(\d+)", coords):
            spans.append((int(a) - 1, int(b)))

    # simple CDS
    for m in re.finditer(
        r"CDS\s+(?:complement\()?<?(\d+)\.\.>?(\d+)(.*?)(?=\n\s{5}\S|\nORIGIN|\nFEATURES|\Z)",
        text,
        flags=re.S,
    ):
        a, b, block = int(m.group(1)) - 1, int(m.group(2)), m.group(3).lower()
        if any(k in block for k in POL_KEYS) or ("polyprotein" in block and "orf1" in block):
            spans.append((a, b))

    # merge overlaps
    if not spans:
        return []
    spans = sorted(spans)
    merged = [list(spans[0])]
    for s, e in spans[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def fallback_orf_from_t32(acc: str, genome_len: int) -> list[tuple[int, int]]:
    """Infer full ORF1ab from C-terminal-third coordinates: CT = last third of ORF1ab."""
    t = pd.read_csv(T32)
    rows = t[(t["accession"] == acc) & (t["locus_id"] == "ORF1ab_C_terminal_third")]
    if not len(rows):
        # last resort: first ~70% (CoV-typical polymerase frac from Table27)
        return [(0, int(round(genome_len * 0.70)))]
    start_ct = int(rows.iloc[0]["locus_start"])
    end = int(rows.iloc[0]["locus_end"])
    # end - start_ct = L/3 => L = 3*(end-start_ct); start = end - L
    L = 3 * (end - start_ct)
    start = max(0, end - L)
    return [(start, min(genome_len, end))]


def pol_spans_for(acc: str, genome_len: int) -> list[tuple[int, int]]:
    gb = GB_CACHE / f"{acc}.gb"
    spans = parse_gb_pol_spans(gb)
    if spans:
        return spans
    return fallback_orf_from_t32(acc, genome_len)


def ct_span_for(acc: str) -> tuple[int, int] | None:
    t = pd.read_csv(T32)
    rows = t[(t["accession"] == acc) & (t["locus_id"] == "ORF1ab_C_terminal_third")]
    if not len(rows):
        return None
    return int(rows.iloc[0]["locus_start"]), int(rows.iloc[0]["locus_end"])


def mask_from_spans(n: int, spans: list[tuple[int, int]]) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    for s, e in spans:
        s2, e2 = max(0, s), min(n, e)
        if e2 > s2:
            m[s2:e2] = True
    return m


def window_edges(n: int, w: int) -> list[tuple[int, int]]:
    edges = np.linspace(0, n, w + 1, dtype=int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(w)]


def fisher_or(table: np.ndarray):
    oddsratio, p = stats.fisher_exact(table)
    a, b, c, d = table.astype(float).ravel()
    or_c = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))
    return float(oddsratio), float(p), float(or_c)


def analyze_W(W: int, mask_mode: str) -> tuple[dict, pd.DataFrame, np.ndarray]:
    per = []
    a = b = c = d = 0
    signs = []
    zs = []
    for acc in ACCESSIONS:
        df = load_track(acc)
        n = len(df)
        attr = df["attr_abs_mean"].to_numpy(dtype=float)
        if mask_mode == "orf1ab_full":
            spans = pol_spans_for(acc, n)
            mask = mask_from_spans(n, spans)
            o0, o1 = spans[0][0], spans[-1][1]
        elif mask_mode == "orf1ab_ct":
            ct = ct_span_for(acc)
            if ct is None:
                spans = pol_spans_for(acc, n)
                # approximate CT as last third of pol mask span
                o0 = spans[0][0] + 2 * (spans[-1][1] - spans[0][0]) // 3
                o1 = spans[-1][1]
            else:
                o0, o1 = ct
            mask = mask_from_spans(n, [(o0, o1)])
        else:
            raise ValueError(mask_mode)

        wins = []
        for i, (s, e) in enumerate(window_edges(n, W)):
            if e <= s:
                continue
            mean_abs = float(np.mean(attr[s:e]))
            frac = float(mask[s:e].mean()) if e > s else 0.0
            pol = frac >= 0.5
            wins.append(
                {
                    "i": i,
                    "start": s,
                    "end": e,
                    "mean_abs": mean_abs,
                    "pol_frac": frac,
                    "pol": pol,
                }
            )
        wdf = pd.DataFrame(wins)
        top = wdf.sort_values("mean_abs", ascending=False).iloc[0]
        signs.append(1 if top["pol"] else 0)
        for _, r in wdf.iterrows():
            high = r["i"] == top["i"]
            if high and r["pol"]:
                a += 1
            elif high and not r["pol"]:
                b += 1
            elif (not high) and r["pol"]:
                c += 1
            else:
                d += 1

        orf_mean = float(attr[mask].mean()) if mask.any() else np.nan
        gen_mean = float(attr.mean())
        enr = orf_mean / gen_mean if gen_mean > 0 else np.nan
        if np.isfinite(enr) and enr > 0:
            zs.append(float(np.clip(np.log(enr) / 0.25, -3, 3)))

        per.append(
            {
                "accession": acc,
                "W": W,
                "mask_mode": mask_mode,
                "genome_len": n,
                "mask_start": o0,
                "mask_end": o1,
                "mask_frac": float(mask.mean()),
                "top_window": int(top["i"]),
                "top_pol_frac": float(top["pol_frac"]),
                "top_is_pol": bool(top["pol"]),
                "top_mean_abs": float(top["mean_abs"]),
                "mask_mean_abs": orf_mean,
                "genome_mean_abs": gen_mean,
                "mask_enrichment": enr,
                "span_source": "genbank" if parse_gb_pol_spans(GB_CACHE / f"{acc}.gb") else "t32_fallback",
            }
        )

    table = np.array([[a, b], [c, d]], dtype=int)
    odds, p, or_c = fisher_or(table)
    n_top_pol = int(sum(signs))
    p0 = float(np.mean([r["mask_frac"] for r in per]))
    p0_clip = min(max(p0, 1e-6), 1 - 1e-6)
    # Primary null = genome-wide mask coverage (≈0.69 for ORF1ab); 0.5 is anti-conservative
    binom_p = float(stats.binomtest(n_top_pol, len(signs), p=p0_clip, alternative="greater").pvalue)
    sign_p_anti = float(stats.binomtest(n_top_pol, len(ACCESSIONS), p=0.5, alternative="greater").pvalue)

    if zs:
        T = float(np.sum(zs) / np.sqrt(len(zs)))
        p_stouffer = float(1 - stats.norm.cdf(T))
    else:
        T, p_stouffer = np.nan, np.nan

    summary = {
        "W": W,
        "mask_mode": mask_mode,
        "n_accessions": len(ACCESSIONS),
        "contingency_high_pol_high_non_low_pol_low_non": [a, b, c, d],
        "fisher_OR": odds,
        "fisher_OR_haldane": or_c,
        "fisher_p": p,
        "n_top_window_in_pol": n_top_pol,
        "frac_top_in_pol": n_top_pol / len(ACCESSIONS),
        "mean_mask_genome_fraction": p0,
        "sign_test_p_vs_coverage": binom_p,
        "sign_test_p_vs_0.5_anticonservative": sign_p_anti,
        "stouffer_T_log_mask_enrich": T,
        "stouffer_p": p_stouffer,
    }
    return summary, pd.DataFrame(per), table



def analyze_W_peakfrag(W: int, frag: int = 250) -> tuple[dict, pd.DataFrame, np.ndarray]:
    """Classify windows by top-frag overlap with ORF1ab (mirrors original SF design)."""
    per = []
    a = b = c = d = 0
    signs = []
    for acc in ACCESSIONS:
        df = load_track(acc)
        n = len(df)
        attr = df["attr_abs_mean"].to_numpy(dtype=float)
        spans = pol_spans_for(acc, n)
        mask = mask_from_spans(n, spans)
        wins = []
        for i, (s, e) in enumerate(window_edges(n, W)):
            if e <= s:
                continue
            mean_abs = float(np.mean(attr[s:e]))
            # top frag inside window
            L = e - s
            fl = min(frag, L)
            best_j, best_m = s, -1.0
            for j in range(s, e - fl + 1):
                m = float(np.mean(attr[j : j + fl]))
                if m > best_m:
                    best_m, best_j = m, j
            frag_frac = float(mask[best_j : best_j + fl].mean())
            pol = frag_frac >= 0.5
            wins.append({"i": i, "mean_abs": mean_abs, "pol": pol, "frag_frac": frag_frac, "frag_start": best_j})
        wdf = pd.DataFrame(wins)
        top = wdf.sort_values("mean_abs", ascending=False).iloc[0]
        signs.append(1 if top["pol"] else 0)
        for _, r in wdf.iterrows():
            high = r["i"] == top["i"]
            if high and r["pol"]:
                a += 1
            elif high and not r["pol"]:
                b += 1
            elif (not high) and r["pol"]:
                c += 1
            else:
                d += 1
        per.append({"accession": acc, "W": W, "mask_mode": "peakfrag250_orf1ab", "top_is_pol": bool(top["pol"]), "top_frag_frac": float(top["frag_frac"]), "top_window": int(top["i"])})
    table = np.array([[a, b], [c, d]], dtype=int)
    odds, p, or_c = fisher_or(table)
    n_top = int(sum(signs))
    summary = {
        "W": W,
        "mask_mode": "peakfrag250_orf1ab",
        "n_accessions": len(ACCESSIONS),
        "contingency_high_pol_high_non_low_pol_low_non": [a, b, c, d],
        "fisher_OR": odds,
        "fisher_OR_haldane": or_c,
        "fisher_p": p,
        "n_top_window_in_pol": n_top,
        "frac_top_in_pol": n_top / len(ACCESSIONS),
        # coverage ≈ ORF1ab genome fraction; do not use 0.5 null in reporting
        "mean_orf_genome_fraction_approx": 0.69,
        "sign_test_p_vs_coverage": float(
            stats.binomtest(n_top, len(ACCESSIONS), p=0.69, alternative="greater").pvalue
        ),
        "sign_test_p_vs_0.5_anticonservative": float(
            stats.binomtest(n_top, len(ACCESSIONS), p=0.5, alternative="greater").pvalue
        ),
    }
    return summary, pd.DataFrame(per), table

def main():
    sample = load_track(ACCESSIONS[0])
    print("sample", ACCESSIONS[0], list(sample.columns), "n", len(sample), flush=True)

    # sanity: GB parse coverage
    cov = []
    for acc in ACCESSIONS:
        n = len(load_track(acc))
        spans = parse_gb_pol_spans(GB_CACHE / f"{acc}.gb")
        cov.append({"accession": acc, "n_gb_spans": len(spans), "spans": spans})
    print("GB span coverage:", cov, flush=True)

    all_per = []
    summaries = []
    for mask_mode in ("orf1ab_full", "orf1ab_ct"):
        for W in (3, 5, 7, 10):
            print(f"=== {mask_mode} W={W} ===", flush=True)
            summary, per, table = analyze_W(W, mask_mode)
            summaries.append(summary)
            all_per.append(per)
            print(summary, flush=True)
            print("table", table, flush=True)

    for W in (3, 5, 7, 10):
        print(f"=== peakfrag250_orf1ab W={W} ===", flush=True)
        summary, per, table = analyze_W_peakfrag(W)
        summaries.append(summary)
        all_per.append(per)
        print(summary, flush=True)
        print("table", table, flush=True)

    pd.DataFrame(summaries).to_csv(OUT / "Table_cov_dense_window_enrichment_summary.csv", index=False)
    pd.concat(all_per, ignore_index=True).to_csv(OUT / "Table_cov_dense_window_per_accession.csv", index=False)
    meta = {
        "accessions": ACCESSIONS,
        "note": (
            "Same 11 held-out CoV accessions; denser equal partitions of dense stitched "
            "attribution tracks (attr_abs_mean). Primary: top-1 window by mean |attr| vs rest; "
            "polymerase if >=50% overlap with ORF1ab/replicase GenBank mask (join-aware; "
            "T32 C-terminal-third back-calculation fallback). Parallel mask: ORF1ab C-terminal "
            "third. Not an expanded independent organism panel; attribution column bug in prior "
            "run (genomic_pos mistaken for attr) fixed."
        ),
        "summaries": summaries,
        "gb_span_coverage": cov,
    }
    (OUT / "run_meta_cov_dense_windows_wxd0804.json").write_text(json.dumps(meta, indent=2, default=str))
    print("DONE", OUT, flush=True)


if __name__ == "__main__":
    main()
