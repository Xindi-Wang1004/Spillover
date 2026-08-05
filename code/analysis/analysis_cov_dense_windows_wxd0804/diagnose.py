#!/usr/bin/env python3
"""Diagnose ORF spans vs peak positions; reproduce OR=36 from SF table."""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ACCESSIONS = [
    "EF065509", "GU190215", "KF268336", "KF268337", "MF370205", "MG762674",
    "MT549854", "MW218395", "NC_048211", "ON648240", "OQ297728",
]
TRACK = Path("/home/wangxindi/evo/evo_data/ig_sliding_dense/genome_tracks")
GB = Path("/home/wangxindi/evo/evo_data/genbank_cache_cov_subset")
SF = Path("/home/wangxindi/evo-main/paper/analysis_host_interaction_enrichment/sliding_fragments_reannotated.csv")
T32 = Path("/home/wangxindi/evo-main/paper/bib_tables/Table32_known_loci_attribution_scores.csv")


def parse_orf1ab(path: Path):
    text = path.read_text(errors="ignore")
    spans = []
    for m in re.finditer(r"CDS\s+(?:complement\()?<?(\d+)\.\.>?(\d+)", text):
        a, b = int(m.group(1)) - 1, int(m.group(2))
        block = text[m.start() : m.start() + 900].lower()
        if any(k in block for k in ("orf1ab", "replicase", "polyprotein 1ab", "orf1a", "1ab polyprotein", "orf1ab polyprotein")):
            spans.append((a, b, b - a, block[:120].replace("\n", " ")))
    spans.sort(key=lambda x: -x[2])
    return spans[:3]


# reproduce OR=36
sf = pd.read_csv(SF)
print("SF accessions", sorted(sf.accession.unique()), "n", sf.accession.nunique())
# top1 vs rest by window_attr_sum within accession
rows = []
a = b = c = d = 0
for acc, g in sf.groupby("accession"):
    g = g.sort_values("window_attr_sum", ascending=False)
    top = g.iloc[0]
    for i, r in g.iterrows():
        high = r.name == top.name or r.window_index == top.window_index
        # use class overlap
        pol = str(r.top_fragment_class_overlap) == "polymerase_replicase"
        if high and pol:
            a += 1
        elif high and not pol:
            b += 1
        elif (not high) and pol:
            c += 1
        else:
            d += 1
    rows.append({"accession": acc, "top_class": top.top_fragment_class_overlap, "top_annot": top.top_fragment_annotation})
print(pd.DataFrame(rows))
table = np.array([[a, b], [c, d]])
print("contingency", table, "fisher", stats.fisher_exact(table))

# peaks vs ORF1ab
t32 = pd.read_csv(T32)
print("\n=== peaks vs spans ===")
for acc in ACCESSIONS:
    df = pd.read_csv(TRACK / f"{acc}_stitched_attr.csv")
    peak = int(df.loc[df.attr_abs_mean.idxmax(), "genomic_pos"])
    gb = GB / f"{acc}.gb"
    spans = parse_orf1ab(gb) if gb.exists() else []
    ct = t32[(t32.accession == acc) & (t32.locus_id == "ORF1ab_C_terminal_third")]
    ct_s = int(ct.iloc[0].locus_start) if len(ct) else None
    ct_e = int(ct.iloc[0].locus_end) if len(ct) else None
    o0, o1 = (spans[0][0], spans[0][1]) if spans else (None, None)
    in_orf = (o0 is not None and o0 <= peak < o1)
    in_ct = (ct_s is not None and ct_s <= peak < ct_e)
    # top window among W=3
    n = len(df)
    edges = np.linspace(0, n, 4, dtype=int)
    means = [df.attr_abs_mean.iloc[edges[i]:edges[i+1]].mean() for i in range(3)]
    top_w = int(np.argmax(means))
    print(f"{acc}: peak={peak} orf={o0}-{o1} in_orf={in_orf} ct={ct_s}-{ct_e} in_ct={in_ct} W3_top={top_w} means={[round(m,4) for m in means]} nspans={len(spans)}")
    if spans:
        print("   top span note:", spans[0][3][:80])
