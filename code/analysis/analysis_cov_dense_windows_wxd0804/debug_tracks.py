#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import glob

paths = [
    "/home/wangxindi/evo/evo_data/ig_sliding_dense/genome_tracks",
    "/home/wangxindi/evo-main/paper/ig_sliding_dense/genome_tracks",
]
for d in paths:
    p = Path(d)
    print("DIR", d, "exists", p.exists())
    if not p.exists():
        continue
    fs = sorted(p.glob("*_stitched_attr.csv"))
    print("  n tracks", len(fs))
    for f in fs[:2]:
        df = pd.read_csv(f)
        print(" ", f.name, "cols", list(df.columns), "n", len(df))
        print("  pos", df["genomic_pos"].min(), df["genomic_pos"].max(), "unique", df["genomic_pos"].nunique())
        print("  step median", df["genomic_pos"].diff().median())
        print(df.nlargest(3, "attr_abs_mean")[["genomic_pos", "attr_abs_mean"]].to_string(index=False))

t32 = Path("/home/wangxindi/evo-main/paper/bib_tables/Table32_known_loci_attribution_scores.csv")
print("T32", t32.exists())
if t32.exists():
    t = pd.read_csv(t32)
    print(t.columns.tolist())
    sub = t[t["locus_id"] == "ORF1ab_C_terminal_third"]
    print("ORF1ab CT rows", len(sub), "attr_source", sub["attr_source"].value_counts().to_dict() if "attr_source" in sub else None)
    print(sub[["accession", "locus_start", "locus_end", "attr_source"]].head(15).to_string(index=False))

gb = Path("/home/wangxindi/evo/evo_data/genbank_cache_cov_subset")
print("GB", gb.exists(), "n", len(list(gb.glob("*"))) if gb.exists() else 0)
if gb.exists():
    print(list(gb.glob("*"))[:10])
