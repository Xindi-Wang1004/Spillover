#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import os

p = Path("/home/wangxindi/evo-main/paper/bib_tables/Table19_sliding_ig_windows_long.csv")
df = pd.read_csv(p)
print("T19 cols", df.columns.tolist())
print(df.head(2).to_string())
print("n", len(df), "acc", df.get("accession", pd.Series()).nunique())
for c in df.columns:
    if any(k in c.lower() for k in ("class", "annot", "product", "gene", "blast")):
        print("col", c, "sample", df[c].dropna().astype(str).head(3).tolist())
        if df[c].dtype == object or str(df[c].dtype) == "category":
            print(df[c].value_counts().head(8))

sf = Path("/home/wangxindi/evo-main/paper/analysis_host_interaction_enrichment/sliding_fragments_reannotated.csv")
s = pd.read_csv(sf)
print("\nSF cols", s.columns.tolist())
print(s.head(2).to_string())
print("SF n", len(s), "acc", s.get("accession", pd.Series()).nunique())
for c in s.columns:
    if any(k in c.lower() for k in ("class", "annot", "product", "gene", "blast", "start", "end", "attr")):
        print("SFcol", c)

blast = Path("/home/wangxindi/evo/evo_data/ig_blast_results_cov_subset")
print("\nblast exists", blast.exists())
if blast.exists():
    print(os.listdir(blast)[:30])

# also dump_gb products
gb = Path("/home/wangxindi/evo/evo_data/genbank_cache_cov_subset/EF065509.gb")
text = gb.read_text(errors="ignore")
print("GB len", len(text), "has orf1ab", "orf1ab" in text.lower())
