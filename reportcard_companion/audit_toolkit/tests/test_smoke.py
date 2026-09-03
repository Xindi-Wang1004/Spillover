#!/usr/bin/env python3
"""Smoke test + toy data for genome-ml-reportcard."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TOY = HERE / "toy_data"
TOY.mkdir(exist_ok=True)

rng = np.random.default_rng(42)
n_groups, gpg, n_feat = 4, 5, 16
groups = np.repeat(np.arange(n_groups), gpg)
proto = rng.normal(size=(n_groups, n_feat))
X = np.vstack([proto[g] + 0.1 * rng.normal(size=n_feat) for g in groups])
y = groups.astype(float)  # group-constant
df = pd.DataFrame(
    {
        "sequence_id": [f"g{i}" for i in range(len(y))],
        "group": [f"sp{g}" for g in groups],
        "label": y,
    }
)
df.to_csv(TOY / "manifest.tsv", sep="\t", index=False)
np.save(TOY / "X.npy", X)

sys.path.insert(0, str(HERE.parent))
from genome_ml_reportcard.cli import main
from genome_ml_reportcard.geometry import geometry_report

out = TOY / "report.json"
rc = main(
    [
        "--table",
        str(TOY / "manifest.tsv"),
        "--features",
        str(TOY / "X.npy"),
        "--out",
        str(out),
    ]
)
assert rc == 0
rep = json.loads(out.read_text())
delta = rep["probe"]["delta"]
assert delta > 0.3, delta
assert "geometry" in rep
assert rep["geometry"]["random_cv_shared_block_fraction"] > 0.5
assert (TOY / "report.md").is_file()

# geometry helper direct check
g = geometry_report(y, df["group"].to_numpy(), df["group"].to_numpy())
assert g["within_block_homogeneity"] > 0.99

print("SMOKE_OK", delta, g["random_cv_shared_block_fraction"])
