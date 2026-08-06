#!/usr/bin/env python3
"""Smoke-check Spillover_public(_git) after packaging."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def must(p: Path, label: str) -> None:
    if not p.exists():
        raise SystemExit(f"FAIL missing {label}: {p}")


def main() -> int:
    print("ROOT =", ROOT)
    must(ROOT / "data" / "overlap_cohort.csv", "overlap")
    must(ROOT / "data" / "spillover_rankings.csv", "rankings")
    emb = list((ROOT / "embeddings").glob("embeddings_regression*.csv"))
    if not emb:
        raise SystemExit("FAIL: no regression embeddings")
    must(ROOT / "tables" / "Table32_known_loci_attribution_scores.csv", "Table32")
    must(ROOT / "code" / "analysis" / "analysis_p0p1_circularity_baselines_wxd0804" / "run_all.py", "P0/P1")
    must(ROOT / "code" / "train" / "train_regression_genome_fusion_lora.py", "train script")
    must(ROOT / "docs" / "REPRODUCIBILITY.md", "docs")
    must(ROOT / "models" / "checkpoint_manifest.json", "checkpoint manifest")

    orf = ROOT / "code" / "analysis" / "analysis_orf1ab_posperm_wxd0804"
    frag = orf / "sliding_fragments_reannotated.csv"
    script = orf / "run_orf1ab_position_perm_wxd0804.py"
    if script.is_file() and frag.is_file():
        print("=== Run ORF1ab position permutation ===")
        subprocess.run([sys.executable, str(script)], cwd=str(orf), check=True)
    else:
        print("WARN: ORF1ab S43 script/data incomplete — skipped")

    man = json.loads((ROOT / "models" / "checkpoint_manifest.json").read_text())
    print(f"checkpoints in manifest: {len(man)}")
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
