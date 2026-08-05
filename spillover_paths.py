"""Path constants for Spillover_public release (server-built)."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("SPILLOVER_ROOT", Path(__file__).resolve().parent))

DATA = ROOT / "data"
EMB = ROOT / "embeddings"
TABLES = ROOT / "tables"
CODE = ROOT / "code"
MODELS = ROOT / "models"
DOCS = ROOT / "docs"

OVERLAP = DATA / "overlap_cohort.csv"
RANKINGS = DATA / "spillover_rankings.csv"
SLIDING_FRAGMENTS = DATA / "sliding_fragments_reannotated.csv"

def latest_emb(prefix: str) -> Path:
    cands = sorted(EMB.glob(f"{prefix}*.csv"))
    if not cands:
        raise FileNotFoundError(f"No embedding: {prefix}* in {EMB}")
    return cands[-1]

EMB_REG = latest_emb("embeddings_regression")
