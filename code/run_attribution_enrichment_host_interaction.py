#!/usr/bin/env python3
"""Attribution enrichment in host-interaction regions (CoV subset).

Upgrades the BLAST / case-study narrative to inferential enrichment:

1) Overlap-based GenBank functional masks (entry / polymerase / accessory / structural)
2) Spike-centered per-base IG tracks: high-attr enrichment + circular-shift permutation
3) Unbiased sliding-window top fragments: placement enrichment vs random genomic draws
4) High-attr vs low-attr window contrast (Fisher / odds ratio)
5) Pooled and per-accession summaries suitable for Supplementary tables

Outputs under paper/bib_tables/ and paper/bib_figures/:
  Table27_host_interaction_region_lengths.csv
  Table28_spike_window_attr_enrichment.csv
  Table29_spike_window_permutation.csv
  Table30_sliding_fragment_placement_enrichment.csv
  Table31_high_vs_low_attr_window_classes.csv
  SupplementaryFigure_S9_host_interaction_enrichment.png
  attribution_enrichment_run_meta.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
PAPER = REPO / "paper"
TBL = PAPER / "bib_tables"
FIG = PAPER / "bib_figures"
OUT_DIR = PAPER / "analysis_host_interaction_enrichment"
EVO_DATA = Path("/home/wangxindi/evo/evo_data")
GB_CACHE = EVO_DATA / "genbank_cache_cov_subset"
SPIKE_ATTR_DIR = EVO_DATA / "ig_blast_results_cov_subset"
SLIDING_LONG = TBL / "Table19_sliding_ig_windows_long.csv"
SPIKE_SUMMARY = SPIKE_ATTR_DIR / "ig_blast_summary.csv"
FASTA = EVO_DATA / "ig_analysis_cov_subset.fasta"
PRED_CSV = REPO / "test_result/regression_genome_fusion/infer_regression_20260512_080939_per_sequence_predictions.csv"

# SARS-CoV-2 RBD (1-based inclusive) fallback when Region feature absent
SARS2_RBD_1BASED = (22517, 23185)

HOST_INTERACTION_CLASSES = (
    "entry_interface",
    "polymerase_replicase",
    "accessory_immune",
)
FOCUS_CLASSES = ("entry_interface", "polymerase_replicase", "host_interaction_broad")


@dataclass
class FeatureSpan:
    ftype: str
    start: int  # 0-based half-open
    end: int
    label: str
    gene: str = ""
    product: str = ""


def load_fasta(path: Path) -> dict[str, str]:
    """Parse FASTA; keys are accessions.

    Supports headers like:
      >GU190215 ...
      >Bat coronavirus ...|GU190215|29276nt
    """
    out: dict[str, str] = {}
    hid = None
    buf: list[str] = []

    def _accession_from_header(h: str) -> str:
        h = h[1:].strip()
        parts = [x.strip() for x in h.split("|") if x.strip()]
        for tok in parts:
            t = tok.split()[0]
            if re.search(r"\d", t) and len(t) <= 20 and not t.lower().endswith("nt"):
                return t
        return h.split()[0]

    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if hid is not None:
                    out[hid] = "".join(buf)
                hid = _accession_from_header(line)
                buf = []
            else:
                buf.append(re.sub(r"[^ACGTNacgtn]", "", line.upper()))
        if hid is not None:
            out[hid] = "".join(buf)
    return out


def fetch_genbank_text(accession: str) -> str:
    p = GB_CACHE / f"{accession}.gb"
    if not p.is_file():
        # try without version
        cands = list(GB_CACHE.glob(f"{accession.split('.')[0]}*.gb"))
        if not cands:
            raise FileNotFoundError(accession)
        p = cands[0]
    return p.read_text(encoding="utf-8", errors="replace")


def _parse_location(loc: str) -> list[tuple[int, int]]:
    loc = loc.strip().replace("\n", "").replace(" ", "")
    if loc.startswith("complement"):
        loc = loc[len("complement(") : -1]
    if loc.startswith("join"):
        loc = loc[len("join(") : -1]
    parts = []
    for chunk in loc.split(","):
        if ".." not in chunk:
            continue
        a, b = chunk.split("..", 1)
        a = int(re.sub(r"[^0-9]", "", a))
        b = int(re.sub(r"[^0-9]", "", b))
        if a and b and b >= a:
            parts.append((a - 1, b))
    return parts


def parse_genbank_features(gb_text: str) -> list[FeatureSpan]:
    spans: list[FeatureSpan] = []
    lines = gb_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"\s{5}(\S+)\s+(.+)$", line)
        if not m:
            i += 1
            continue
        ftype, loc = m.group(1), m.group(2)
        while i + 1 < len(lines) and lines[i + 1].startswith(" " * 21):
            nxt = lines[i + 1].strip()
            if nxt.startswith("/"):
                break
            loc += nxt
            i += 1
        gene = ""
        product = ""
        region = ""
        j = i + 1
        while j < len(lines) and lines[j].startswith(" " * 21):
            q = lines[j].strip()
            if q.startswith('/product="'):
                product = q.split('"')[1]
            elif q.startswith('/gene="'):
                gene = q.split('"')[1]
            elif q.startswith('/region_name="'):
                region = q.split('"')[1]
            j += 1
        label = (region or product or gene).lower()
        if ftype in {"CDS", "mat_peptide", "Region", "gene"}:
            for a0, a1 in _parse_location(loc):
                spans.append(
                    FeatureSpan(
                        ftype=ftype,
                        start=a0,
                        end=a1,
                        label=label,
                        gene=gene.lower(),
                        product=product.lower(),
                    )
                )
        i += 1
    return spans


def _tokens(span: FeatureSpan) -> set[str]:
    return {t for t in (span.gene, span.product, span.label) if t}


def _is_entry(span: FeatureSpan) -> bool:
    toks = _tokens(span)
    if toks & {"s", "he", "spike"}:
        return True
    blob = " ".join(toks)
    keys = (
        "spike",
        "surface glycoprotein",
        "receptor bind",
        "rbd",
        "hemagglutinin-esterase",
        "s glycoprotein",
    )
    return any(k in blob for k in keys)


def _is_rbd(span: FeatureSpan, accession: str) -> bool:
    blob = " ".join(_tokens(span))
    return "rbd" in blob or "receptor bind" in blob


def _is_polymerase(span: FeatureSpan) -> bool:
    toks = _tokens(span)
    if toks & {"orf1ab", "orf1a", "orf1b", "1a", "1ab", "1b"}:
        return True
    blob = " ".join(toks)
    keys = (
        "orf1",
        "polyprotein",
        "replicase",
        "rna-dependent rna polymerase",
        "rdrp",
        "helicase",
        "nsp",
        "methyltransferase",
        "exonuclease",
        "endorna",
    )
    return any(k in blob for k in keys)


def _is_accessory(span: FeatureSpan) -> bool:
    toks = _tokens(span)
    for t in toks:
        if re.fullmatch(r"(orf)?([3-9]|10)a?", t) or re.fullmatch(r"ns\d+", t):
            return True
    blob = " ".join(toks)
    keys = ("orf3", "orf6", "orf7", "orf8", "orf10", "accessory", "interferon", "immune", "ns3", "ns7")
    return any(k in blob for k in keys)


def _is_structural_other(span: FeatureSpan) -> bool:
    toks = _tokens(span)
    if toks & {"e", "m", "n"}:
        return True
    blob = " ".join(toks)
    keys = ("envelope", "membrane", "nucleocapsid", "nucleoprotein")
    return any(k in blob for k in keys)


def build_class_mask(n: int, spans: list[FeatureSpan], accession: str) -> dict[str, np.ndarray]:
    """Priority: RBD ⊂ entry; then polymerase; accessory; structural; other."""
    masks = {
        "entry_interface": np.zeros(n, dtype=bool),
        "rbd": np.zeros(n, dtype=bool),
        "polymerase_replicase": np.zeros(n, dtype=bool),
        "accessory_immune": np.zeros(n, dtype=bool),
        "structural_other": np.zeros(n, dtype=bool),
    }
    for s in spans:
        a0, a1 = max(0, s.start), min(n, s.end)
        if a0 >= a1:
            continue
        if _is_rbd(s, accession):
            masks["rbd"][a0:a1] = True
            masks["entry_interface"][a0:a1] = True
        elif _is_entry(s):
            masks["entry_interface"][a0:a1] = True
        elif _is_polymerase(s):
            masks["polymerase_replicase"][a0:a1] = True
        elif _is_accessory(s):
            masks["accessory_immune"][a0:a1] = True
        elif _is_structural_other(s):
            masks["structural_other"][a0:a1] = True

    # SARS-CoV-2 RBD fallback
    if accession.upper().startswith("ON") or accession.upper().startswith("NC_045512"):
        r0, r1 = SARS2_RBD_1BASED
        a0, a1 = max(0, r0 - 1), min(n, r1)
        if a0 < a1:
            masks["rbd"][a0:a1] = True
            masks["entry_interface"][a0:a1] = True

    masks["host_interaction_broad"] = (
        masks["entry_interface"] | masks["polymerase_replicase"] | masks["accessory_immune"]
    )
    masks["other"] = ~(
        masks["entry_interface"]
        | masks["polymerase_replicase"]
        | masks["accessory_immune"]
        | masks["structural_other"]
    )
    return masks


def classify_interval_by_overlap(
    start: int, end: int, masks: dict[str, np.ndarray]
) -> str:
    """Assign class by maximum overlap; ties broken by priority."""
    a0, a1 = max(0, start), min(len(next(iter(masks.values()))), end)
    if a0 >= a1:
        return "other"
    priority = [
        "rbd",
        "entry_interface",
        "polymerase_replicase",
        "accessory_immune",
        "structural_other",
        "other",
    ]
    best, best_ov = "other", -1
    for cls in priority:
        if cls not in masks:
            continue
        ov = int(masks[cls][a0:a1].sum())
        if ov > best_ov:
            best, best_ov = cls, ov
    if best == "rbd":
        return "entry_interface"  # roll RBD into entry for fragment-level tables
    return best if best_ov > 0 else "other"


def enrichment_ratio(attr: np.ndarray, mask: np.ndarray) -> dict:
    attr = np.asarray(attr, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if attr.size == 0 or mask.size != attr.size:
        return {"enrichment": np.nan, "mean_in": np.nan, "mean_out": np.nan, "frac_len": np.nan}
    if mask.any() and (~mask).any():
        mean_in = float(attr[mask].mean())
        mean_out = float(attr[~mask].mean())
        enr = mean_in / mean_out if mean_out > 0 else np.inf
    elif mask.any():
        mean_in = float(attr[mask].mean())
        mean_out = np.nan
        enr = np.nan
    else:
        mean_in = np.nan
        mean_out = float(attr.mean()) if attr.size else np.nan
        enr = np.nan
    # mass enrichment: fraction of |attr| mass in mask / length fraction
    mass = np.abs(attr)
    total = float(mass.sum()) + 1e-12
    mass_frac = float(mass[mask].sum()) / total
    len_frac = float(mask.mean())
    mass_enr = mass_frac / (len_frac + 1e-12)
    return {
        "enrichment": enr,
        "mean_in": mean_in,
        "mean_out": mean_out,
        "frac_len": len_frac,
        "mass_frac": mass_frac,
        "mass_enrichment": mass_enr,
    }


def circular_shift_permutation(
    attr: np.ndarray,
    mask: np.ndarray,
    n_perm: int,
    seed: int,
    stat: str = "mass_enrichment",
) -> tuple[float, float, np.ndarray]:
    """Permute by circularly shifting attribution along the window/genome."""
    rng = np.random.default_rng(seed)
    obs = enrichment_ratio(attr, mask)[stat]
    null = np.empty(n_perm, dtype=float)
    n = len(attr)
    for i in range(n_perm):
        shift = int(rng.integers(0, n))
        null[i] = enrichment_ratio(np.roll(attr, shift), mask)[stat]
    # one-sided: enrichment > null
    if np.isnan(obs):
        p = np.nan
    else:
        p = (1.0 + float(np.sum(null >= obs))) / (n_perm + 1.0)
    return float(obs), float(p), null


def top_quantile_overlap_test(
    attr: np.ndarray,
    mask: np.ndarray,
    q: float,
    n_perm: int,
    seed: int,
) -> dict:
    """Are top-q attribution positions enriched inside mask? Hypergeometric + shift null."""
    n = len(attr)
    k = max(1, int(np.ceil(q * n)))
    order = np.argsort(-np.abs(attr))
    top = np.zeros(n, dtype=bool)
    top[order[:k]] = True
    a = int((top & mask).sum())  # top in mask
    K = int(mask.sum())
    # hypergeometric P(X >= a)
    p_hg = float(stats.hypergeom.sf(a - 1, n, K, k)) if K > 0 else np.nan
    obs_frac = a / k
    expected = K / n if n else np.nan
    # permutation of top labels via attr shift
    _, p_shift, _ = circular_shift_permutation(
        np.abs(attr), mask, n_perm=n_perm, seed=seed, stat="mass_enrichment"
    )
    # also permute: shift then recompute top overlap frac
    rng = np.random.default_rng(seed + 7)
    null_frac = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        shifted = np.roll(np.abs(attr), int(rng.integers(0, n)))
        ord2 = np.argsort(-shifted)[:k]
        null_frac[i] = float(mask[ord2].mean())
    p_top = (1.0 + float(np.sum(null_frac >= obs_frac))) / (n_perm + 1.0)
    return {
        "top_n": k,
        "top_in_mask": a,
        "mask_n": K,
        "obs_top_frac_in_mask": obs_frac,
        "expected_frac": expected,
        "enrichment_vs_length": obs_frac / (expected + 1e-12),
        "p_hypergeom": p_hg,
        "p_shift_top_frac": p_top,
        "p_shift_mass_enrichment": p_shift,
    }


def random_fragment_placement_test(
    genomes: dict[str, dict],
    fragments: pd.DataFrame,
    class_name: str,
    n_perm: int,
    seed: int,
    frag_len: int = 250,
) -> dict:
    """Test whether observed top fragments land in class more than random placements."""
    rng = np.random.default_rng(seed)
    obs_hits = 0
    total = 0
    # per-row observed
    for _, r in fragments.iterrows():
        acc = str(r["accession"])
        if acc not in genomes:
            continue
        masks = genomes[acc]["masks"]
        n = genomes[acc]["n"]
        fs, fe = int(r["top_fragment_start"]), int(r["top_fragment_end"])
        cls = classify_interval_by_overlap(fs, fe, masks)
        # map rbd->entry already handled
        hit = cls == class_name or (
            class_name == "host_interaction_broad"
            and cls in {"entry_interface", "polymerase_replicase", "accessory_immune"}
        )
        # for host_interaction_broad use mask overlap
        if class_name in masks:
            mid_cls = classify_interval_by_overlap(fs, fe, masks)
            if class_name == "host_interaction_broad":
                hit = bool(masks["host_interaction_broad"][fs:fe].mean() >= 0.5)
            else:
                hit = mid_cls == class_name or (
                    class_name == "entry_interface" and mid_cls == "entry_interface"
                )
                if class_name in masks:
                    hit = bool(masks[class_name][fs:fe].mean() >= 0.5)
        obs_hits += int(hit)
        total += 1
    if total == 0:
        return {"n_fragments": 0, "obs_hits": 0, "obs_frac": np.nan, "p_value": np.nan}

    null_fracs = np.empty(n_perm, dtype=float)
    acc_list = [str(a) for a in fragments["accession"] if str(a) in genomes]
    for i in range(n_perm):
        hits = 0
        for acc in acc_list:
            n = genomes[acc]["n"]
            masks = genomes[acc]["masks"]
            if n <= frag_len:
                start = 0
            else:
                start = int(rng.integers(0, n - frag_len + 1))
            end = start + frag_len
            if class_name not in masks:
                hits += 0
            else:
                hits += int(masks[class_name][start:end].mean() >= 0.5)
        null_fracs[i] = hits / total
    obs_frac = obs_hits / total
    p = (1.0 + float(np.sum(null_fracs >= obs_frac))) / (n_perm + 1.0)
    return {
        "class": class_name,
        "n_fragments": total,
        "obs_hits": obs_hits,
        "obs_frac": obs_frac,
        "null_mean": float(null_fracs.mean()),
        "null_p95": float(np.quantile(null_fracs, 0.95)),
        "enrichment_vs_null": obs_frac / (float(null_fracs.mean()) + 1e-12),
        "p_value": p,
    }


def high_vs_low_window_fisher(sliding: pd.DataFrame, genomes: dict, class_name: str) -> dict:
    """Within each accession, compare top half vs bottom half windows by attr_sum."""
    # pooled contingency
    high_in = low_in = high_out = low_out = 0
    for acc, g in sliding.groupby("accession"):
        if acc not in genomes:
            continue
        g = g.sort_values("window_attr_sum", ascending=False)
        mid = max(1, len(g) // 2)
        high = g.iloc[:mid]
        low = g.iloc[mid:]
        masks = genomes[acc]["masks"]
        for _, r in high.iterrows():
            fs, fe = int(r["top_fragment_start"]), int(r["top_fragment_end"])
            hit = bool(masks[class_name][fs:fe].mean() >= 0.5) if class_name in masks else False
            if hit:
                high_in += 1
            else:
                high_out += 1
        for _, r in low.iterrows():
            fs, fe = int(r["top_fragment_start"]), int(r["top_fragment_end"])
            hit = bool(masks[class_name][fs:fe].mean() >= 0.5) if class_name in masks else False
            if hit:
                low_in += 1
            else:
                low_out += 1
    table = np.array([[high_in, high_out], [low_in, low_out]])
    if table.sum() == 0 or (table == 0).all():
        oddsratio, p = np.nan, np.nan
    else:
        oddsratio, p = stats.fisher_exact(table)
    return {
        "class": class_name,
        "high_in": high_in,
        "high_out": high_out,
        "low_in": low_in,
        "low_out": low_out,
        "odds_ratio": float(oddsratio) if oddsratio == oddsratio else np.nan,
        "p_fisher": float(p) if p == p else np.nan,
    }


def load_spike_attr(accession: str) -> pd.DataFrame:
    p = SPIKE_ATTR_DIR / f"ig_attr_{accession}.csv"
    return pd.read_csv(p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top-q", type=float, default=0.10)
    args = ap.parse_args()

    TBL.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    seqs = load_fasta(FASTA)
    summary = pd.read_csv(SPIKE_SUMMARY)
    sliding = pd.read_csv(SLIDING_LONG)

    genomes: dict[str, dict] = {}
    length_rows = []
    for acc, seq in seqs.items():
        n = len(seq)
        try:
            spans = parse_genbank_features(fetch_genbank_text(acc))
        except Exception as e:
            print(f"[warn] GenBank missing for {acc}: {e}", flush=True)
            spans = []
        masks = build_class_mask(n, spans, acc)
        genomes[acc] = {"n": n, "seq": seq, "spans": spans, "masks": masks}
        row = {"accession": acc, "genome_len": n, "n_features": len(spans)}
        for cls, m in masks.items():
            row[f"len_{cls}"] = int(m.sum())
            row[f"frac_{cls}"] = float(m.mean())
        length_rows.append(row)
    length_df = pd.DataFrame(length_rows)
    length_path = TBL / "Table27_host_interaction_region_lengths.csv"
    length_df.to_csv(length_path, index=False)
    print(f"Wrote {length_path}", flush=True)

    # ---- Spike-centered window enrichment (with explicit prior caveat) ----
    spike_rows = []
    perm_rows = []
    for _, r in summary.iterrows():
        acc = str(r["accession"])
        if acc not in genomes:
            continue
        attr_df = load_spike_attr(acc)
        gpos = attr_df["genomic_pos"].to_numpy(dtype=int)
        scores = attr_df["attr_score"].to_numpy(dtype=float)
        w0, w1 = int(gpos.min()), int(gpos.max()) + 1
        n = genomes[acc]["n"]
        # build window-local attr vector aligned to genomic coords
        local_attr = np.zeros(w1 - w0, dtype=float)
        for p, s in zip(gpos, scores):
            if w0 <= p < w1:
                local_attr[p - w0] = s
        for cls in FOCUS_CLASSES + ("rbd", "structural_other", "other"):
            full_mask = genomes[acc]["masks"].get(cls)
            if full_mask is None:
                continue
            local_mask = full_mask[w0:w1]
            if local_mask.size != local_attr.size:
                continue
            enr = enrichment_ratio(local_attr, local_mask)
            top = top_quantile_overlap_test(
                local_attr, local_mask, q=args.top_q, n_perm=args.n_perm, seed=args.seed
            )
            obs, p_mass, _ = circular_shift_permutation(
                local_attr, local_mask, n_perm=args.n_perm, seed=args.seed, stat="mass_enrichment"
            )
            spike_rows.append(
                {
                    "accession": acc,
                    "organism": r.get("lineage", ""),
                    "analysis": "spike_centered_window",
                    "window_start": w0,
                    "window_end": w1,
                    "class": cls,
                    "frac_len_in_window": enr["frac_len"],
                    "mean_attr_in": enr["mean_in"],
                    "mean_attr_out": enr["mean_out"],
                    "mean_enrichment": enr["enrichment"],
                    "mass_frac": enr["mass_frac"],
                    "mass_enrichment": enr["mass_enrichment"],
                    "top_q": args.top_q,
                    **{f"top_{k}": v for k, v in top.items()},
                    "p_circular_shift_mass": p_mass,
                    "note": "IG window was spike-prioritized; interpret within-window only",
                }
            )
            perm_rows.append(
                {
                    "accession": acc,
                    "class": cls,
                    "stat": "mass_enrichment",
                    "observed": obs,
                    "p_circular_shift": p_mass,
                    "n_perm": args.n_perm,
                }
            )

    spike_df = pd.DataFrame(spike_rows)
    spike_path = TBL / "Table28_spike_window_attr_enrichment.csv"
    spike_df.to_csv(spike_path, index=False)
    perm_df = pd.DataFrame(perm_rows)
    perm_path = TBL / "Table29_spike_window_permutation.csv"
    perm_df.to_csv(perm_path, index=False)
    print(f"Wrote {spike_path}", flush=True)
    print(f"Wrote {perm_path}", flush=True)

    # ---- Re-annotate sliding fragments with overlap classifier ----
    ann_rows = []
    for _, r in sliding.iterrows():
        acc = str(r["accession"])
        if acc not in genomes:
            continue
        fs, fe = int(r["top_fragment_start"]), int(r["top_fragment_end"])
        cls = classify_interval_by_overlap(fs, fe, genomes[acc]["masks"])
        ann_rows.append(
            {
                **r.to_dict(),
                "top_fragment_class_overlap": cls,
                "frac_entry": float(
                    genomes[acc]["masks"]["entry_interface"][fs:fe].mean()
                ),
                "frac_polymerase": float(
                    genomes[acc]["masks"]["polymerase_replicase"][fs:fe].mean()
                ),
                "frac_host_interaction_broad": float(
                    genomes[acc]["masks"]["host_interaction_broad"][fs:fe].mean()
                ),
            }
        )
    sliding_ann = pd.DataFrame(ann_rows)
    sliding_ann_path = OUT_DIR / "sliding_fragments_reannotated.csv"
    sliding_ann.to_csv(sliding_ann_path, index=False)

    # ---- Random placement enrichment (unbiased sliding top fragments) ----
    place_rows = []
    for cls in FOCUS_CLASSES:
        res = random_fragment_placement_test(
            genomes, sliding_ann, cls, n_perm=args.n_perm, seed=args.seed, frag_len=250
        )
        res["analysis"] = "unbiased_sliding_top250_placement"
        place_rows.append(res)
        # also restrict to best window per accession
        best = sliding_ann.sort_values("window_attr_sum", ascending=False).groupby("accession").head(1)
        res_b = random_fragment_placement_test(
            genomes, best, cls, n_perm=args.n_perm, seed=args.seed + 1, frag_len=250
        )
        res_b["analysis"] = "unbiased_sliding_best_window_top250_placement"
        place_rows.append(res_b)
    place_df = pd.DataFrame(place_rows)
    place_path = TBL / "Table30_sliding_fragment_placement_enrichment.csv"
    place_df.to_csv(place_path, index=False)
    print(f"Wrote {place_path}", flush=True)

    # ---- High vs low attr windows ----
    hv_rows = []
    for cls in FOCUS_CLASSES:
        hv_rows.append(high_vs_low_window_fisher(sliding_ann, genomes, cls))
    hv_df = pd.DataFrame(hv_rows)
    hv_path = TBL / "Table31_high_vs_low_attr_window_classes.csv"
    hv_df.to_csv(hv_path, index=False)
    print(f"Wrote {hv_path}", flush=True)

    # ---- Pooled spike-window summary across accessions ----
    pooled = []
    for cls, g in spike_df.groupby("class"):
        pooled.append(
            {
                "class": cls,
                "n_accessions": len(g),
                "median_mass_enrichment": float(g["mass_enrichment"].median()),
                "mean_mass_enrichment": float(g["mass_enrichment"].mean()),
                "median_p_circular_shift": float(g["p_circular_shift_mass"].median()),
                "n_sig_p005": int((g["p_circular_shift_mass"] < 0.05).sum()),
                "median_top_enrichment_vs_length": float(g["top_enrichment_vs_length"].median()),
                "n_top_sig_hypergeom_005": int((g["top_p_hypergeom"] < 0.05).sum()),
            }
        )
    pooled_df = pd.DataFrame(pooled)
    pooled_path = OUT_DIR / "spike_window_enrichment_pooled.csv"
    pooled_df.to_csv(pooled_path, index=False)

    # ---- Figure ----
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))

    # A: region length fractions
    frac_cols = [
        "frac_entry_interface",
        "frac_polymerase_replicase",
        "frac_accessory_immune",
        "frac_structural_other",
    ]
    labels = ["entry", "polymerase", "accessory", "structural"]
    means = [float(length_df[c].mean()) for c in frac_cols]
    axes[0].bar(labels, means, color=["#2c7fb8", "#7fcdbb", "#fdae61", "#969696"])
    axes[0].set_ylabel("Mean genome fraction")
    axes[0].set_title("A. CoV functional region lengths")
    axes[0].tick_params(axis="x", rotation=25)

    # B: placement enrichment
    sub = place_df[place_df["analysis"] == "unbiased_sliding_top250_placement"]
    axes[1].barh(sub["class"], sub["obs_frac"], color="#2c7fb8", label="observed", alpha=0.85)
    axes[1].barh(sub["class"], sub["null_mean"], color="#cccccc", label="random null", alpha=0.85)
    for _, row in sub.iterrows():
        axes[1].text(
            max(row["obs_frac"], row["null_mean"]) + 0.01,
            row["class"],
            f"p={row['p_value']:.3f}",
            va="center",
            fontsize=8,
        )
    axes[1].set_xlabel("Fraction of top-250nt fragments")
    axes[1].set_title("B. Sliding IG placement vs random")
    axes[1].legend(fontsize=7, loc="lower right")

    # C: within spike-window mass enrichment for entry/RBD
    sub2 = spike_df[spike_df["class"].isin(["entry_interface", "rbd", "polymerase_replicase"])]
    if len(sub2):
        classes = ["entry_interface", "rbd", "polymerase_replicase"]
        data = [sub2.loc[sub2["class"] == c, "mass_enrichment"].dropna().values for c in classes]
        axes[2].boxplot(data, labels=["entry", "RBD", "pol"])
        axes[2].axhline(1.0, color="grey", ls="--", lw=0.8)
        axes[2].set_ylabel("Attribution mass enrichment")
        axes[2].set_title("C. Spike-prior window (caveat)")
    fig.tight_layout()
    fig_path = FIG / "SupplementaryFigure_S9_host_interaction_enrichment.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {fig_path}", flush=True)

    meta = {
        "finished": datetime.now(timezone.utc).isoformat(),
        "n_perm": args.n_perm,
        "seed": args.seed,
        "top_q": args.top_q,
        "n_genomes": len(genomes),
        "outputs": {
            "Table27": str(length_path),
            "Table28": str(spike_path),
            "Table29": str(perm_path),
            "Table30": str(place_path),
            "Table31": str(hv_path),
            "Figure_S9": str(fig_path),
            "sliding_reannotated": str(sliding_ann_path),
            "pooled": str(pooled_path),
        },
        "placement_summary": place_df.to_dict(orient="records"),
        "high_vs_low_summary": hv_df.to_dict(orient="records"),
        "pooled_spike_window": pooled_df.to_dict(orient="records"),
        "interpretation_notes": [
            "Spike-centered IG windows were selected with a spike prior; Table28/29 are within-window only.",
            "Table30/31 use unbiased sliding-window IG (Table19) and are the primary genome-wide enrichment tests.",
            "host_interaction_broad = entry_interface ∪ polymerase_replicase ∪ accessory_immune.",
        ],
    }
    meta_path = PAPER / "attribution_enrichment_run_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    # also copy under analysis dir
    (OUT_DIR / "attribution_enrichment_run_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(f"Wrote {meta_path}", flush=True)
    print("\n=== Placement enrichment (primary) ===", flush=True)
    print(place_df.to_string(index=False), flush=True)
    print("\n=== High vs low windows ===", flush=True)
    print(hv_df.to_string(index=False), flush=True)
    print("\n=== Pooled spike-window ===", flush=True)
    print(pooled_df.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
