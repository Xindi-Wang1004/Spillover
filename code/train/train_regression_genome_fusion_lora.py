#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-genome late fusion: Evo+LoRA encodes each 8192 (or --max_seq_len) window; window embeddings
are mean-pooled across windows, then a small MLP predicts one regression score (same label as
full-genome sliding training). Single-GPU only (no FSDP).

Sliding windows: default --window_stride 4096 with --max_seq_len 8192 (about half window overlap).
When --window_stride<=0, stride is derived from --window-overlap-fraction (10%–50% of window).

Launch (from evo-main root):
  bash train_fullgenome_notrunction/scripts/launch_genome_fusion_single_gpu_idle.sh [args...]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from peft import LoraConfig
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from stripedhyena.tokenizer import CharLevelTokenizer
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reuse HF cache + Hyena precompute from sibling script (no package __init__ required).
_tr_spec = importlib.util.spec_from_file_location(
    "_tr_fusion_ref",
    Path(__file__).resolve().parent / "train_regression_fullgenome_notrunc_lora.py",
)
_tr = importlib.util.module_from_spec(_tr_spec)
assert _tr_spec.loader is not None
_tr_spec.loader.exec_module(_tr)

_tr._configure_hf_cache_and_proxy_defaults()
# nohup/无外网节点上常见 HF_HUB_OFFLINE=0 覆盖 setdefault，导致仍去拉 HF；默认强制离线走本地 cache。
if os.environ.get("EVO_HF_ALLOW_ONLINE", "").strip().lower() not in ("1", "true", "yes"):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

_hf_home = os.environ.get("HF_HOME") or "/home/wangxindi/hf-cache"
os.environ["HF_HOME"] = _hf_home
if os.environ.get("EVO_HF_TRUST_PARENT_CACHE_PATHS", "").strip().lower() not in ("1", "true", "yes"):
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(_hf_home, "hub")
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(_hf_home, "hub")

from data_processing_multitask_strain import prepare_multitask_data
from model_ablation_single_task import EvoRegressorOnly
from train_split_leak_safe import prepare_splits_leak_safe

from _datasets_notrunc import _sliding_window_starts


def _stripedhyena_for_precompute(model: nn.Module) -> nn.Module | None:
    return _tr._stripedhyena_for_precompute(model)


def _precompute_hyena_filters_compat(striped: nn.Module, L: int, device: torch.device, rank: int) -> None:
    return _tr._precompute_hyena_filters_compat(striped, L, device, rank)


def _infer_model_max_seq_len(base_model_path: str) -> int | None:
    return _tr._infer_model_max_seq_len(base_model_path)


@dataclass
class GenomeRecord:
    name: str
    windows: list[torch.Tensor]
    target: float


def _build_windows_for_tokens(toks: list[int], max_seq_len: int, stride: int) -> list[torch.Tensor]:
    L = len(toks)
    if L == 0:
        return []
    t = torch.tensor(toks, dtype=torch.long)
    if L <= max_seq_len:
        return [t]
    s_eff = stride if stride > 0 else max_seq_len
    starts = _sliding_window_starts(L, max_seq_len, s_eff)
    return [t[s : s + max_seq_len].clone() for s in starts]


def _subsample_windows(windows: list[torch.Tensor], max_windows: int) -> list[torch.Tensor]:
    if max_windows <= 0 or len(windows) <= max_windows:
        return windows
    idx = torch.linspace(0, len(windows) - 1, max_windows).long().tolist()
    return [windows[i] for i in idx]


class GlobalFusionHead(nn.Module):
    """Mean over window embeddings [K, D] -> MLP -> scalar."""

    def __init__(self, dim: int = 512, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        h2 = max(8, hidden // 2)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, 1),
        )

    def forward(self, stacked: torch.Tensor) -> torch.Tensor:
        # stacked: [K, D]
        x = stacked.mean(dim=0)
        return self.net(x.to(dtype=torch.bfloat16))


def _pad_batch_1d(windows: list[torch.Tensor], pad_id: int) -> torch.Tensor:
    mx = max(int(w.numel()) for w in windows)
    out = torch.full((len(windows), mx), pad_id, dtype=torch.long)
    for i, w in enumerate(windows):
        out[i, : w.numel()] = w
    return out


def build_genome_records(
    df: pd.DataFrame,
    tokenizer: CharLevelTokenizer,
    *,
    max_seq_len: int,
    window_stride: int,
    max_windows_per_genome: int,
    group_col: str = "Organism_Name",
) -> list[GenomeRecord]:
    records: list[GenomeRecord] = []
    stride_eff = window_stride if window_stride > 0 else max_seq_len
    for _, row in df.iterrows():
        name = str(row[group_col])
        seq = str(row["Sequence"])
        toks = tokenizer.tokenize(seq)
        wins = _build_windows_for_tokens(toks, max_seq_len, stride_eff)
        wins = _subsample_windows(wins, max_windows_per_genome)
        if not wins:
            continue
        y = float(row["Spillover_Score_Normalized"])
        records.append(GenomeRecord(name=name, windows=wins, target=y))
    return records


def train_one_epoch(
    backbone: EvoRegressorOnly,
    fusion: GlobalFusionHead,
    records: list[GenomeRecord],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    pad_id: int,
    window_microbatch: int,
    *,
    shuffle: bool,
) -> float:
    backbone.train()
    fusion.train()
    order = list(range(len(records)))
    if shuffle:
        random.shuffle(order)
    total_loss = 0.0
    n = 0
    t0 = time.time()
    for j, idx in enumerate(order):
        rec = records[idx]
        embs: list[torch.Tensor] = []
        for i in range(0, len(rec.windows), window_microbatch):
            chunk = rec.windows[i : i + window_microbatch]
            batch = _pad_batch_1d(chunk, pad_id).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                h = backbone.encode_pooled(batch)
            embs.append(h)
        stacked = torch.cat(embs, dim=0)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = fusion(stacked)
            tgt = torch.tensor([rec.target], device=device, dtype=torch.bfloat16)
            loss = criterion(pred.view(-1), tgt.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        n += 1
        if j % 10 == 0:
            elapsed = time.time() - t0
            print(
                f"  genome {j + 1}/{len(order)} loss={loss.item():.4f} "
                f"windows={len(rec.windows)} elapsed={timedelta(seconds=int(elapsed))}",
                flush=True,
            )
    return total_loss / max(1, n)


@torch.no_grad()
def eval_epoch(
    backbone: EvoRegressorOnly,
    fusion: GlobalFusionHead,
    records: list[GenomeRecord],
    criterion: nn.Module,
    device: torch.device,
    pad_id: int,
    window_microbatch: int,
) -> tuple[float, dict[str, float]]:
    backbone.eval()
    fusion.eval()
    preds: list[float] = []
    targs: list[float] = []
    total_loss = 0.0
    for rec in records:
        embs: list[torch.Tensor] = []
        for i in range(0, len(rec.windows), window_microbatch):
            chunk = rec.windows[i : i + window_microbatch]
            batch = _pad_batch_1d(chunk, pad_id).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                h = backbone.encode_pooled(batch)
            embs.append(h)
        stacked = torch.cat(embs, dim=0)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = fusion(stacked)
            tgt = torch.tensor([rec.target], device=device, dtype=torch.bfloat16)
            loss = criterion(pred.view(-1), tgt.view(-1))
        total_loss += float(loss.item())
        preds.append(float(pred.view(-1).float().cpu().item()))
        targs.append(rec.target)
    y_p = torch.tensor(preds, dtype=torch.float32)
    y_t = torch.tensor(targs, dtype=torch.float32)
    mse = mean_squared_error(y_t.numpy(), y_p.numpy())
    mae = mean_absolute_error(y_t.numpy(), y_p.numpy())
    r2 = r2_score(y_t.numpy(), y_p.numpy())
    pr = pearsonr(y_t.numpy(), y_p.numpy())[0] if len(preds) > 2 else float("nan")
    sp = spearmanr(y_t.numpy(), y_p.numpy())[0] if len(preds) > 2 else float("nan")
    metrics = {"val_mse": mse, "val_mae": mae, "val_r2": r2, "val_pearson": pr, "val_spearman": sp}
    return total_loss / max(1, len(records)), metrics


def main() -> None:
    p = argparse.ArgumentParser("LoRA + per-genome window fusion (mean pool + MLP)")
    p.add_argument("--base_model_path", type=str, default="evo-1-131k-base")
    p.add_argument("--data_csv", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--learning_rate", type=float, default=3e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--max_seq_len", type=int, default=8192)
    p.add_argument(
        "--window-overlap-fraction",
        type=float,
        default=0.1,
        dest="window_overlap_fraction",
        metavar="F",
        help="Used only when --window_stride<=0. Overlap = F * max_seq_len (F clamped to [0.1,0.5]). "
        "Stride = round(max_seq_len * (1-F)). Default 0.1 (~10%% overlap): fewer windows than 0.5.",
    )
    p.add_argument(
        "--window_stride",
        type=int,
        default=4096,
        help=">0: stride in tokens (default 4096 with max_seq_len 8192). <=0: use --window-overlap-fraction.",
    )
    p.add_argument(
        "--window_microbatch",
        type=int,
        default=1,
        help="Windows per encode_pooled forward (keep max_seq_len fixed; use 1 to minimize peak VRAM)",
    )
    p.add_argument("--max_windows_per_genome", type=int, default=256, help="Cap windows for very long genomes")
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=float, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    p.add_argument("--validation_split", type=float, default=0.2)
    p.add_argument("--output_root", type=str, default="train_fullgenome_notrunction")
    p.add_argument("--enable-gradient-checkpointing", dest="enable_gradient_checkpointing", type=int, default=1, choices=[0, 1])
    p.add_argument("--legacy-oversample-before-split", action="store_true")
    p.add_argument("--split-random-state", type=int, dest="split_random_state", default=42)
    args = p.parse_args()

    if os.environ.get("EVO_REQUIRE_CUDA", "1").strip().lower() not in ("0", "false", "no", "off"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for this script (single GPU).")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda:0")

    if args.data_csv:
        df = pd.read_csv(args.data_csv)
    else:
        train_df = pd.read_csv("evo_data/processed_ref_train_with_sequences.csv")
        spillover_df = pd.read_csv("evo_data/SpilloverRankings.csv")
        test_df = pd.read_csv("evo_data/processed_test_with_sequences_sampled.csv")
        df = prepare_multitask_data(train_df, spillover_df, test_df=test_df, max_test_samples=10000)
        df.to_csv("evo_data/processed_ref_train_with_spillover-ensambled-evo-all-strain.csv", index=False)

    tokenizer = CharLevelTokenizer(512)
    pad_id = int(getattr(tokenizer, "pad_id", 1))

    train_df, val_df = prepare_splits_leak_safe(
        df,
        test_size=args.validation_split,
        random_state=args.split_random_state,
        group_col="Organism_Name",
        oversample_train_only=not args.legacy_oversample_before_split,
        use_legacy_split=args.legacy_oversample_before_split,
    )

    model_max = _infer_model_max_seq_len(args.base_model_path)
    max_len = args.max_seq_len
    if model_max is not None:
        max_len = min(max_len, model_max)
    if args.window_stride > 0:
        stride_eff = int(args.window_stride)
        overlap_note = f"overlap=from_stride(stride={stride_eff})"
    else:
        oc = max(0.1, min(0.5, float(args.window_overlap_fraction)))
        stride_eff = max(1, int(round(max_len * (1.0 - oc))))
        overlap_note = f"overlap_frac={oc:.2f} stride={stride_eff}"
    print(
        f"max_seq_len={max_len} {overlap_note} max_windows_per_genome={args.max_windows_per_genome}",
        flush=True,
    )

    train_records = build_genome_records(
        train_df,
        tokenizer,
        max_seq_len=max_len,
        window_stride=stride_eff,
        max_windows_per_genome=args.max_windows_per_genome,
    )
    val_records = build_genome_records(
        val_df,
        tokenizer,
        max_seq_len=max_len,
        window_stride=stride_eff,
        max_windows_per_genome=args.max_windows_per_genome,
    )
    print(f"train_genomes={len(train_records)} val_genomes={len(val_records)}", flush=True)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["projections", "out_filter_dense", "Wqkv", "out_proj", "l1", "l2", "l3"],
        modules_to_save=["regressor"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="SEQ_CLS",
        inference_mode=False,
    )
    backbone = EvoRegressorOnly(
        base_model_path=args.base_model_path,
        device=device,
        lora_config=lora_config,
        enable_gradient_checkpointing=bool(args.enable_gradient_checkpointing),
    )
    backbone = backbone.to(dtype=torch.bfloat16)
    fusion = GlobalFusionHead(dim=512, hidden=256).to(device=device, dtype=torch.bfloat16)

    sh = _stripedhyena_for_precompute(backbone)
    if sh is not None and max_len > 0:
        _precompute_hyena_filters_compat(sh, int(max_len), device, 0)

    # Train fusion head + backbone (exclude unused per-window regressor to save optimizer state).
    params = [p for n, p in backbone.named_parameters() if not n.startswith("regressor.")] + list(fusion.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    out_root = os.path.join(args.output_root, "regression_genome_fusion_lora")
    os.makedirs(out_root, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_path = os.path.join(out_root, f"best_genome_fusion_{ts}.pth")

    best_val = float("inf")
    metrics_log: dict[str, list] = {
        "train_loss": [],
        "val_loss": [],
        "val_mse": [],
        "val_mae": [],
        "val_r2": [],
        "val_pearson": [],
        "val_spearman": [],
    }

    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch + 1}/{args.num_epochs} train...", flush=True)
        tr_loss = train_one_epoch(
            backbone,
            fusion,
            train_records,
            optimizer,
            criterion,
            device,
            pad_id,
            args.window_microbatch,
            shuffle=True,
        )
        print(f"Epoch {epoch + 1} train_loss={tr_loss:.6f} val...", flush=True)
        val_loss, vm = eval_epoch(
            backbone,
            fusion,
            val_records,
            criterion,
            device,
            pad_id,
            args.window_microbatch,
        )
        print(
            f"Epoch {epoch + 1} val_loss={val_loss:.6f} val_mse={vm.get('val_mse', float('nan')):.6f} "
            f"val_mae={vm.get('val_mae', float('nan')):.6f} val_r2={vm.get('val_r2', float('nan')):.4f} "
            f"val_pearson={vm.get('val_pearson', float('nan')):.4f} val_spearman={vm.get('val_spearman', float('nan')):.4f}",
            flush=True,
        )
        metrics_log["train_loss"].append(tr_loss)
        metrics_log["val_loss"].append(val_loss)
        for k in ("val_mse", "val_mae", "val_r2", "val_pearson", "val_spearman"):
            metrics_log[k].append(float(vm.get(k, float("nan"))))

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "backbone": backbone.state_dict(),
                    "fusion": fusion.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "metrics": vm,
                    "args": vars(args),
                },
                best_path,
            )
            print(f"  saved best -> {best_path}", flush=True)

    with open(os.path.join(out_root, f"metrics_genome_fusion_{ts}.json"), "w") as f:
        json.dump(metrics_log, f, indent=2)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
