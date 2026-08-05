#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-genome late fusion (multitask): sliding windows (default max_seq_len=8192, stride=4096),
EvoMultitaskOnly.encode_pooled per window, mean-pool [K,512] then shared classifier+regressor heads
(same architecture as EvoMultitaskOnly). Loss = alpha * BCE(cls) + (1-alpha) * MSE(reg).
Single-GPU only.

When --window_stride<=0, stride is derived from --window-overlap-fraction.
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
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from stripedhyena.tokenizer import CharLevelTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_tr_spec = importlib.util.spec_from_file_location(
    "_tr_mt_fusion_ref",
    Path(__file__).resolve().parent / "train_regression_fullgenome_notrunc_lora.py",
)
_tr = importlib.util.module_from_spec(_tr_spec)
assert _tr_spec.loader is not None
_tr_spec.loader.exec_module(_tr)

_tr._configure_hf_cache_and_proxy_defaults()
if os.environ.get("EVO_HF_ALLOW_ONLINE", "").strip().lower() not in ("1", "true", "yes"):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

_hf_home = os.environ.get("HF_HOME") or "/home/wangxindi/hf-cache"
os.environ["HF_HOME"] = _hf_home
if os.environ.get("EVO_HF_TRUST_PARENT_CACHE_PATHS", "").strip().lower() not in ("1", "true", "yes"):
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(_hf_home, "hub")
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(_hf_home, "hub")

from data_processing_multitask_strain import prepare_multitask_data
from model_ablation_single_task import EvoMultitaskOnly
from train_split_leak_safe import prepare_splits_leak_safe

from _datasets_notrunc import _sliding_window_starts


def _stripedhyena_for_precompute(model: nn.Module) -> nn.Module | None:
    return _tr._stripedhyena_for_precompute(model)


def _precompute_hyena_filters_compat(striped: nn.Module, L: int, device: torch.device, rank: int) -> None:
    return _tr._precompute_hyena_filters_compat(striped, L, device, rank)


def _infer_model_max_seq_len(base_model_path: str) -> int | None:
    return _tr._infer_model_max_seq_len(base_model_path)


@dataclass
class MTGenomeRecord:
    name: str
    windows: list[torch.Tensor]
    cls_label: float
    reg_target: float


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


def _pad_batch_1d(windows: list[torch.Tensor], pad_id: int) -> torch.Tensor:
    mx = max(int(w.numel()) for w in windows)
    out = torch.full((len(windows), mx), pad_id, dtype=torch.long)
    for i, w in enumerate(windows):
        out[i, : w.numel()] = w
    return out


def build_mt_genome_records(
    df: pd.DataFrame,
    tokenizer: CharLevelTokenizer,
    *,
    max_seq_len: int,
    window_stride: int,
    max_windows_per_genome: int,
    group_col: str = "Organism_Name",
) -> list[MTGenomeRecord]:
    records: list[MTGenomeRecord] = []
    stride_eff = window_stride if window_stride > 0 else max_seq_len
    for _, row in df.iterrows():
        name = str(row[group_col])
        seq = str(row["Sequence"])
        toks = tokenizer.tokenize(seq)
        wins = _build_windows_for_tokens(toks, max_seq_len, stride_eff)
        wins = _subsample_windows(wins, max_windows_per_genome)
        if not wins:
            continue
        records.append(
            MTGenomeRecord(
                name=name,
                windows=wins,
                cls_label=float(row["Host"]),
                reg_target=float(row["Spillover_Score_Normalized"]),
            )
        )
    return records


def train_one_epoch(
    model: EvoMultitaskOnly,
    records: list[MTGenomeRecord],
    optimizer: torch.optim.Optimizer,
    cls_crit: nn.Module,
    reg_crit: nn.Module,
    alpha: float,
    device: torch.device,
    pad_id: int,
    window_microbatch: int,
    *,
    shuffle: bool,
) -> float:
    model.train()
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
                h = model.encode_pooled(batch)
            embs.append(h)
        stacked = torch.cat(embs, dim=0)
        fused = stacked.mean(dim=0, keepdim=True).to(dtype=torch.bfloat16)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            cls_pred = model.classifier(fused)
            reg_pred = model.regressor(fused)
        # BCELoss is not autocast-safe with bf16; compute losses in fp32 outside autocast.
        cls_t = torch.tensor([[rec.cls_label]], device=device, dtype=torch.float32)
        reg_t = torch.tensor([[rec.reg_target]], device=device, dtype=torch.float32)
        cls_loss = cls_crit(cls_pred.float(), cls_t)
        reg_loss = reg_crit(reg_pred.float(), reg_t)
        loss = alpha * cls_loss + (1.0 - alpha) * reg_loss
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


def _val_cls_metrics(
    raw: np.ndarray, labels: np.ndarray, *, from_logits: bool = True
) -> dict[str, float]:
    """Binary classification metrics on val. Use from_logits=False when head ends with Sigmoid (probs)."""
    raw = raw.reshape(-1)
    labels = labels.reshape(-1)
    if from_logits:
        probs = 1.0 / (1.0 + np.exp(-raw))
    else:
        probs = raw.astype(np.float64)
    preds = (probs >= 0.5).astype(int)
    if len(np.unique(labels)) > 1:
        _sr = spearmanr(labels, probs)
        _sp = getattr(_sr, "statistic", None)
        if _sp is None:
            _sp = getattr(_sr, "correlation", float("nan"))
        spv = float(_sp)
    else:
        spv = float("nan")
    return {
        "val_cls_accuracy": float(accuracy_score(labels, preds)),
        "val_cls_precision": float(precision_score(labels, preds, zero_division=0)),
        "val_cls_recall": float(recall_score(labels, preds, zero_division=0)),
        "val_cls_f1": float(f1_score(labels, preds, zero_division=0)),
        "val_cls_auc": float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else float("nan"),
        "val_cls_ap": float(average_precision_score(labels, probs)) if len(np.unique(labels)) > 1 else float("nan"),
        "val_cls_mse": float(mean_squared_error(labels, probs)),
        "val_cls_spearman": spv,
    }


@torch.no_grad()
def eval_epoch(
    model: EvoMultitaskOnly,
    records: list[MTGenomeRecord],
    cls_crit: nn.Module,
    reg_crit: nn.Module,
    alpha: float,
    device: torch.device,
    pad_id: int,
    window_microbatch: int,
) -> tuple[float, dict[str, float]]:
    model.eval()
    losses: list[float] = []
    reg_preds: list[float] = []
    reg_targs: list[float] = []
    cls_logits: list[float] = []
    cls_labels: list[float] = []
    for rec in records:
        embs: list[torch.Tensor] = []
        for i in range(0, len(rec.windows), window_microbatch):
            chunk = rec.windows[i : i + window_microbatch]
            batch = _pad_batch_1d(chunk, pad_id).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                h = model.encode_pooled(batch)
            embs.append(h)
        stacked = torch.cat(embs, dim=0)
        fused = stacked.mean(dim=0, keepdim=True).to(dtype=torch.bfloat16)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            cls_pred = model.classifier(fused)
            reg_pred = model.regressor(fused)
        cls_t = torch.tensor([[rec.cls_label]], device=device, dtype=torch.float32)
        reg_t = torch.tensor([[rec.reg_target]], device=device, dtype=torch.float32)
        cls_loss = cls_crit(cls_pred.float(), cls_t)
        reg_loss = reg_crit(reg_pred.float(), reg_t)
        loss = alpha * cls_loss + (1.0 - alpha) * reg_loss
        losses.append(float(loss.item()))
        reg_preds.append(float(reg_pred.view(-1).float().cpu().item()))
        reg_targs.append(rec.reg_target)
        cls_logits.append(float(cls_pred.view(-1).float().cpu().item()))
        cls_labels.append(float(rec.cls_label))
    avg = float(np.mean(losses)) if losses else float("nan")
    rp = np.array(reg_preds, dtype=np.float32)
    rt = np.array(reg_targs, dtype=np.float32)
    y_l = np.array(cls_logits, dtype=np.float32)
    y_c = np.array(cls_labels, dtype=np.float32)
    cm = _val_cls_metrics(y_l, y_c, from_logits=False)
    if rt.size > 1:
        _sr = spearmanr(rt, rp)
        _v = getattr(_sr, "statistic", None)
        if _v is None:
            _v = getattr(_sr, "correlation", float("nan"))
        sp = float(_v)
    else:
        sp = float("nan")
    reg_out: dict[str, float] = {
        "val_reg_mse": float(mean_squared_error(rt, rp)) if rt.size else float("nan"),
        "val_reg_mae": float(mean_absolute_error(rt, rp)) if rt.size else float("nan"),
        "val_reg_r2": float(r2_score(rt, rp)) if rt.size > 1 else float("nan"),
        "val_reg_pearson": float(pearsonr(rt, rp)[0]) if rt.size > 2 else float("nan"),
        "val_reg_spearman": sp,
    }
    out = {"val_loss": avg, **cm, **reg_out}
    return avg, out


def main() -> None:
    p = argparse.ArgumentParser("LoRA + per-genome window fusion (multitask cls+reg)")
    p.add_argument("--base_model_path", type=str, default="evo-1-131k-base")
    p.add_argument("--data_csv", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--alpha", type=float, default=0.5, help="weight on classification term")
    p.add_argument("--max_seq_len", type=int, default=8192)
    p.add_argument(
        "--window-overlap-fraction",
        type=float,
        default=0.1,
        dest="window_overlap_fraction",
        metavar="F",
        help="Used only when --window_stride<=0.",
    )
    p.add_argument(
        "--window_stride",
        type=int,
        default=4096,
        help=">0: stride in tokens (default 4096 with max_seq_len 8192). <=0: overlap-fraction.",
    )
    p.add_argument("--window_microbatch", type=int, default=1)
    p.add_argument("--max_windows_per_genome", type=int, default=256)
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
        f"max_seq_len={max_len} {overlap_note} max_windows_per_genome={args.max_windows_per_genome} alpha={args.alpha}",
        flush=True,
    )

    train_records = build_mt_genome_records(
        train_df, tokenizer, max_seq_len=max_len, window_stride=stride_eff, max_windows_per_genome=args.max_windows_per_genome
    )
    val_records = build_mt_genome_records(
        val_df, tokenizer, max_seq_len=max_len, window_stride=stride_eff, max_windows_per_genome=args.max_windows_per_genome
    )
    print(f"train_genomes={len(train_records)} val_genomes={len(val_records)}", flush=True)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["projections", "out_filter_dense", "Wqkv", "out_proj", "l1", "l2", "l3"],
        modules_to_save=["classifier", "regressor"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="SEQ_CLS",
        inference_mode=False,
    )
    model = EvoMultitaskOnly(
        base_model_path=args.base_model_path,
        device=device,
        lora_config=lora_config,
        enable_gradient_checkpointing=bool(args.enable_gradient_checkpointing),
    )
    model = model.to(dtype=torch.bfloat16)

    sh = _stripedhyena_for_precompute(model)
    if sh is not None and max_len > 0:
        _precompute_hyena_filters_compat(sh, int(max_len), device, 0)

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate, weight_decay=args.weight_decay)
    cls_crit = nn.BCELoss()
    reg_crit = nn.MSELoss()

    out_root = os.path.join(args.output_root, f"multitask_genome_fusion_alpha{args.alpha:g}_lora")
    os.makedirs(out_root, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_path = os.path.join(out_root, f"best_multitask_genome_fusion_{ts}.pth")

    best_val = float("inf")
    _mt_val_keys = [
        "val_loss",
        "val_cls_accuracy",
        "val_cls_precision",
        "val_cls_recall",
        "val_cls_f1",
        "val_cls_auc",
        "val_cls_ap",
        "val_cls_mse",
        "val_cls_spearman",
        "val_reg_mse",
        "val_reg_mae",
        "val_reg_r2",
        "val_reg_pearson",
        "val_reg_spearman",
    ]
    metrics_log: dict[str, list] = {"train_loss": [], **{k: [] for k in _mt_val_keys}}

    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch + 1}/{args.num_epochs} train...", flush=True)
        tr_loss = train_one_epoch(
            model,
            train_records,
            optimizer,
            cls_crit,
            reg_crit,
            float(args.alpha),
            device,
            pad_id,
            args.window_microbatch,
            shuffle=True,
        )
        print(f"Epoch {epoch + 1} train_loss={tr_loss:.6f} val...", flush=True)
        val_loss, vm = eval_epoch(
            model,
            val_records,
            cls_crit,
            reg_crit,
            float(args.alpha),
            device,
            pad_id,
            args.window_microbatch,
        )
        print(
            f"Epoch {epoch + 1} val_loss={val_loss:.6f} "
            f"val_cls_accuracy={vm.get('val_cls_accuracy', float('nan')):.4f} "
            f"val_cls_precision={vm.get('val_cls_precision', float('nan')):.4f} "
            f"val_cls_recall={vm.get('val_cls_recall', float('nan')):.4f} "
            f"val_cls_f1={vm.get('val_cls_f1', float('nan')):.4f} "
            f"val_cls_auc={vm.get('val_cls_auc', float('nan')):.4f} "
            f"val_cls_ap={vm.get('val_cls_ap', float('nan')):.4f} "
            f"val_cls_mse={vm.get('val_cls_mse', float('nan')):.4f} "
            f"val_cls_spearman={vm.get('val_cls_spearman', float('nan')):.4f} "
            f"val_reg_mse={vm.get('val_reg_mse', float('nan')):.6f} "
            f"val_reg_mae={vm.get('val_reg_mae', float('nan')):.6f} "
            f"val_reg_r2={vm.get('val_reg_r2', float('nan')):.4f} "
            f"val_reg_pearson={vm.get('val_reg_pearson', float('nan')):.4f} "
            f"val_reg_spearman={vm.get('val_reg_spearman', float('nan')):.4f}",
            flush=True,
        )
        metrics_log["train_loss"].append(tr_loss)
        for k in _mt_val_keys:
            metrics_log[k].append(float(vm.get(k, float("nan"))))

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss, "metrics": vm, "args": vars(args)}, best_path)
            print(f"  saved best -> {best_path}", flush=True)

    with open(os.path.join(out_root, f"metrics_multitask_genome_fusion_{ts}.json"), "w") as f:
        json.dump(metrics_log, f, indent=2)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
