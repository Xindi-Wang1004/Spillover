#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class SeqRow:
    input_ids: torch.Tensor
    cls_label: float | None
    reg_target: float | None


def _sliding_window_starts(seq_len: int, window_len: int, stride: int) -> list[int]:
    """
    Start offsets for sliding windows of length window_len that cover [0, seq_len).
    - If stride >= window_len: disjoint tiles (stride==window_len => non-overlap chunks).
    - If stride < window_len: overlapping scan; always includes last start seq_len - window_len.
    """
    if window_len <= 0:
        return [0]
    if seq_len <= window_len:
        return [0]
    stride = max(1, int(stride))
    if stride >= window_len:
        n = (seq_len + window_len - 1) // window_len
        return [i * window_len for i in range(n)]
    last = seq_len - window_len
    starts = list(range(0, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)
    return starts


def _pick_window_start(seq_len: int, window_len: int, *, window_idx: int, windows_per_sequence: int) -> int:
    """
    Choose a window start offset that covers the full sequence.
    Deterministic: spreads windows roughly uniformly across [0, seq_len - window_len].
    """
    if window_len <= 0 or seq_len <= window_len:
        return 0
    max_start = seq_len - window_len
    if windows_per_sequence <= 1:
        return 0
    # Evenly spaced positions across [0, max_start]
    # window_idx in [0, windows_per_sequence-1]
    t = window_idx / (windows_per_sequence - 1)
    return int(round(t * max_start))


class FullSequenceMultitaskDataset(Dataset):
    """Full-sequence dataset: uniform anchors or sliding windows along each sequence."""

    def __init__(
        self,
        sequences: List[str],
        class_labels,
        regression_targets,
        tokenizer,
        *,
        max_seq_len: int | None = None,
        truncate_side: str = "right",
        windows_per_sequence: int = 1,
        windowing: str = "sliding",
        window_stride: int | None = None,
    ):
        self.rows: List[SeqRow] = []
        self.max_seq_len = max_seq_len if (max_seq_len is not None and max_seq_len > 0) else None
        self.truncate_side = truncate_side
        self.windows_per_sequence = max(1, int(windows_per_sequence))
        self.windowing = (windowing or "sliding").lower()
        if self.max_seq_len is not None and self.max_seq_len > 0:
            self._stride = int(window_stride) if window_stride is not None else int(self.max_seq_len)
        else:
            self._stride = 1
        for s, c, r in zip(sequences, class_labels, regression_targets):
            toks = tokenizer.tokenize(str(s))
            if len(toks) == 0:
                continue
            self.rows.append(
                SeqRow(
                    input_ids=torch.tensor(toks, dtype=torch.long),
                    cls_label=float(c),
                    reg_target=float(r),
                )
            )

        self._indices: List[Tuple[int, int]] | None = None
        if self.windowing == "sliding":
            self._indices = []
            if self.max_seq_len is None:
                for i in range(len(self.rows)):
                    self._indices.append((i, 0))
            else:
                for i, row in enumerate(self.rows):
                    L = int(row.input_ids.numel())
                    W = int(self.max_seq_len)
                    if L <= W:
                        self._indices.append((i, 0))
                    else:
                        for st in _sliding_window_starts(L, W, self._stride):
                            self._indices.append((i, st))

    def __len__(self) -> int:
        if self._indices is not None:
            return len(self._indices)
        return len(self.rows) * self.windows_per_sequence

    def __getitem__(self, idx: int):
        if self._indices is not None:
            row_idx, start = self._indices[idx]
            row = self.rows[row_idx]
            x = row.input_ids
            if self.max_seq_len is None:
                pass
            else:
                W = int(self.max_seq_len)
                if x.numel() <= W:
                    pass
                else:
                    x = x[start : start + W]
            return x, torch.tensor(row.cls_label, dtype=torch.float32), torch.tensor(row.reg_target, dtype=torch.float32)

        row_idx = idx // self.windows_per_sequence
        window_idx = idx % self.windows_per_sequence
        row = self.rows[row_idx]
        x = row.input_ids

        if self.max_seq_len is not None and x.numel() > self.max_seq_len:
            start = _pick_window_start(int(x.numel()), int(self.max_seq_len), window_idx=window_idx, windows_per_sequence=self.windows_per_sequence)
            x = x[start : start + int(self.max_seq_len)]
        return x, torch.tensor(row.cls_label, dtype=torch.float32), torch.tensor(row.reg_target, dtype=torch.float32)


class FullSequenceRegressionDataset(Dataset):
    """Full-sequence regression: uniform anchors or sliding windows along each sequence."""

    def __init__(
        self,
        sequences: List[str],
        regression_targets,
        tokenizer,
        *,
        max_seq_len: int | None = None,
        truncate_side: str = "right",
        windows_per_sequence: int = 1,
        windowing: str = "sliding",
        window_stride: int | None = None,
    ):
        self.rows: List[SeqRow] = []
        self.max_seq_len = max_seq_len if (max_seq_len is not None and max_seq_len > 0) else None
        self.truncate_side = truncate_side
        self.windows_per_sequence = max(1, int(windows_per_sequence))
        self.windowing = (windowing or "sliding").lower()
        if self.max_seq_len is not None and self.max_seq_len > 0:
            self._stride = int(window_stride) if window_stride is not None else int(self.max_seq_len)
        else:
            self._stride = 1
        for s, r in zip(sequences, regression_targets):
            toks = tokenizer.tokenize(str(s))
            if len(toks) == 0:
                continue
            self.rows.append(
                SeqRow(
                    input_ids=torch.tensor(toks, dtype=torch.long),
                    cls_label=None,
                    reg_target=float(r),
                )
            )

        self._indices: List[Tuple[int, int]] | None = None
        if self.windowing == "sliding":
            self._indices = []
            if self.max_seq_len is None:
                for i in range(len(self.rows)):
                    self._indices.append((i, 0))
            else:
                for i, row in enumerate(self.rows):
                    L = int(row.input_ids.numel())
                    W = int(self.max_seq_len)
                    if L <= W:
                        self._indices.append((i, 0))
                    else:
                        for st in _sliding_window_starts(L, W, self._stride):
                            self._indices.append((i, st))

    def __len__(self) -> int:
        if self._indices is not None:
            return len(self._indices)
        return len(self.rows) * self.windows_per_sequence

    def __getitem__(self, idx: int):
        if self._indices is not None:
            row_idx, start = self._indices[idx]
            row = self.rows[row_idx]
            x = row.input_ids
            if self.max_seq_len is None:
                pass
            else:
                W = int(self.max_seq_len)
                if x.numel() <= W:
                    pass
                else:
                    x = x[start : start + W]
            return x, torch.tensor(row.reg_target, dtype=torch.float32)

        row_idx = idx // self.windows_per_sequence
        window_idx = idx % self.windows_per_sequence
        row = self.rows[row_idx]
        x = row.input_ids
        if self.max_seq_len is not None and x.numel() > self.max_seq_len:
            start = _pick_window_start(int(x.numel()), int(self.max_seq_len), window_idx=window_idx, windows_per_sequence=self.windows_per_sequence)
            x = x[start : start + int(self.max_seq_len)]
        return x, torch.tensor(row.reg_target, dtype=torch.float32)


class FullSequenceClassificationDataset(Dataset):
    """Full-sequence classification: uniform anchors or sliding windows along each sequence."""

    def __init__(
        self,
        sequences: List[str],
        class_labels,
        tokenizer,
        *,
        max_seq_len: int | None = None,
        truncate_side: str = "right",
        windows_per_sequence: int = 1,
        windowing: str = "sliding",
        window_stride: int | None = None,
    ):
        self.rows: List[SeqRow] = []
        self.max_seq_len = max_seq_len if (max_seq_len is not None and max_seq_len > 0) else None
        self.truncate_side = truncate_side
        self.windows_per_sequence = max(1, int(windows_per_sequence))
        self.windowing = (windowing or "sliding").lower()
        if self.max_seq_len is not None and self.max_seq_len > 0:
            self._stride = int(window_stride) if window_stride is not None else int(self.max_seq_len)
        else:
            self._stride = 1
        for s, c in zip(sequences, class_labels):
            toks = tokenizer.tokenize(str(s))
            if len(toks) == 0:
                continue
            self.rows.append(
                SeqRow(
                    input_ids=torch.tensor(toks, dtype=torch.long),
                    cls_label=float(c),
                    reg_target=None,
                )
            )

        self._indices: List[Tuple[int, int]] | None = None
        if self.windowing == "sliding":
            self._indices = []
            if self.max_seq_len is None:
                for i in range(len(self.rows)):
                    self._indices.append((i, 0))
            else:
                for i, row in enumerate(self.rows):
                    L = int(row.input_ids.numel())
                    W = int(self.max_seq_len)
                    if L <= W:
                        self._indices.append((i, 0))
                    else:
                        for st in _sliding_window_starts(L, W, self._stride):
                            self._indices.append((i, st))

    def __len__(self) -> int:
        if self._indices is not None:
            return len(self._indices)
        return len(self.rows) * self.windows_per_sequence

    def __getitem__(self, idx: int):
        if self._indices is not None:
            row_idx, start = self._indices[idx]
            row = self.rows[row_idx]
            x = row.input_ids
            if self.max_seq_len is None:
                pass
            else:
                W = int(self.max_seq_len)
                if x.numel() <= W:
                    pass
                else:
                    x = x[start : start + W]
            return x, torch.tensor(row.cls_label, dtype=torch.float32)

        row_idx = idx // self.windows_per_sequence
        window_idx = idx % self.windows_per_sequence
        row = self.rows[row_idx]
        x = row.input_ids
        if self.max_seq_len is not None and x.numel() > self.max_seq_len:
            start = _pick_window_start(int(x.numel()), int(self.max_seq_len), window_idx=window_idx, windows_per_sequence=self.windows_per_sequence)
            x = x[start : start + int(self.max_seq_len)]
        return x, torch.tensor(row.cls_label, dtype=torch.float32)


def pad_collate_1d(batch, *, pad_id: int) -> torch.Tensor:
    lens = [int(x.numel()) for x in batch]
    mx = max(lens)
    out = torch.full((len(batch), mx), int(pad_id), dtype=torch.long)
    for i, x in enumerate(batch):
        out[i, : x.numel()] = x
    return out


def collate_multitask(batch, *, pad_id: int):
    xs, c, r = zip(*batch)
    xpad = pad_collate_1d(xs, pad_id=pad_id)
    return xpad, torch.stack(c).float(), torch.stack(r).float()


def collate_regression(batch, *, pad_id: int):
    xs, r = zip(*batch)
    xpad = pad_collate_1d(xs, pad_id=pad_id)
    return xpad, torch.stack(r).float()


def collate_classification(batch, *, pad_id: int):
    xs, c = zip(*batch)
    xpad = pad_collate_1d(xs, pad_id=pad_id)
    return xpad, torch.stack(c).float()

