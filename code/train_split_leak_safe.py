#!/usr/bin/env python3
"""
Leak-safe train/validation preparation for full-genome scripts.

Policy (rebuttal / Methods):
1) Split on the original, non-duplicated dataframe first.
2) Group-wise split by Organism_Name so species with species-level spillover labels never
   appear in both train and val (GroupShuffleSplit).
3) Oversample minority Host class only on the training split (validation keeps natural imbalance).
4) Assert zero overlap between train and val on Accession (or sequence hash fallback).

Legacy behavior (oversample-then-split) is available via use_legacy_split=True for ablations only.
"""
from __future__ import annotations

import hashlib
from typing import Tuple

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def row_sample_ids(df: pd.DataFrame) -> pd.Series:
    """Stable per-row ID for leakage checks: Accession if present, else SHA256 of Sequence."""
    if "Accession" in df.columns:
        return df["Accession"].astype(str)
    return df["Sequence"].apply(
        lambda s: hashlib.sha256(str(s).encode("utf-8", errors="ignore")).hexdigest()[:40]
    )


def oversample_host_minority(df: pd.DataFrame, *, reset_index: bool = True) -> pd.DataFrame:
    """Repeat minority Host rows to roughly balance counts (same rule as original scripts)."""
    host_1_samples = df[df["Host"] == 1]
    host_0_samples = df[df["Host"] == 0]
    if len(host_1_samples) == 0 or len(host_0_samples) == 0:
        raise ValueError("Both Host classes must have at least one sample for oversampling.")
    if len(host_1_samples) < len(host_0_samples):
        oversample_ratio = max(1, len(host_0_samples) // len(host_1_samples))
        host_1_oversampled = pd.concat([host_1_samples] * oversample_ratio, ignore_index=True)
        balanced = pd.concat([host_0_samples, host_1_oversampled], ignore_index=True)
    else:
        oversample_ratio = max(1, len(host_1_samples) // len(host_0_samples))
        host_0_oversampled = pd.concat([host_0_samples] * oversample_ratio, ignore_index=True)
        balanced = pd.concat([host_1_samples, host_0_oversampled], ignore_index=True)
    if reset_index:
        return balanced.reset_index(drop=True)
    return balanced


def group_train_val_split(
    df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    group_col: str = "Organism_Name",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Train/val split with disjoint species (groups). Rows with NaN group are dropped."""
    if group_col not in df.columns:
        raise ValueError(
            f"Column {group_col!r} is required for group split (species-level label safety)."
        )
    clean = df.dropna(subset=[group_col]).copy()
    if len(clean) < 2:
        raise ValueError("Not enough rows after dropping NaN groups for GroupShuffleSplit.")
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    groups = clean[group_col].astype(str).values
    idx_train, idx_val = next(gss.split(clean, groups=groups))
    train_df = clean.iloc[idx_train].reset_index(drop=True)
    val_df = clean.iloc[idx_val].reset_index(drop=True)
    return train_df, val_df


def assert_train_val_disjoint(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    group_col: str = "Organism_Name",
    id_hint: str = "Accession",
) -> None:
    """Hard checks: no shared species; no shared accession / sequence id."""
    g_train = set(train_df[group_col].astype(str))
    g_val = set(val_df[group_col].astype(str))
    inter_g = g_train & g_val
    if inter_g:
        sample = list(inter_g)[:5]
        raise AssertionError(
            f"train/val overlap on {group_col}: {len(inter_g)} groups, e.g. {sample}"
        )
    if id_hint in train_df.columns and id_hint in val_df.columns:
        i_train = set(train_df[id_hint].astype(str))
        i_val = set(val_df[id_hint].astype(str))
        inter_i = i_train & i_val
        if inter_i:
            raise AssertionError(
                f"train/val overlap on {id_hint}: {len(inter_i)} ids, e.g. {list(inter_i)[:5]}"
            )
    else:
        ht = set(row_sample_ids(train_df))
        hv = set(row_sample_ids(val_df))
        inter = ht & hv
        if inter:
            raise AssertionError(
                f"train/val overlap on derived sequence/accession ids: {len(inter)}"
            )


def assert_val_not_in_train_after_oversample(
    train_oversampled: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    id_hint: str = "Accession",
) -> None:
    """After oversampling, train may repeat rows; val rows must still be disjoint by id."""
    if id_hint in train_oversampled.columns and id_hint in val_df.columns:
        tset = set(train_oversampled[id_hint].astype(str))
        vset = set(val_df[id_hint].astype(str))
    else:
        tset = set(row_sample_ids(train_oversampled))
        vset = set(row_sample_ids(val_df))
    inter = tset & vset
    if inter:
        raise AssertionError(
            f"After oversampling, val ids still overlap train unique ids: {len(inter)} e.g. {list(inter)[:3]}"
        )


def prepare_splits_leak_safe(
    df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    group_col: str = "Organism_Name",
    oversample_train_only: bool = True,
    use_legacy_split: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (train_df_for_model, val_df) where train_df may be oversampled if oversample_train_only.

    If use_legacy_split: restores old oversample-then-split (for comparison / deprecated).
    """
    if use_legacy_split:
        balanced = oversample_host_minority(df)
        train_df, val_df = train_test_split(
            balanced,
            test_size=test_size,
            stratify=balanced["Host"],
            random_state=random_state,
        )
        return train_df.reset_index(drop=True), val_df.reset_index(drop=True)

    train_df, val_df = group_train_val_split(
        df, test_size=test_size, random_state=random_state, group_col=group_col
    )
    assert_train_val_disjoint(train_df, val_df, group_col=group_col)
    if oversample_train_only:
        train_out = oversample_host_minority(train_df)
        assert_val_not_in_train_after_oversample(train_out, val_df)
        return train_out, val_df
    return train_df, val_df
