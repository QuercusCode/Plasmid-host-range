"""Group-aware train/val/test splitting.

We split by host *species* so that near-identical plasmids from the same species do
not end up on both sides of the train/test boundary. This is a cheap approximation
of a proper ANI-clustered split; swap in a cluster-aware grouping here if you have
Mash/skani clusters available.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def group_split(
    df: pd.DataFrame,
    group_col: str = "species",
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Split ``df`` into train/val/test by unique values of ``group_col``."""
    if group_col not in df.columns:
        raise KeyError(f"{group_col!r} column required for grouped split")
    groups = df[group_col].fillna("__unknown__").astype(str).to_numpy()

    # First carve out test
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    trainval_idx, test_idx = next(gss1.split(df, groups=groups))
    trainval = df.iloc[trainval_idx].reset_index(drop=True)
    test = df.iloc[test_idx].reset_index(drop=True)

    # Then carve val out of the remainder, renormalizing the fraction
    rel_val = val_frac / (1.0 - test_frac)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=rel_val, random_state=seed + 1)
    tv_groups = trainval[group_col].fillna("__unknown__").astype(str).to_numpy()
    train_idx, val_idx = next(gss2.split(trainval, groups=tv_groups))
    train = trainval.iloc[train_idx].reset_index(drop=True)
    val = trainval.iloc[val_idx].reset_index(drop=True)

    return {"train": train, "val": val, "test": test}


def assert_no_group_leakage(splits: dict[str, pd.DataFrame], group_col: str) -> None:
    seen: dict[str, str] = {}
    for name, part in splits.items():
        for g in part[group_col].dropna().unique():
            if g in seen and seen[g] != name:
                raise AssertionError(
                    f"group {g!r} appears in both {seen[g]!r} and {name!r} splits"
                )
            seen[g] = name


def compute_class_weights(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Inverse-frequency class weights, normalized so the mean is 1."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    w = counts.sum() / (num_classes * counts)
    return (w / w.mean()).astype(np.float32)
