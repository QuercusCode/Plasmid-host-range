"""Torch Dataset that chunks plasmids into fixed-length nucleotide windows on the fly."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler


class PlasmidWindowDataset(Dataset):
    """Each ``__getitem__`` returns a tokenized window from a plasmid.

    * In ``mode='train'``, ``train_windows_per_plasmid`` random windows are
      enumerated per epoch (data augmentation through multi-window sampling).
      This means the model sees multiple views of each plasmid per epoch
      instead of just one, dramatically increasing effective sequence coverage.
    * In ``mode='eval'``, ``eval_windows_per_plasmid`` evenly-spaced windows
      are enumerated deterministically for aggregation.
    """

    def __init__(
        self,
        parquet_path: str | Path,
        tokenizer,
        window_size: int = 1_000,
        max_tokens: int = 512,
        mode: Literal["train", "eval"] = "train",
        train_windows_per_plasmid: int = 1,
        eval_windows_per_plasmid: int = 8,
        subsample: int | None = None,
        seed: int = 0,
    ) -> None:
        self.df = pd.read_parquet(parquet_path).reset_index(drop=True)
        if subsample is not None and subsample < len(self.df):
            self.df = self.df.sample(n=subsample, random_state=seed).reset_index(drop=True)
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.max_tokens = max_tokens
        self.mode = mode
        self.train_windows = max(1, train_windows_per_plasmid)
        self.eval_windows = eval_windows_per_plasmid
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        if self.mode == "train":
            return len(self.df) * self.train_windows
        return len(self.df) * self.eval_windows

    def _sample_window(self, seq: str) -> str:
        n = len(seq)
        if n <= self.window_size:
            return seq
        start = int(self._rng.integers(0, n - self.window_size + 1))
        return seq[start : start + self.window_size]

    def _eval_window(self, seq: str, k: int) -> str:
        n = len(seq)
        if n <= self.window_size:
            return seq
        stops = max(1, self.eval_windows)
        if stops == 1:
            start = (n - self.window_size) // 2
        else:
            start = int(round(k * (n - self.window_size) / (stops - 1)))
        return seq[start : start + self.window_size]

    def __getitem__(self, idx: int) -> dict:
        if self.mode == "train":
            plasmid_idx = idx % len(self.df)
            row = self.df.iloc[plasmid_idx]
            window = self._sample_window(row["sequence"])
        else:
            plasmid_idx, k = divmod(idx, self.eval_windows)
            row = self.df.iloc[plasmid_idx]
            window = self._eval_window(row["sequence"], k)

        enc = self.tokenizer(
            window,
            truncation=True,
            padding="max_length",
            max_length=self.max_tokens,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(int(row["label"]), dtype=torch.long)
        return item

    @property
    def plasmid_count(self) -> int:
        return len(self.df)

    @property
    def labels(self) -> np.ndarray:
        return self.df["label"].to_numpy()

    @property
    def expanded_labels(self) -> np.ndarray:
        """Labels expanded for multi-window: one label per training sample."""
        base = self.df["label"].to_numpy()
        if self.mode == "train":
            return np.tile(base, self.train_windows)
        return np.tile(base, self.eval_windows)


def build_oversampler(dataset: PlasmidWindowDataset) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler that oversamples rare classes so each class
    is seen roughly equally often per epoch. Critical for imbalanced datasets
    where 'Other' and 'Escherichia' dominate."""
    labels = dataset.expanded_labels
    class_counts = np.bincount(labels)
    # Inverse frequency: rarer classes get higher weight
    class_weights = 1.0 / np.where(class_counts == 0, 1, class_counts)
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float64),
        num_samples=len(labels),
        replacement=True,
    )
