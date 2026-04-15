"""Torch Dataset that chunks plasmids into fixed-length nucleotide windows on the fly."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class PlasmidWindowDataset(Dataset):
    """Each ``__getitem__`` returns a tokenized window from a plasmid.

    * In ``mode='train'`` a random window is sampled per call (data augmentation).
    * In ``mode='eval'`` ``eval_windows_per_plasmid`` evenly-spaced windows are enumerated
      deterministically; ``__len__`` accounts for this.
    """

    def __init__(
        self,
        parquet_path: str | Path,
        tokenizer,
        window_size: int = 1_000,
        max_tokens: int = 512,
        mode: Literal["train", "eval"] = "train",
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
        self.eval_windows = eval_windows_per_plasmid
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        if self.mode == "train":
            return len(self.df)
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
        # k-th of eval_windows evenly-spaced starts
        stops = max(1, self.eval_windows)
        if stops == 1:
            start = (n - self.window_size) // 2
        else:
            start = int(round(k * (n - self.window_size) / (stops - 1)))
        return seq[start : start + self.window_size]

    def __getitem__(self, idx: int) -> dict:
        if self.mode == "train":
            row = self.df.iloc[idx]
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
