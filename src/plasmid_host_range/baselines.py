"""Simple k-mer + logistic regression baseline, reported alongside the DL model.

The point of this file is to keep the DL claim honest: any improvement from fine-tuning
should be measured against a cheap classical baseline, not assumed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score


def _seq_to_kmer_string(seq: str, k: int = 6) -> str:
    seq = seq.upper()
    return " ".join(seq[i : i + k] for i in range(0, len(seq) - k + 1, 1))


def _vectorize(seqs: list[str], k: int = 6, n_features: int = 2**18) -> np.ndarray:
    vec = HashingVectorizer(
        n_features=n_features,
        analyzer="word",
        alternate_sign=False,
        norm="l2",
    )
    return vec.transform(_seq_to_kmer_string(s, k) for s in seqs)


def kmer_baseline_eval(
    train_path: Path,
    test_path: Path,
    num_labels: int,
    k: int = 6,
    max_train: int = 20_000,
    subsample_test: int | None = None,
    seed: int = 0,
) -> dict:
    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)
    if subsample_test is not None and subsample_test < len(test):
        test = test.sample(n=subsample_test, random_state=seed).reset_index(drop=True)
    if len(train) > max_train:
        train = train.sample(n=max_train, random_state=seed).reset_index(drop=True)

    Xtr = _vectorize(train["sequence"].tolist(), k=k)
    Xte = _vectorize(test["sequence"].tolist(), k=k)
    ytr = train["label"].to_numpy()
    yte = test["label"].to_numpy()

    clf = LogisticRegression(max_iter=1000, n_jobs=-1, class_weight="balanced")
    clf.fit(Xtr, ytr)
    probs = clf.predict_proba(Xte)
    preds = probs.argmax(-1)

    class_list = list(range(num_labels))
    return {
        "top1_accuracy": float(accuracy_score(yte, preds)),
        "top3_accuracy": float(
            top_k_accuracy_score(yte, probs, k=min(3, num_labels), labels=class_list)
        ),
        "macro_f1": float(f1_score(yte, preds, average="macro", zero_division=0)),
    }
