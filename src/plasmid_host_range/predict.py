"""Public API: ``predict_host_genus``.

This is the headline "function about working with plasmids": given a plasmid sequence or
FASTA file, predict the bacterial host genus using a fine-tuned DNA foundation model.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import torch
from Bio import SeqIO
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_CHECKPOINT = Path("checkpoints/default/best")


@dataclass
class HostPrediction:
    accession: str
    top_genera: list[str]
    scores: list[float]

    def to_dict(self) -> dict:
        return asdict(self)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / ex.sum(axis=axis, keepdims=True)


def _chunk(seq: str, window: int, stride: int) -> list[str]:
    if len(seq) <= window:
        return [seq]
    out = [seq[i : i + window] for i in range(0, len(seq) - window + 1, stride)]
    # Make sure the tail is covered
    if (len(seq) - window) % stride != 0:
        out.append(seq[-window:])
    return out


def _looks_like_dna(s: str) -> bool:
    """Heuristic: a non-empty string whose first 1 kb is only A/C/G/T/N is DNA."""
    if not s:
        return False
    sample = s[:1000].upper()
    return set(sample) <= set("ACGTN")


def _load_records(sequence_or_fasta: str | Path) -> list[tuple[str, str]]:
    """Return list of (accession, sequence) tuples."""
    # Raw DNA string check first — must happen before Path(), because long DNA
    # strings trip macOS "File name too long" when Path.exists() is called.
    if isinstance(sequence_or_fasta, str) and _looks_like_dna(sequence_or_fasta):
        return [("query", sequence_or_fasta.upper())]
    path = Path(sequence_or_fasta)
    if not path.exists():
        raise FileNotFoundError(f"Not a DNA string and not a file: {sequence_or_fasta}")
    records = []
    for rec in SeqIO.parse(str(path), "fasta"):
        records.append((rec.id, str(rec.seq).upper()))
    if not records:
        raise ValueError(f"No FASTA records found in {path}")
    return records


@torch.no_grad()
def predict_host_genus(
    sequence_or_fasta: str | Path,
    model_dir: str | Path = DEFAULT_CHECKPOINT,
    top_k: int = 3,
    window_size: int = 1_000,
    stride: int | None = None,
    max_tokens: int = 512,
    batch_size: int = 16,
    aggregate: Literal["mean", "max"] = "mean",
    device: str | None = None,
) -> list[HostPrediction]:
    """Predict the bacterial host genus of one or more plasmids.

    Parameters
    ----------
    sequence_or_fasta:
        A raw DNA string (A/C/G/T/N), or a path to a FASTA file with one or more records.
    model_dir:
        Directory containing a fine-tuned checkpoint (from ``plasmid-host-range train``),
        including ``label_names.json``.
    top_k:
        Number of top predictions to return per plasmid.
    window_size, stride, max_tokens:
        Windowing + tokenization controls. Windows are aggregated back to a per-plasmid score.
    aggregate:
        ``"mean"`` averages per-window logits (default); ``"max"`` takes the elementwise max.

    Returns
    -------
    list[HostPrediction]
        One prediction per input plasmid, in input order.
    """
    model_dir = Path(model_dir)
    if not (model_dir / "label_names.json").exists():
        raise FileNotFoundError(
            f"{model_dir} is missing label_names.json — is this a fine-tuned checkpoint?"
        )
    label_names: Sequence[str] = json.loads((model_dir / "label_names.json").read_text())

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir), trust_remote_code=True
    )
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    stride = stride or window_size  # non-overlapping by default
    records = _load_records(sequence_or_fasta)

    results: list[HostPrediction] = []
    for acc, seq in records:
        windows = _chunk(seq, window_size, stride)
        logits_per_window = []
        for start in range(0, len(windows), batch_size):
            batch = windows[start : start + batch_size]
            enc = tokenizer(
                batch,
                truncation=True,
                padding="max_length",
                max_length=max_tokens,
                return_tensors="pt",
            ).to(device)
            out = model(**enc).logits.float().cpu().numpy()
            logits_per_window.append(out)
        logits = np.concatenate(logits_per_window, axis=0)

        agg = logits.mean(0) if aggregate == "mean" else logits.max(0)
        probs = _softmax(agg)
        order = np.argsort(probs)[::-1][:top_k]
        results.append(
            HostPrediction(
                accession=acc,
                top_genera=[label_names[i] for i in order],
                scores=[float(probs[i]) for i in order],
            )
        )

    return results
