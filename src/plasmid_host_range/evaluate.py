"""Evaluate a trained checkpoint on the test split; compare against k-mer baseline."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    top_k_accuracy_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .baselines import kmer_baseline_eval
from .data.dataset import PlasmidWindowDataset


@torch.no_grad()
def _predict_plasmid_logits(
    model,
    dataset: PlasmidWindowDataset,
    batch_size: int = 32,
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate per-window logits back to per-plasmid by averaging."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    n_plasmids = dataset.plasmid_count
    windows_per = dataset.eval_windows
    num_labels = model.config.num_labels

    agg = np.zeros((n_plasmids, num_labels), dtype=np.float64)
    counts = np.zeros(n_plasmids, dtype=np.int64)

    cursor = 0
    for batch in tqdm(loader, desc=f"eval [{device}]"):
        batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits.float().cpu().numpy()
        for i, row_logits in enumerate(logits):
            flat_idx = cursor + i
            plasmid_idx = flat_idx // windows_per
            if plasmid_idx < n_plasmids:
                agg[plasmid_idx] += row_logits
                counts[plasmid_idx] += 1
        cursor += len(logits)

    counts = np.clip(counts, 1, None)
    agg /= counts[:, None]
    y_true = dataset.labels
    return agg, y_true


def _metrics_dict(logits: np.ndarray, y_true: np.ndarray, label_names: list[str]) -> dict:
    """Compute top-1/top-3 accuracy, macro F1, and per-class report."""
    y_pred = logits.argmax(-1)
    num_labels = len(label_names)
    class_list = list(range(num_labels))
    return {
        "top1_accuracy": float(accuracy_score(y_true, y_pred)),
        "top3_accuracy": float(
            top_k_accuracy_score(y_true, logits, k=min(3, num_labels), labels=class_list)
        ),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class": classification_report(
            y_true, y_pred, target_names=label_names, zero_division=0, output_dict=True
        ),
    }


def _save_confusion_matrix(
    logits: np.ndarray,
    y_true: np.ndarray,
    label_names: list[str],
    out_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt

        y_pred = logits.argmax(-1)
        num_labels = len(label_names)
        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_labels)))
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(num_labels))
        ax.set_yticks(range(num_labels))
        ax.set_xticklabels(label_names, rotation=90)
        ax.set_yticklabels(label_names)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"Host-genus confusion matrix — {out_path.stem}")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[evaluate] confusion matrix saved → {out_path}")
    except Exception as e:
        print(f"[evaluate] confusion-matrix plot skipped: {e}")


def evaluate_checkpoint(
    checkpoint_dir: str | Path,
    processed_dir: str | Path,
    reports_dir: str | Path = "reports",
    window_size: int = 1_000,
    max_tokens: int = 512,
    eval_windows_per_plasmid: int = 8,
    batch_size: int = 32,
    run_baseline: bool = True,
    test_parquet: str | Path | None = None,
    device: str | None = None,
    subsample: int | None = None,
    seed: int = 42,
) -> dict:
    """Evaluate *checkpoint_dir* on a test split and (optionally) a k-mer baseline.

    Parameters
    ----------
    checkpoint_dir:
        Path to the fine-tuned model directory (must contain ``label_names.json``).
    processed_dir:
        Directory containing ``train.parquet`` / ``val.parquet`` / ``test.parquet``.
    reports_dir:
        Where to write ``test_metrics.json`` and ``confusion_matrix.png``.
    test_parquet:
        Override the test set.  Useful for ANI-clustered evaluation:
        pass the path to ``test_ani.parquet`` instead of the default ``test.parquet``.
    device:
        ``"cpu"`` or ``"cuda"``.  Auto-detected if *None*.
    subsample:
        If set, randomly sample this many plasmids from the test set.
        Useful for fast CPU evaluation (e.g. subsample=500).
    seed:
        Random seed for subsampling.
    """
    checkpoint_dir = Path(checkpoint_dir)
    processed_dir = Path(processed_dir)
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    label_names = json.loads((checkpoint_dir / "label_names.json").read_text())

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir), trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(checkpoint_dir), trust_remote_code=True
    )

    # Choose which test parquet to use
    if test_parquet is not None:
        _test_path = Path(test_parquet)
    else:
        _test_path = processed_dir / "test.parquet"

    test_ds = PlasmidWindowDataset(
        _test_path,
        tokenizer=tokenizer,
        window_size=window_size,
        max_tokens=max_tokens,
        mode="eval",
        eval_windows_per_plasmid=eval_windows_per_plasmid,
        subsample=subsample,
        seed=seed,
    )

    logits, y_true = _predict_plasmid_logits(
        model, test_ds, batch_size=batch_size, device=device
    )

    split_tag = _test_path.stem  # e.g. "test" or "test_ani"
    metrics = {"model": _metrics_dict(logits, y_true, label_names)}

    if run_baseline:
        metrics["baseline_kmer"] = kmer_baseline_eval(
            train_path=processed_dir / "train.parquet",
            test_path=_test_path,
            num_labels=len(label_names),
            subsample_test=subsample,
            seed=seed,
        )

    out_json = reports_dir / f"{split_tag}_metrics.json"
    out_json.write_text(json.dumps(metrics, indent=2))
    print(f"[evaluate] metrics saved → {out_json}")

    _save_confusion_matrix(
        logits, y_true, label_names,
        out_path=reports_dir / f"confusion_matrix_{split_tag}.png",
    )

    return metrics
