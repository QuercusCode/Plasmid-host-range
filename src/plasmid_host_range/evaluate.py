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
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    n_plasmids = dataset.plasmid_count
    windows_per = dataset.eval_windows
    num_labels = model.config.num_labels

    agg = np.zeros((n_plasmids, num_labels), dtype=np.float64)
    counts = np.zeros(n_plasmids, dtype=np.int64)

    cursor = 0
    for batch in tqdm(loader, desc="eval"):
        labels = batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits.float().cpu().numpy()
        for i, row_logits in enumerate(logits):
            flat_idx = cursor + i
            plasmid_idx = flat_idx // windows_per
            agg[plasmid_idx] += row_logits
            counts[plasmid_idx] += 1
        cursor += len(logits)

    counts = np.clip(counts, 1, None)
    agg /= counts[:, None]
    y_true = dataset.labels
    return agg, y_true


def evaluate_checkpoint(
    checkpoint_dir: str | Path,
    processed_dir: str | Path,
    reports_dir: str | Path = "reports",
    window_size: int = 1_000,
    max_tokens: int = 512,
    eval_windows_per_plasmid: int = 8,
    batch_size: int = 32,
    run_baseline: bool = True,
) -> dict:
    checkpoint_dir = Path(checkpoint_dir)
    processed_dir = Path(processed_dir)
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    label_names = json.loads((checkpoint_dir / "label_names.json").read_text())

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir), trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(checkpoint_dir), trust_remote_code=True
    )

    test_ds = PlasmidWindowDataset(
        processed_dir / "test.parquet",
        tokenizer=tokenizer,
        window_size=window_size,
        max_tokens=max_tokens,
        mode="eval",
        eval_windows_per_plasmid=eval_windows_per_plasmid,
    )

    logits, y_true = _predict_plasmid_logits(model, test_ds, batch_size=batch_size)
    y_pred = logits.argmax(-1)

    num_labels = len(label_names)
    class_list = list(range(num_labels))
    metrics = {
        "model": {
            "top1_accuracy": float(accuracy_score(y_true, y_pred)),
            "top3_accuracy": float(
                top_k_accuracy_score(y_true, logits, k=min(3, num_labels), labels=class_list)
            ),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "per_class": classification_report(
                y_true, y_pred, target_names=label_names, zero_division=0, output_dict=True
            ),
        }
    }

    if run_baseline:
        metrics["baseline_kmer"] = kmer_baseline_eval(
            train_path=processed_dir / "train.parquet",
            test_path=processed_dir / "test.parquet",
            num_labels=num_labels,
        )

    (reports_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2))

    try:
        import matplotlib.pyplot as plt

        cm = confusion_matrix(y_true, y_pred, labels=class_list)
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(num_labels))
        ax.set_yticks(range(num_labels))
        ax.set_xticklabels(label_names, rotation=90)
        ax.set_yticklabels(label_names)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title("Host-genus confusion matrix")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(reports_dir / "confusion_matrix.png", dpi=150)
        plt.close(fig)
    except Exception as e:  # plotting is best-effort
        print(f"[evaluate] confusion-matrix plot skipped: {e}")

    return metrics
