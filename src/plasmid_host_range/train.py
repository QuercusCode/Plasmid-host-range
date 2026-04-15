"""Fine-tuning loop. Uses HuggingFace Trainer for simplicity."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, f1_score
from transformers import Trainer, TrainingArguments

from .data.dataset import PlasmidWindowDataset
from .data.splits import compute_class_weights
from .model import load_model


def _load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _compute_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.asarray(logits).argmax(-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
    }


class _WeightedTrainer(Trainer):
    """Trainer variant that applies class-weighted cross-entropy."""

    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        weight = self._class_weights.to(logits.device) if self._class_weights is not None else None
        loss = nn.functional.cross_entropy(logits, labels, weight=weight)
        return (loss, outputs) if return_outputs else loss


@dataclass
class TrainResult:
    output_dir: Path
    metrics: dict[str, float]
    label_names: list[str]


def train_from_config(config_path: str | Path) -> TrainResult:
    cfg = _load_config(config_path)
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    processed = Path(data_cfg["processed_dir"])
    label_names = json.loads((processed / "label_names.json").read_text())
    num_labels = len(label_names)

    loaded = load_model(
        cfg["model_name"],
        num_labels=num_labels,
        trust_remote_code=cfg.get("trust_remote_code", True),
    )

    train_ds = PlasmidWindowDataset(
        processed / "train.parquet",
        tokenizer=loaded.tokenizer,
        window_size=data_cfg["window_size"],
        max_tokens=data_cfg["max_tokens"],
        mode="train",
        subsample=data_cfg.get("subsample_train"),
        seed=train_cfg.get("seed", 0),
    )
    val_ds = PlasmidWindowDataset(
        processed / "val.parquet",
        tokenizer=loaded.tokenizer,
        window_size=data_cfg["window_size"],
        max_tokens=data_cfg["max_tokens"],
        mode="eval",
        eval_windows_per_plasmid=data_cfg["eval_windows_per_plasmid"],
        subsample=data_cfg.get("subsample_eval"),
        seed=train_cfg.get("seed", 0),
    )

    class_weights = None
    if train_cfg.get("class_weighted_loss", False):
        w = compute_class_weights(train_ds.labels, num_labels)
        class_weights = torch.tensor(w, dtype=torch.float32)

    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 0.0),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.0),
        eval_strategy=train_cfg.get("eval_strategy", "epoch"),
        save_strategy=train_cfg.get("save_strategy", "epoch"),
        save_total_limit=train_cfg.get("save_total_limit", 2),
        load_best_model_at_end=train_cfg.get("load_best_model_at_end", False),
        metric_for_best_model=train_cfg.get("metric_for_best_model", "macro_f1"),
        greater_is_better=train_cfg.get("greater_is_better", True),
        fp16=train_cfg.get("fp16", False),
        seed=train_cfg.get("seed", 42),
        report_to=[],
        logging_steps=20,
    )

    trainer = _WeightedTrainer(
        model=loaded.model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=_compute_metrics,
        class_weights=class_weights,
    )

    trainer.train()
    metrics = trainer.evaluate()

    best_dir = output_dir / "best"
    trainer.save_model(str(best_dir))
    loaded.tokenizer.save_pretrained(str(best_dir))
    (best_dir / "label_names.json").write_text(json.dumps(label_names, indent=2))
    (best_dir / "val_metrics.json").write_text(json.dumps(metrics, indent=2))

    return TrainResult(output_dir=best_dir, metrics=metrics, label_names=label_names)
