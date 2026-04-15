"""Model + tokenizer loading. Thin wrapper around HuggingFace AutoModel classes."""
from __future__ import annotations

from dataclasses import dataclass

from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class LoadedModel:
    tokenizer: object
    model: object
    num_labels: int


def load_model(
    model_name: str,
    num_labels: int,
    trust_remote_code: bool = True,
) -> LoadedModel:
    """Load tokenizer + sequence-classification head for a DNA foundation model."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        trust_remote_code=trust_remote_code,
    )
    return LoadedModel(tokenizer=tokenizer, model=model, num_labels=num_labels)
