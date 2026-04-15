"""End-to-end smoke test for predict_host_genus.

Uses a tiny locally-constructed HuggingFace model (not downloaded) so the test runs
offline in a few seconds and exercises the full chunk -> tokenize -> aggregate -> top-k
path without needing a real fine-tuned checkpoint.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

transformers = pytest.importorskip("transformers")
torch = pytest.importorskip("torch")

from transformers import AutoTokenizer, BertConfig, BertForSequenceClassification  # noqa: E402

from plasmid_host_range.predict import predict_host_genus  # noqa: E402


@pytest.fixture
def tiny_checkpoint(tmp_path: Path) -> Path:
    """Build a minimal Bert-style classifier + a char-level tokenizer and save to disk."""
    # Use a tiny pre-existing tokenizer as a stand-in (character-ish vocab is fine for a smoke test)
    tok = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-bert")
    config = BertConfig(
        vocab_size=tok.vocab_size,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=128,
        num_labels=3,
    )
    model = BertForSequenceClassification(config)

    out = tmp_path / "ckpt"
    out.mkdir()
    model.save_pretrained(out)
    tok.save_pretrained(out)
    (out / "label_names.json").write_text(json.dumps(["Escherichia", "Klebsiella", "Other"]))
    return out


def test_predict_host_genus_on_raw_string(tiny_checkpoint: Path):
    seq = "ACGT" * 400  # 1.6 kb
    preds = predict_host_genus(
        seq,
        model_dir=tiny_checkpoint,
        top_k=2,
        window_size=500,
        max_tokens=64,
        batch_size=4,
    )
    assert len(preds) == 1
    p = preds[0]
    assert p.accession == "query"
    assert len(p.top_genera) == 2
    assert len(p.scores) == 2
    assert all(0.0 <= s <= 1.0 for s in p.scores)
    # scores are a softmax subset; they sum to <= 1
    assert sum(p.scores) <= 1.0 + 1e-6


def test_predict_host_genus_on_fasta(tiny_checkpoint: Path, tmp_path: Path):
    fasta = tmp_path / "toy.fasta"
    fasta.write_text(">plasmid_A\n" + ("ACGT" * 300) + "\n>plasmid_B\n" + ("GCTA" * 300) + "\n")
    preds = predict_host_genus(
        fasta,
        model_dir=tiny_checkpoint,
        top_k=3,
        window_size=400,
        max_tokens=64,
    )
    assert [p.accession for p in preds] == ["plasmid_A", "plasmid_B"]
    for p in preds:
        assert len(p.top_genera) == 3
