"""Command-line interface: ``plasmid-host-range {download,preprocess,train,evaluate,predict}``."""
from __future__ import annotations

import json
from pathlib import Path

import typer
from rich import print

from .data.download import DEFAULT_FASTA_URL, DEFAULT_METADATA_URL, download_plsdb
from .data.preprocess import PreprocessConfig, preprocess as run_preprocess

app = typer.Typer(help="Predict plasmid host genus from nucleotide sequence (PLSDB 2025).")


@app.command()
def download(
    out_dir: Path = typer.Option(Path("data/raw"), help="Where to save raw PLSDB files."),
    fasta_url: str = typer.Option(DEFAULT_FASTA_URL, help="Override FASTA URL."),
    metadata_url: str = typer.Option(DEFAULT_METADATA_URL, help="Override metadata TSV URL."),
) -> None:
    """Download PLSDB 2025 nucleotide FASTA and metadata TSV."""
    fasta, meta = download_plsdb(out_dir, fasta_url=fasta_url, metadata_url=metadata_url)
    print(f"[green]fasta:[/green] {fasta}")
    print(f"[green]metadata:[/green] {meta}")


@app.command()
def preprocess(
    raw_dir: Path = typer.Option(Path("data/raw")),
    processed_dir: Path = typer.Option(Path("data/processed")),
    top_n_genera: int = typer.Option(20),
    min_len: int = typer.Option(1_000),
    max_len: int = typer.Option(500_000),
    val_frac: float = typer.Option(0.1),
    test_frac: float = typer.Option(0.1),
    seed: int = typer.Option(42),
) -> None:
    """Join metadata + sequences, filter, label by genus, and write train/val/test splits."""
    cfg = PreprocessConfig(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        top_n_genera=top_n_genera,
        min_len=min_len,
        max_len=max_len,
        val_frac=val_frac,
        test_frac=test_frac,
        seed=seed,
    )
    result = run_preprocess(cfg)
    print(result)


@app.command()
def train(config: Path = typer.Option(..., help="Path to YAML training config.")) -> None:
    """Fine-tune a DNA foundation model on the processed splits."""
    from .train import train_from_config

    result = train_from_config(config)
    print(f"[green]best checkpoint:[/green] {result.output_dir}")
    print(result.metrics)


@app.command()
def evaluate(
    checkpoint: Path = typer.Option(..., help="Path to fine-tuned checkpoint directory."),
    processed_dir: Path = typer.Option(Path("data/processed")),
    reports_dir: Path = typer.Option(Path("reports")),
    window_size: int = typer.Option(1_000),
    max_tokens: int = typer.Option(512),
    eval_windows_per_plasmid: int = typer.Option(8),
    batch_size: int = typer.Option(32),
    no_baseline: bool = typer.Option(False, "--no-baseline", help="Skip the k-mer baseline."),
) -> None:
    """Evaluate a fine-tuned checkpoint on the test split."""
    from .evaluate import evaluate_checkpoint

    metrics = evaluate_checkpoint(
        checkpoint_dir=checkpoint,
        processed_dir=processed_dir,
        reports_dir=reports_dir,
        window_size=window_size,
        max_tokens=max_tokens,
        eval_windows_per_plasmid=eval_windows_per_plasmid,
        batch_size=batch_size,
        run_baseline=not no_baseline,
    )
    print(json.dumps(metrics, indent=2))


@app.command()
def predict(
    sequence_or_fasta: str = typer.Argument(..., help="A FASTA path or a raw DNA string."),
    model_dir: Path = typer.Option(Path("checkpoints/default/best")),
    top_k: int = typer.Option(3),
    window_size: int = typer.Option(1_000),
    max_tokens: int = typer.Option(512),
) -> None:
    """Predict host genus for one or more plasmids."""
    from .predict import predict_host_genus

    preds = predict_host_genus(
        sequence_or_fasta,
        model_dir=model_dir,
        top_k=top_k,
        window_size=window_size,
        max_tokens=max_tokens,
    )
    print(json.dumps([p.to_dict() for p in preds], indent=2))


if __name__ == "__main__":
    app()
