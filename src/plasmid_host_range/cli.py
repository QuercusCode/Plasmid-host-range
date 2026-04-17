"""Command-line interface: ``plasmid-host-range {download,preprocess,train,evaluate,predict}``."""
from __future__ import annotations

import json
from pathlib import Path

import typer
from rich import print

from .data.download import DEFAULT_ARTICLE_ID, download_plsdb
from .data.preprocess import PreprocessConfig, preprocess as run_preprocess

app = typer.Typer(help="Predict plasmid host genus from nucleotide sequence (PLSDB).")


@app.command()
def download(
    out_dir: Path = typer.Option(Path("data/raw"), help="Where to save raw PLSDB files."),
    article_id: int = typer.Option(
        DEFAULT_ARTICLE_ID, help="Figshare article ID. Default is PLSDB 2024_05_31_v2."
    ),
    no_decompress: bool = typer.Option(
        False, "--no-decompress", help="Skip .bz2 decompression (useful for slow disks)."
    ),
) -> None:
    """Download PLSDB FASTA + metadata from Figshare."""
    result = download_plsdb(out_dir, article_id=article_id, decompress=not no_decompress)
    for name, path in result.items():
        print(f"[green]{name}:[/green] {path}")


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
def ani_validate(
    checkpoint: Path = typer.Option(..., help="Path to fine-tuned checkpoint directory."),
    processed_dir: Path = typer.Option(Path("data/processed")),
    reports_dir: Path = typer.Option(Path("reports")),
    window_size: int = typer.Option(3_000),
    max_tokens: int = typer.Option(512),
    eval_windows: int = typer.Option(5, help="Eval windows per plasmid (fewer = faster on CPU)."),
    batch_size: int = typer.Option(8, help="Use 8 for CPU, 32+ for GPU."),
    subsample: int = typer.Option(None, help="Randomly sample N plasmids for fast CPU eval."),
    device: str = typer.Option(None, help="'cpu' or 'cuda'. Auto-detected if omitted."),
    no_baseline: bool = typer.Option(False, "--no-baseline"),
    seed: int = typer.Option(42),
) -> None:
    """Build an ANI-clustered test split and evaluate the model on it.

    This is the publication-grade validation: no plasmid in the test set shares
    ≥95% ANI with any training plasmid, ruling out plasmid-family memorisation.
    Requires:  pip install datasketch
    """
    from .data.ani_cluster import save_ani_test_split
    from .evaluate import evaluate_checkpoint

    print("[bold]Step 1/2:[/bold] Computing ANI clusters and creating test_ani.parquet …")
    print("  (This may take 30–60 minutes on CPU for 70k plasmids — grab a coffee.)")
    test_ani_path = save_ani_test_split(processed_dir, seed=seed)
    print(f"[green]ANI test split:[/green] {test_ani_path}")

    print("\n[bold]Step 2/2:[/bold] Evaluating model on ANI test split …")
    metrics = evaluate_checkpoint(
        checkpoint_dir=checkpoint,
        processed_dir=processed_dir,
        reports_dir=reports_dir,
        window_size=window_size,
        max_tokens=max_tokens,
        eval_windows_per_plasmid=eval_windows,
        batch_size=batch_size,
        run_baseline=not no_baseline,
        test_parquet=test_ani_path,
        device=device,
        subsample=subsample,
        seed=seed,
    )

    print("\n[bold]ANI VALIDATION RESULTS[/bold]")
    m = metrics["model"]
    print(f"  Top-1 accuracy : {m['top1_accuracy']:.3f}")
    print(f"  Top-3 accuracy : {m['top3_accuracy']:.3f}")
    print(f"  Macro F1       : {m['macro_f1']:.3f}")
    if "baseline_kmer" in metrics:
        b = metrics["baseline_kmer"]
        delta = m["macro_f1"] - b["macro_f1"]
        print(f"\n  Baseline macro F1  : {b['macro_f1']:.3f}")
        print(f"  DL improvement     : {delta:+.3f}")
    print(f"\n[green]Full results:[/green] {reports_dir}/test_ani_metrics.json")


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
