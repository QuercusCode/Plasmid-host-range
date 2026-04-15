# plasmid-host-range

Predict the bacterial **host genus** of a plasmid directly from its nucleotide sequence, by fine-tuning a pretrained DNA foundation model (default: DNABERT-2) on **PLSDB 2025**.

The headline public API is:

```python
from plasmid_host_range.predict import predict_host_genus

preds = predict_host_genus("path/to/plasmid.fasta", top_k=3)
for p in preds:
    print(p.accession, p.top_genera, p.scores)
```

## Setup

```bash
cd ml/plasmid-host-range
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Or with `uv`:

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Data

PLSDB 2025 is hosted at <https://ccb-microbe.cs.uni-saarland.de/plsdb2025/>. Download the
nucleotide FASTA and the metadata TSV into `data/raw/`:

```bash
plasmid-host-range download
```

> The exact filenames on the PLSDB 2025 site may change between releases. If the default URLs
> in `src/plasmid_host_range/data/download.py` 404, pass `--fasta-url` and `--metadata-url`
> explicitly, or drop the files into `data/raw/plsdb.fna` and `data/raw/plsdb.tsv` manually.

Then build the processed splits:

```bash
plasmid-host-range preprocess --top-n-genera 20
```

This writes `data/processed/{train,val,test}.parquet` with columns
`{accession, sequence, label, genus}`, using a **host-species-grouped** train/val/test split
to avoid leaking near-identical plasmids across splits.

## Train

Smoke test (tiny subset, CPU-friendly):

```bash
plasmid-host-range train --config configs/smoke.yaml
```

Full fine-tune (GPU recommended):

```bash
plasmid-host-range train --config configs/default.yaml
```

## Evaluate

```bash
plasmid-host-range evaluate --checkpoint checkpoints/best
```

Produces `reports/test_metrics.json` and `reports/confusion_matrix.png`. A k-mer + logistic
regression baseline is reported alongside, so any DL improvement is quantified rather than
assumed.

## Predict on a new plasmid

```bash
plasmid-host-range predict path/to/plasmid.fasta --top-k 3
```

## Layout

```
src/plasmid_host_range/
  data/         # download, preprocess, splits, torch Dataset
  model.py      # HF AutoModelForSequenceClassification wrapper
  train.py      # fine-tuning loop
  evaluate.py   # metrics + confusion matrix
  predict.py    # predict_host_genus() public API
  baselines.py  # k-mer + logistic regression baseline
  cli.py        # typer CLI
tests/          # smoke + unit tests
configs/        # default.yaml, smoke.yaml
```
