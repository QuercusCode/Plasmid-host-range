"""Join PLSDB metadata with sequences, label by host genus, and write splits.

PLSDB (Figshare 27252609, version 2024_05_31_v2) ships metadata as several CSV files.
For host-genus prediction we only need two:

* ``nuccore.csv``    — one row per plasmid (accession, length, taxon ID, …)
* ``taxonomy.csv``   — taxon ID to full lineage (superkingdom … species/strain)

This module joins them on taxon ID, pulls the genus, then joins to the FASTA sequences
on accession, filters by length, and writes train/val/test parquet splits.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from Bio import SeqIO

from .splits import assert_no_group_leakage, group_split

# Column-name probe order. PLSDB has had minor schema tweaks over releases, so each
# logical field lists several candidates and we pick the first one present.
ACCESSION_CANDIDATES = [
    "NUCCORE_ACC", "NUCCORE_Accession", "ACC_NUCCORE", "accession", "Accession", "ACC",
]
NUCCORE_TAXON_ID_CANDIDATES = [
    "TAXONOMY_UID", "NUCCORE_TaxonID", "TAXONOMY_TaxonID", "TaxonID", "taxon_id", "Taxid",
]
TAXONOMY_ID_CANDIDATES = [
    "TAXONOMY_UID", "TAXONOMY_TaxonID", "TaxonID", "taxon_id", "Taxid",
]
GENUS_CANDIDATES = ["TAXONOMY_genus", "genus", "Genus"]
SPECIES_CANDIDATES = ["TAXONOMY_species", "species", "Species"]


@dataclass
class PreprocessConfig:
    raw_dir: Path
    processed_dir: Path
    top_n_genera: int = 20
    other_label: str = "Other"
    min_len: int = 1_000
    max_len: int = 500_000
    val_frac: float = 0.1
    test_frac: float = 0.1
    seed: int = 42


def _first_present(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _require(df: pd.DataFrame, candidates: list[str], label: str, filename: str) -> str:
    col = _first_present(df, candidates)
    if col is None:
        raise RuntimeError(
            f"Could not find a {label} column in {filename}. Tried: {candidates}. "
            f"Columns present: {list(df.columns)}"
        )
    return col


def _read_table(path: Path) -> pd.DataFrame:
    """Read a PLSDB metadata file. Despite the .csv extension some releases use tabs,
    so we sniff the separator from the header line."""
    with path.open("rb") as f:
        head = f.read(4096).decode("utf-8", errors="replace")
    sep = "\t" if head.count("\t") > head.count(",") else ","
    return pd.read_csv(path, sep=sep, dtype=str, low_memory=False)


def load_metadata(raw_dir: Path) -> pd.DataFrame:
    """Return a dataframe with one row per plasmid and columns [accession, genus, species]."""
    nuccore_path = raw_dir / "nuccore.csv"
    taxonomy_path = raw_dir / "taxonomy.csv"
    if not nuccore_path.exists() or not taxonomy_path.exists():
        raise FileNotFoundError(
            f"Expected nuccore.csv and taxonomy.csv under {raw_dir}. "
            f"Run `plasmid-host-range download` first."
        )

    nuc = _read_table(nuccore_path)
    tax = _read_table(taxonomy_path)

    acc_col = _require(nuc, ACCESSION_CANDIDATES, "accession", "nuccore.csv")
    nuc_tax_col = _require(nuc, NUCCORE_TAXON_ID_CANDIDATES, "taxon-id", "nuccore.csv")
    tax_id_col = _require(tax, TAXONOMY_ID_CANDIDATES, "taxon-id", "taxonomy.csv")
    genus_col = _require(tax, GENUS_CANDIDATES, "genus", "taxonomy.csv")
    species_col = _first_present(tax, SPECIES_CANDIDATES)

    nuc = nuc.rename(columns={acc_col: "accession", nuc_tax_col: "taxon_id"})
    keep_tax_cols = {tax_id_col: "taxon_id", genus_col: "genus"}
    if species_col is not None:
        keep_tax_cols[species_col] = "species"
    tax = tax[list(keep_tax_cols.keys())].rename(columns=keep_tax_cols)
    if "species" not in tax.columns:
        tax["species"] = None

    merged = nuc[["accession", "taxon_id"]].merge(tax, on="taxon_id", how="left")
    merged = merged.dropna(subset=["accession", "genus"])
    merged["genus"] = merged["genus"].astype(str).str.strip()
    merged = merged[merged["genus"] != ""]
    # For species-grouped splitting, fall back to genus if species missing
    merged["species"] = merged["species"].fillna(merged["genus"]).astype(str).str.strip()
    return merged[["accession", "genus", "species"]].reset_index(drop=True)


def load_sequences(fasta_path: Path) -> pd.DataFrame:
    """Read the PLSDB FASTA. Accessions are versioned (e.g. ``NZ_CP012345.1``)."""
    rows = []
    for rec in SeqIO.parse(str(fasta_path), "fasta"):
        rows.append(
            {
                "accession": rec.id,
                "accession_base": rec.id.split(".")[0],
                "sequence": str(rec.seq).upper(),
            }
        )
    return pd.DataFrame(rows)


def join_and_filter(meta: pd.DataFrame, seqs: pd.DataFrame, cfg: PreprocessConfig) -> pd.DataFrame:
    merged = seqs.merge(meta, on="accession", how="inner")
    if merged.empty:
        # Fall back to version-stripped match if accessions disagree on the .1 suffix
        meta2 = meta.copy()
        meta2["accession_base"] = meta2["accession"].str.split(".").str[0]
        merged = seqs.merge(
            meta2.drop(columns=["accession"]),
            on="accession_base",
            how="inner",
        )
        merged = merged.rename(columns={"accession_base": "accession"})
    merged = merged.dropna(subset=["sequence", "genus"])
    merged = merged[merged["sequence"].str.len().between(cfg.min_len, cfg.max_len)]
    merged = merged.drop_duplicates(subset=["accession"])
    return merged.reset_index(drop=True)


def assign_labels(df: pd.DataFrame, top_n: int, other_label: str) -> tuple[pd.DataFrame, list[str]]:
    counts = df["genus"].value_counts()
    top = counts.head(top_n).index.tolist()
    label_names = top + [other_label]
    name_to_id = {n: i for i, n in enumerate(label_names)}
    df = df.copy()
    df["label_name"] = df["genus"].where(df["genus"].isin(top), other_label)
    df["label"] = df["label_name"].map(name_to_id).astype("int64")
    return df, label_names


def preprocess(cfg: PreprocessConfig) -> dict:
    raw_dir = Path(cfg.raw_dir)
    out_dir = Path(cfg.processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # FASTA is typically named sequences.fasta after decompression.
    fasta_candidates = [
        raw_dir / "sequences.fasta",
        raw_dir / "sequences.fna",
        raw_dir / "plsdb.fna",
    ]
    fasta = next((p for p in fasta_candidates if p.exists()), None)
    if fasta is None:
        raise FileNotFoundError(
            f"No plasmid FASTA found under {raw_dir}. Tried: "
            f"{[p.name for p in fasta_candidates]}. Run `plasmid-host-range download` first."
        )

    print(f"[preprocess] loading metadata from {raw_dir}")
    meta_df = load_metadata(raw_dir)
    print(f"[preprocess] loading sequences: {fasta}")
    seq_df = load_sequences(fasta)
    print(f"[preprocess] metadata rows={len(meta_df)}, fasta rows={len(seq_df)}")

    merged = join_and_filter(meta_df, seq_df, cfg)
    print(f"[preprocess] after join+filter: {len(merged)} plasmids")
    if merged.empty:
        raise RuntimeError(
            "Join produced zero rows. Likely an accession-format mismatch between "
            "nuccore.csv and the FASTA headers, or an unexpected schema change. "
            "Inspect `head data/raw/nuccore.csv` and the first FASTA header, and share them."
        )

    labeled, label_names = assign_labels(merged, cfg.top_n_genera, cfg.other_label)
    print(f"[preprocess] label classes ({len(label_names)}): {label_names}")
    print(f"[preprocess] class counts:\n{labeled['label_name'].value_counts()}")

    splits = group_split(
        labeled,
        group_col="species",
        val_frac=cfg.val_frac,
        test_frac=cfg.test_frac,
        seed=cfg.seed,
    )
    assert_no_group_leakage(splits, "species")
    for name, part in splits.items():
        path = out_dir / f"{name}.parquet"
        part[["accession", "sequence", "label", "genus", "species"]].to_parquet(path, index=False)
        print(f"[preprocess] wrote {path} ({len(part)} rows)")

    (out_dir / "label_names.json").write_text(json.dumps(label_names, indent=2))
    return {"label_names": label_names, "splits": {k: len(v) for k, v in splits.items()}}
