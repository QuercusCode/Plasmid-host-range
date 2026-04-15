"""Join PLSDB metadata with sequences, filter, label by genus, and write splits."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from Bio import SeqIO

from .splits import assert_no_group_leakage, group_split

# Candidate column names in PLSDB metadata TSV. We probe the first match that exists.
ACCESSION_CANDIDATES = ["NUCCORE_ACC", "ACC_NUCCORE", "accession", "Accession", "ACC"]
TAXON_CANDIDATES = ["TAXONOMY_name", "TAXONOMY_organism", "organism_name", "organism"]
GENUS_CANDIDATES = ["TAXONOMY_genus", "genus"]
SPECIES_CANDIDATES = ["TAXONOMY_species", "species"]


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


def _genus_from_taxon(name: str | float) -> str | None:
    if not isinstance(name, str) or not name.strip():
        return None
    # e.g. "Escherichia coli str. K-12" -> "Escherichia"
    return name.strip().split()[0]


def load_metadata(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)
    acc_col = _first_present(df, ACCESSION_CANDIDATES)
    if acc_col is None:
        raise RuntimeError(
            f"Could not find an accession column in {path}. "
            f"Tried: {ACCESSION_CANDIDATES}. Columns present: {list(df.columns)[:15]}..."
        )
    df = df.rename(columns={acc_col: "accession"})

    genus_col = _first_present(df, GENUS_CANDIDATES)
    if genus_col is not None:
        df["genus"] = df[genus_col]
    else:
        taxon_col = _first_present(df, TAXON_CANDIDATES)
        if taxon_col is None:
            raise RuntimeError(
                "Metadata has neither a genus nor a taxon/organism column to derive it from."
            )
        df["genus"] = df[taxon_col].map(_genus_from_taxon)

    species_col = _first_present(df, SPECIES_CANDIDATES)
    if species_col is not None:
        df["species"] = df[species_col]
    else:
        taxon_col = _first_present(df, TAXON_CANDIDATES)
        df["species"] = df[taxon_col] if taxon_col else df["genus"]

    return df[["accession", "genus", "species"]].dropna(subset=["accession"])


def load_sequences(fasta_path: Path) -> pd.DataFrame:
    rows = []
    for rec in SeqIO.parse(str(fasta_path), "fasta"):
        # FASTA id may be "NZ_CP012345.1" or similar; strip version for a loose match too
        acc = rec.id
        rows.append({"accession": acc, "accession_base": acc.split(".")[0], "sequence": str(rec.seq).upper()})
    return pd.DataFrame(rows)


def join_and_filter(meta: pd.DataFrame, seqs: pd.DataFrame, cfg: PreprocessConfig) -> pd.DataFrame:
    merged = seqs.merge(meta, on="accession", how="inner")
    if merged.empty:
        # Fall back to version-stripped match
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
    merged["genus"] = merged["genus"].str.strip()
    merged = merged[merged["genus"] != ""]
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

    fasta = next((p for p in [raw_dir / "plsdb.fna", raw_dir / "plsdb.fna.gz"] if p.exists()), None)
    meta = next((p for p in [raw_dir / "plsdb.tsv", raw_dir / "plsdb.tsv.gz"] if p.exists()), None)
    if fasta is None or meta is None:
        raise FileNotFoundError(
            f"Expected plsdb.fna and plsdb.tsv under {raw_dir}. Run `plasmid-host-range download` first."
        )

    print(f"[preprocess] loading metadata: {meta}")
    meta_df = load_metadata(meta)
    print(f"[preprocess] loading sequences: {fasta}")
    seq_df = load_sequences(fasta)
    print(f"[preprocess] metadata rows={len(meta_df)}, fasta rows={len(seq_df)}")

    merged = join_and_filter(meta_df, seq_df, cfg)
    print(f"[preprocess] after join+filter: {len(merged)} plasmids")

    labeled, label_names = assign_labels(merged, cfg.top_n_genera, cfg.other_label)
    print(f"[preprocess] label classes ({len(label_names)}): {label_names}")

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

    label_path = out_dir / "label_names.json"
    label_path.write_text(json.dumps(label_names, indent=2))
    return {"label_names": label_names, "splits": {k: len(v) for k, v in splits.items()}}
