"""Unit tests for the preprocessing helpers (no network, no model downloads)."""
from __future__ import annotations

import pandas as pd

from plasmid_host_range.data.preprocess import assign_labels, join_and_filter, PreprocessConfig
from plasmid_host_range.data.splits import (
    assert_no_group_leakage,
    compute_class_weights,
    group_split,
)


def _toy_merged() -> pd.DataFrame:
    rows = []
    genera = ["Escherichia"] * 20 + ["Klebsiella"] * 15 + ["Salmonella"] * 10 + ["Rarebug"] * 3
    for i, g in enumerate(genera):
        rows.append(
            {
                "accession": f"ACC{i:04d}",
                "sequence": "ACGT" * 500,  # 2 kb
                "genus": g,
                "species": f"{g} sp{i % 5}",
            }
        )
    return pd.DataFrame(rows)


def test_assign_labels_collapses_tail_into_other():
    df = _toy_merged()
    labeled, names = assign_labels(df, top_n=2, other_label="Other")
    assert names[:2] == ["Escherichia", "Klebsiella"]
    assert names[-1] == "Other"
    assert (labeled[labeled["genus"] == "Salmonella"]["label_name"] == "Other").all()
    assert labeled["label"].between(0, len(names) - 1).all()


def test_group_split_has_no_species_leakage():
    df = _toy_merged()
    splits = group_split(df, group_col="species", val_frac=0.2, test_frac=0.2, seed=1)
    assert_no_group_leakage(splits, "species")
    total = sum(len(p) for p in splits.values())
    assert total == len(df)


def test_compute_class_weights_inverse_frequency():
    labels = [0, 0, 0, 0, 1, 1, 2]
    w = compute_class_weights(labels=__import__("numpy").array(labels), num_classes=3)
    # Rarer classes should have higher weight
    assert w[2] > w[1] > w[0]


def test_join_and_filter_drops_short_and_missing():
    meta = pd.DataFrame(
        [
            {"accession": "A", "genus": "Escherichia", "species": "E. coli"},
            {"accession": "B", "genus": None, "species": None},
            {"accession": "C", "genus": "Klebsiella", "species": "K. pneumoniae"},
        ]
    )
    seqs = pd.DataFrame(
        [
            {"accession": "A", "accession_base": "A", "sequence": "ACGT" * 500},  # 2 kb, kept
            {"accession": "B", "accession_base": "B", "sequence": "ACGT" * 500},  # dropped: no genus
            {"accession": "C", "accession_base": "C", "sequence": "ACGT" * 10},   # dropped: too short
        ]
    )
    cfg = PreprocessConfig(raw_dir=".", processed_dir=".", min_len=1_000, max_len=500_000)
    out = join_and_filter(meta, seqs, cfg)
    assert list(out["accession"]) == ["A"]
