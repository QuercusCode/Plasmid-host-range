"""Download PLSDB 2025 nucleotide FASTA and metadata TSV.

The PLSDB 2025 release is hosted at https://ccb-microbe.cs.uni-saarland.de/plsdb2025/.
Exact download URLs for bulk files may change between releases; override with the
``--fasta-url`` / ``--metadata-url`` CLI flags if the defaults 404.
"""
from __future__ import annotations

from pathlib import Path
import shutil
import sys

import requests
from tqdm import tqdm

# Best-effort defaults. If PLSDB 2025 moves these files, pass explicit URLs instead.
DEFAULT_FASTA_URL = "https://ccb-microbe.cs.uni-saarland.de/plsdb2025/download/plsdb.fna.bz2"
DEFAULT_METADATA_URL = "https://ccb-microbe.cs.uni-saarland.de/plsdb2025/download/plsdb.tsv"


def _stream_download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with dest.open("wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name
        ) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))


def _maybe_decompress(path: Path) -> Path:
    """If ``path`` is .bz2 or .gz, decompress in place and return the new path."""
    if path.suffix == ".bz2":
        import bz2

        out = path.with_suffix("")
        with bz2.open(path, "rb") as src, out.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        path.unlink()
        return out
    if path.suffix == ".gz":
        import gzip

        out = path.with_suffix("")
        with gzip.open(path, "rb") as src, out.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        path.unlink()
        return out
    return path


def download_plsdb(
    out_dir: Path,
    fasta_url: str = DEFAULT_FASTA_URL,
    metadata_url: str = DEFAULT_METADATA_URL,
) -> tuple[Path, Path]:
    """Download PLSDB FASTA + metadata into ``out_dir``. Returns (fasta_path, metadata_path)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fasta_raw = out_dir / Path(fasta_url).name
    meta_raw = out_dir / Path(metadata_url).name

    if not fasta_raw.exists() and not fasta_raw.with_suffix("").exists():
        print(f"[download] {fasta_url}", file=sys.stderr)
        _stream_download(fasta_url, fasta_raw)
    fasta_path = _maybe_decompress(fasta_raw) if fasta_raw.exists() else fasta_raw.with_suffix("")

    if not meta_raw.exists():
        print(f"[download] {metadata_url}", file=sys.stderr)
        _stream_download(metadata_url, meta_raw)
    meta_path = _maybe_decompress(meta_raw)

    return fasta_path, meta_path
