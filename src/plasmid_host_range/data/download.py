"""Download PLSDB from Figshare.

PLSDB is distributed through Figshare as a dataset with many files. We only need a
small subset for host-genus prediction (sequences FASTA + a couple of metadata CSVs),
so we hit the public Figshare API, pick the files we need by name, and stream them
to ``data/raw/``.

Default version: PLSDB 2024_05_31_v2 (article 27252609). You can override with
``--article-id`` if a newer release appears.
"""
from __future__ import annotations

from pathlib import Path
import shutil
import sys

import requests
from tqdm import tqdm

# https://figshare.com/articles/dataset/PLSDB_2024_05_31_v2/27252609
DEFAULT_ARTICLE_ID = 27252609
FIGSHARE_API = "https://api.figshare.com/v2/articles/{article_id}/files"

# Files we pull by default. Everything else on the Figshare article is skipped.
REQUIRED_FILES: tuple[str, ...] = (
    "sequences.fasta.bz2",  # 1.91 GB — plasmid nucleotide sequences
    "nuccore.csv",          # per-plasmid metadata incl. taxon ID
    "taxonomy.csv",         # taxon ID -> lineage (genus, species, …)
    "biosample.csv",        # host info from biosamples (optional fallback)
    "README.md",            # column documentation, handy to keep around
)


def _list_figshare_files(article_id: int) -> list[dict]:
    """Return every file in the Figshare article, paging through the API."""
    url = FIGSHARE_API.format(article_id=article_id)
    all_files: list[dict] = []
    page = 1
    page_size = 1000  # Figshare's documented max
    while True:
        r = requests.get(url, params={"page": page, "page_size": page_size}, timeout=60)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        all_files.extend(batch)
        if len(batch) < page_size:
            break
        page += 1
    return all_files


def _stream_download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with tmp.open("wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name
        ) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
    tmp.replace(dest)


def _decompress_bz2(path: Path) -> Path:
    """Decompress foo.bz2 -> foo (in-place), removing the .bz2 file."""
    import bz2

    out = path.with_suffix("")
    if out.exists() and out.stat().st_size > 0:
        return out
    print(f"[download] decompressing {path.name} -> {out.name}", file=sys.stderr)
    with bz2.open(path, "rb") as src, out.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=1 << 20)
    path.unlink()
    return out


def download_plsdb(
    out_dir: Path,
    article_id: int = DEFAULT_ARTICLE_ID,
    files: tuple[str, ...] = REQUIRED_FILES,
    decompress: bool = True,
) -> dict[str, Path]:
    """Download the PLSDB files we need from Figshare into ``out_dir``.

    Returns a dict mapping each requested filename (post-decompression) to its local path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[download] listing Figshare article {article_id}", file=sys.stderr)
    index = _list_figshare_files(article_id)
    by_name = {entry["name"]: entry for entry in index}

    missing = [f for f in files if f not in by_name]
    if missing:
        available = sorted(by_name.keys())
        raise RuntimeError(
            f"Figshare article {article_id} does not contain: {missing}. "
            f"Available files: {available}"
        )

    out: dict[str, Path] = {}
    for name in files:
        entry = by_name[name]
        local = out_dir / name
        final = local.with_suffix("") if (decompress and name.endswith(".bz2")) else local

        if final.exists() and final.stat().st_size > 0:
            print(f"[download] {final.name} already present, skipping", file=sys.stderr)
            out[name] = final
            continue

        url = entry["download_url"]
        size_mb = entry.get("size", 0) / 1e6
        print(f"[download] {name}  ({size_mb:.1f} MB)", file=sys.stderr)
        _stream_download(url, local)

        if decompress and name.endswith(".bz2"):
            final = _decompress_bz2(local)
        out[name] = final

    return out
