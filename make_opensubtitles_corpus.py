"""Build data/opensubtitles.txt from real OPUS-OpenSubtitles parallel data.

Source: OPUS-OpenSubtitles v2018 (https://opus.nlpl.eu/OpenSubtitles.php).
The OpenSubtitles 2018 release is the standard parallel-dialogue corpus
used in MT/LM research; it contains aligned movie/TV subtitle pairs.

License: CC-BY 2.0. Original subtitles from www.opensubtitles.org.

Mirror: https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/moses/en-es.txt.zip
        ~1.8 GB compressed, ~5-7 GB uncompressed, ~30M en-es pairs.

Output (data/opensubtitles.txt):
  - Sampled subset (default 500 MB) of `<English> :: <Spanish>\n` pairs
  - Shuffled with deterministic seed
  - Filtered: drops near-empty lines, deduplicates exact pairs

To use the FULL corpus instead of a sample, pass --full.
At ~5 GB the byte tensor will fit in M4 Pro RAM but training one
epoch through it on MPS is ~14 days; sample first, scale up later.

Run:
    python make_opensubtitles_corpus.py                  # 500 MB sample
    python make_opensubtitles_corpus.py --target-mb 200  # 200 MB sample
    python make_opensubtitles_corpus.py --full           # full corpus
    python make_opensubtitles_corpus.py --refresh        # re-download

The download is cached in data/_opensubtitles_cache/ and survives
re-runs. Cache + output corpus are gitignored — only this script and
data/README.md are committed.
"""
from __future__ import annotations
import argparse
import os
import random
import time
import urllib.request
import zipfile
from pathlib import Path

DATA_DIR = Path("data")
OUT = DATA_DIR / "opensubtitles.txt"
CACHE_DIR = DATA_DIR / "_opensubtitles_cache"
ZIP_URL = "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/moses/en-es.txt.zip"
ZIP_PATH = CACHE_DIR / "en-es.zip"
EN_PATH  = CACHE_DIR / "OpenSubtitles.en-es.en"
ES_PATH  = CACHE_DIR / "OpenSubtitles.en-es.es"
SEED = 42

# Filtering — drop pairs that are likely junk
MIN_LEN = 4         # min chars per side after strip
MAX_LEN = 200       # max chars per side; subtitles are short, kill paragraphs
UNARY_FRACTION = 0.05
UNARY_MAX_N = 30


def _progress_hook(count: int, block_size: int, total_size: int) -> None:
    if total_size <= 0:
        return
    bytes_done = count * block_size
    pct = min(100.0, 100.0 * bytes_done / total_size)
    if count % 200 == 0 or bytes_done >= total_size:
        print(f"  ... {bytes_done/2**20:.1f}/{total_size/2**20:.1f} MB ({pct:.1f}%)", flush=True)


def download(refresh: bool = False) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if EN_PATH.exists() and ES_PATH.exists() and not refresh:
        en_size = EN_PATH.stat().st_size
        es_size = ES_PATH.stat().st_size
        print(f"using cached files in {CACHE_DIR}/  "
              f"(en={en_size/2**20:.0f} MB, es={es_size/2**20:.0f} MB)")
        return
    print(f"downloading {ZIP_URL}")
    print(f"  ~1.8 GB compressed, may take 5-30 min depending on bandwidth")
    urllib.request.urlretrieve(ZIP_URL, ZIP_PATH, reporthook=_progress_hook)
    print(f"  got {ZIP_PATH} ({ZIP_PATH.stat().st_size:,} bytes)")
    print("extracting parallel files (this is the slow part)...")
    with zipfile.ZipFile(ZIP_PATH) as zf:
        for name in zf.namelist():
            if name.startswith("OpenSubtitles.en-es") or name in ("README", "LICENSE"):
                zf.extract(name, CACHE_DIR)
                print(f"  extracted {name} ({(CACHE_DIR / name).stat().st_size/2**20:.0f} MB)")


def build_corpus(target_mb: float | None, full: bool) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    # Stream both files line-by-line — 5+ GB of text would be slow to
    # readlines() in one go. Use TextIO with iteration.
    print(f"streaming pairs from {EN_PATH.name} + {ES_PATH.name}")
    n_total = n_kept = n_unary = 0
    seen_pairs: set[tuple[str, str]] = set()  # exact-dup filter (cheap, scoped to kept)
    bytes_target = None if full else int(target_mb * 2**20)

    with open(EN_PATH, "r", encoding="utf-8", errors="replace") as fen, \
         open(ES_PATH, "r", encoding="utf-8", errors="replace") as fes, \
         open(OUT, "w", encoding="utf-8") as fout:
        bytes_written = 0
        for en, es in zip(fen, fes):
            n_total += 1
            en = en.strip()
            es = es.strip()
            if not (MIN_LEN <= len(en) <= MAX_LEN and MIN_LEN <= len(es) <= MAX_LEN):
                continue
            key = (en, es)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)

            line = f"{en} :: {es}\n"
            fout.write(line)
            bytes_written += len(line.encode("utf-8"))
            n_kept += 1

            # ~5% unary cortex mixin
            if rng.random() < UNARY_FRACTION:
                n = rng.randint(1, UNARY_MAX_N)
                u = "*" * n + ":" + "a" * n + "\n"
                fout.write(u)
                bytes_written += len(u.encode("utf-8"))
                n_unary += 1

            if bytes_target is not None and bytes_written >= bytes_target:
                break

            if n_total % 500_000 == 0:
                print(f"  scanned {n_total:,} pairs, kept {n_kept:,}, "
                      f"wrote {bytes_written/2**20:.1f} MB", flush=True)

    print(f"\nwrote {OUT}: {bytes_written/2**20:.1f} MB, {n_kept:,} kept pairs"
          f" + {n_unary:,} unary mixin lines (scanned {n_total:,} pairs total)")
    # Sample preview
    head = OUT.read_text(encoding="utf-8", errors="replace")[:1200]
    print("---first 1200 bytes---")
    print(head)

    # Archive the generated corpus to the HF bucket (no-op without HF_TOKEN).
    # `_*` excludes catch the multi-GB _opensubtitles_cache/.
    try:
        from cloud_archive import CloudArchive
        a = CloudArchive(
            experiment_kind="corpus",
            run_name=f"opensubtitles-{time.strftime('%Y-%m-%d')}",
            local_dir=str(DATA_DIR),
        )
        a.complete()
    except ImportError:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true",
                    help="force re-download of OpenSubtitles archive")
    ap.add_argument("--target-mb", type=float, default=500.0,
                    help="target output corpus size in MB (default 500)")
    ap.add_argument("--full", action="store_true",
                    help="write the FULL corpus instead of a sampled subset")
    args = ap.parse_args()
    download(refresh=args.refresh)
    build_corpus(target_mb=args.target_mb, full=args.full)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
