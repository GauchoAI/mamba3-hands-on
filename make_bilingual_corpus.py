"""Build data/bilingual.txt from real Tatoeba English-Spanish parallel data.

Source: OPUS-Tatoeba v2023-04-12 (https://opus.nlpl.eu/Tatoeba.php).
Format: Creative Commons BY 2.0 — the upstream file is CC-BY licensed.
Mirror: https://object.pouta.csc.fi/OPUS-Tatoeba/v2023-04-12/moses/en-es.txt.zip

Output (data/bilingual.txt):
  - 222k parallel sentences as `<English> :: <Spanish>\n`
  - Shuffled with deterministic seed
  - ~5% unary counting mixin (`*N:aN`) so the cortex counter primitive
    is in-distribution if it gets attached to a language-trained LM later.

Reproducible: deterministic seed, idempotent download (cached at
data/_tatoeba_cache/). To rebuild from scratch, delete the cache.

Run:
    python make_bilingual_corpus.py            # default: uses cached download
    python make_bilingual_corpus.py --refresh  # re-download Tatoeba

The previous version of this script generated a primitive synthetic corpus
from sentence templates. That code is kept as `_synthetic_corpus_DEPRECATED.py`
in case the Tatoeba mirror is unreachable.
"""
from __future__ import annotations
import argparse
import os
import random
import urllib.request
import zipfile
from pathlib import Path

DATA_DIR = Path("data")
OUT = DATA_DIR / "bilingual.txt"
CACHE_DIR = DATA_DIR / "_tatoeba_cache"
ZIP_URL = "https://object.pouta.csc.fi/OPUS-Tatoeba/v2023-04-12/moses/en-es.txt.zip"
ZIP_PATH = CACHE_DIR / "en-es.zip"
EN_PATH  = CACHE_DIR / "Tatoeba.en-es.en"
ES_PATH  = CACHE_DIR / "Tatoeba.en-es.es"
SEED = 42
UNARY_FRACTION = 0.05
UNARY_MAX_N = 30   # keep the cortex training distribution


def download_tatoeba(refresh: bool = False) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if EN_PATH.exists() and ES_PATH.exists() and not refresh:
        print(f"using cached Tatoeba files in {CACHE_DIR}/")
        return
    print(f"downloading {ZIP_URL} ...")
    with urllib.request.urlopen(ZIP_URL) as r:
        ZIP_PATH.write_bytes(r.read())
    print(f"  got {ZIP_PATH} ({ZIP_PATH.stat().st_size:,} bytes)")
    with zipfile.ZipFile(ZIP_PATH) as zf:
        for name in ["Tatoeba.en-es.en", "Tatoeba.en-es.es", "README", "LICENSE"]:
            zf.extract(name, CACHE_DIR)
    print(f"  extracted to {CACHE_DIR}/")


def build_corpus() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    en_lines = EN_PATH.read_text(encoding="utf-8").splitlines()
    es_lines = ES_PATH.read_text(encoding="utf-8").splitlines()
    if len(en_lines) != len(es_lines):
        raise ValueError(f"line count mismatch: en={len(en_lines)} es={len(es_lines)}")

    pairs = list(zip(en_lines, es_lines))
    rng.shuffle(pairs)

    out_buf: list[str] = []
    for en, es in pairs:
        en = en.strip()
        es = es.strip()
        if not en or not es:
            continue
        out_buf.append(f"{en} :: {es}\n")
        # Sprinkle unary cortex examples ~UNARY_FRACTION of lines.
        if rng.random() < UNARY_FRACTION:
            n = rng.randint(1, UNARY_MAX_N)
            out_buf.append("*" * n + ":" + "a" * n + "\n")

    text = "".join(out_buf)
    OUT.write_bytes(text.encode("utf-8"))
    n_bytes = len(text.encode("utf-8"))
    print(f"wrote {OUT}: {n_bytes:,} bytes, {text.count(chr(10)):,} lines")
    print("---first 800 bytes---")
    print(text[:800])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true",
                    help="force re-download of Tatoeba archive")
    args = ap.parse_args()
    download_tatoeba(refresh=args.refresh)
    build_corpus()


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
