"""Extract within-movie dialogue pairs from OpenSubtitles raw cache.

Reads the .en/.es/.ids triple in lockstep, emits `<en> :: <es>\n` lines, with
a blank line whenever the movie ID changes. Existing `iter_subtitle_pairs` in
make_subtitle_thoughts.py treats blank lines as boundaries (resets `prev`), so
pairs never cross movie boundaries when reading this file.

Why: the existing data/opensubtitles.txt was built by globbing and concating —
half the consecutive-line "dialogue pairs" we'd been training on were
cross-movie noise. Cleaner corpus = more reliable Q→A latent structure.
"""
from __future__ import annotations
import sys
from pathlib import Path

CACHE = Path("data/_opensubtitles_cache")
EN = CACHE / "OpenSubtitles.en-es.en"
ES = CACHE / "OpenSubtitles.en-es.es"
IDS = CACHE / "OpenSubtitles.en-es.ids"
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "data/movie_pairs_clean.txt")
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 0  # 0 = no limit


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n_lines, n_movies, n_emitted = 0, 0, 0
    prev_movie = None
    with EN.open("r", encoding="utf-8", errors="replace") as fen, \
         ES.open("r", encoding="utf-8", errors="replace") as fes, \
         IDS.open("r", encoding="utf-8", errors="replace") as fids, \
         OUT.open("w", encoding="utf-8") as fout:
        for en_line, es_line, ids_line in zip(fen, fes, fids):
            n_lines += 1
            parts = ids_line.split("\t")
            if not parts:
                continue
            movie = parts[0]
            if movie != prev_movie:
                if prev_movie is not None:
                    fout.write("\n")
                prev_movie = movie
                n_movies += 1
            en_clean = en_line.strip()
            es_clean = es_line.strip()
            if en_clean and es_clean and " :: " not in en_clean:
                fout.write(f"{en_clean} :: {es_clean}\n")
                n_emitted += 1
            if LIMIT and n_emitted >= LIMIT:
                break
            if n_lines % 1_000_000 == 0:
                print(f"  read {n_lines/1e6:.1f}M lines, "
                      f"{n_movies} movies, {n_emitted} emitted",
                      flush=True)
    print(f"\ndone: {OUT} — {n_emitted} pairs across {n_movies} movies "
          f"({OUT.stat().st_size/2**20:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
