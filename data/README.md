# `data/` — corpus files for LM training

`data/` is gitignored (`.gitignore` line `data/`). Files here must be
regenerable from a checked-in script, or downloaded from a documented
source. We do not commit corpora directly — even small ones — so the
build path stays reproducible.

## `bilingual.txt` — Tatoeba English+Spanish parallel corpus (~17.7 MB)

**What it is:** byte-level English+Spanish parallel sentences from
[Tatoeba](https://tatoeba.org/en/downloads) (222,073 pairs from the
OPUS-Tatoeba v2023-04-12 mirror), formatted as `<en> :: <es>\n` per
line, shuffled, with a ~5% mixin of unary cortex counting strings
(`*N:aN`) so the counter primitive is in-distribution if attached to
this language-trained LM later. Used to train
`train_bilingual_cortex_lm.py`.

**License:** Tatoeba sentences are CC-BY 2.0 (see
`data/_tatoeba_cache/LICENSE` after running the script). Attribution:
The Tatoeba Project — https://tatoeba.org. Mirror provided by OPUS.

**Status:** *not committed*. Regenerate with one command:

```bash
python make_bilingual_corpus.py            # uses cached download if present
python make_bilingual_corpus.py --refresh  # force re-download from OPUS
```

The script downloads
`https://object.pouta.csc.fi/OPUS-Tatoeba/v2023-04-12/moses/en-es.txt.zip`
(~7 MB) into `data/_tatoeba_cache/` (also gitignored), unzips, joins
the parallel `Tatoeba.en-es.en` and `Tatoeba.en-es.es` files, shuffles
deterministically (`seed=42`), interleaves the unary cortex mixin, and
writes `data/bilingual.txt`.

### History

The previous-session `bilingual.txt` (referenced in `CORTEX_HANDOFF.md`
and `ARCHITECTURE.md`) was Tatoeba-sourced too. That specific file
was lost (not present on m4-pro, not on m4-mini at 192.168.0.170,
likely vanished with the vast.ai pod-archive era). The current
script reproduces an equivalent corpus from the OPUS mirror and
should be considered the canonical regeneration path going forward.

A previous synthetic-template fallback that ran without internet is
in git history at the same path (rough vocab-substitution sentences,
~500 KB). The script was rewritten in favor of the real Tatoeba
download because the synthetic version's grammar didn't support
gender/number agreement and was too small for meaningful LM training.

## Why no git LFS

The corpus is a few hundred KB, regenerable from a checked-in
deterministic script, and likely to evolve as we extend templates or
swap in a real Tatoeba dump. Git LFS would add infrastructure overhead
without reducing reproducibility. If we ever need a real multi-MB
corpus that *can't* be regenerated, LFS is the right move; today it
isn't.
