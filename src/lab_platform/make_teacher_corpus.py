"""Generate a bilingual teacher corpus by running an open multilingual LM.

Path A from the distillation plan: pseudo-label distillation. We don't
KL-match teacher logits (incompatible vocabularies — student is
byte-level, teacher is BPE/SentencePiece). Instead we use the teacher
to produce high-quality bilingual text and train the student on that
text as a pure corpus.

Teacher: Qwen-2.5-1.5B-Instruct via mlx-lm (Apple Metal). Multilingual,
strong at en/es, ~3 GB. Runs comfortably on M4 with 16 GB+ RAM.

Format mirrors data/bilingual.txt: `<English> :: <Spanish>\n` lines
plus the cortex unary mixin (~5%) so the counter primitive stays in
distribution if attached later.

Strategy:
1. Seed the teacher with diverse bilingual prompts (a few hundred
   dialogue / topic / scene templates).
2. For each seed, ask the teacher to produce N parallel sentence
   pairs (en :: es).
3. Light filtering: drop lines where format isn't satisfied.
4. Append cortex unary lines.

Run on m4-mini overnight (4+ TB disk, no GPU contention with the
M4 Pro's training):

    ssh m4-mini 'cd ~/mamba3-hands-on && \
        nohup .venv/bin/python make_teacher_corpus.py \
            --target-mb 200 --out data/teacher_corpus.txt > teacher.log 2>&1 &'

Then sync back:
    rsync -av m4-mini:~/mamba3-hands-on/data/teacher_corpus.txt data/

Args:
    --model           HF model id (default: Qwen-2.5-1.5B-Instruct)
    --target-mb       output corpus size in MB (default 200)
    --out             output path (default data/teacher_corpus.txt)
    --batches-per-seed how many generations per seed prompt (default 3)
    --seed            RNG seed for prompt selection
"""
from __future__ import annotations
import argparse
import os
import random
import re
import sys
from pathlib import Path

# --- Seed prompts ----------------------------------------------------
# Each prompt asks the teacher to produce N parallel English / Spanish
# pairs in our format. Variety: dialogue, narrative, instruction,
# domain (cooking, travel, science, kids, formal, slang).

SYSTEM_PROMPT = (
    "You write high-quality parallel English-Spanish text. "
    "Output only lines of the format: <English sentence> :: <Spanish sentence>\\n. "
    "No headers, no numbering, no explanations. "
    "Use natural register that matches the topic — colloquial when colloquial, "
    "formal when formal. Cover a wide variety of vocabulary."
)

TOPIC_TEMPLATES = [
    "Write {n} parallel English-Spanish dialogue lines about {topic}.",
    "Write {n} parallel English-Spanish narrative sentences set in {topic}.",
    "Write {n} parallel English-Spanish question-and-answer pairs about {topic}.",
    "Write {n} parallel English-Spanish casual conversation lines about {topic}.",
    "Write {n} parallel English-Spanish instructional sentences about {topic}.",
]

TOPICS = [
    "cooking and recipes", "travel and directions",
    "science and discovery", "everyday family life",
    "school and learning", "work and the office",
    "weather and seasons", "music and art",
    "children's stories", "history and old times",
    "computers and technology", "sports and games",
    "health and feelings", "food at a restaurant",
    "shopping and money", "books and reading",
    "animals and nature", "trains and cars",
    "love and friendship", "fear and bravery",
    "morning routines", "rainy days",
    "moving to a new city", "an old friend visiting",
    "learning a new skill", "fixing something broken",
    "a long phone call", "an unexpected guest",
    "working from home", "a small misunderstanding",
    "celebrating a birthday", "missing a flight",
    "a quiet evening", "a noisy market",
    "dreaming and remembering", "asking for help",
    "asking for directions", "reading the newspaper",
    "watching the stars", "the first day of school",
]


def build_prompts(rng: random.Random, n_prompts: int,
                  pairs_per_prompt: int) -> list[str]:
    prompts = []
    for _ in range(n_prompts):
        topic = rng.choice(TOPICS)
        template = rng.choice(TOPIC_TEMPLATES)
        prompts.append(template.format(n=pairs_per_prompt, topic=topic))
    return prompts


# --- Output filter ---------------------------------------------------
PAIR_RE = re.compile(r"^(.+?)\s+::\s+(.+?)\s*$")
MIN_LEN, MAX_LEN = 5, 200


def parse_pairs(text: str) -> list[str]:
    """Extract well-formed `en :: es` lines from teacher output."""
    out = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line: continue
        # Strip leading numbering like "1. " or "- " or "* "
        line = re.sub(r"^\s*(?:\d+[\.\)]\s*|[-*•]\s*)", "", line)
        m = PAIR_RE.match(line)
        if not m: continue
        en, es = m.group(1).strip(), m.group(2).strip()
        if not (MIN_LEN <= len(en) <= MAX_LEN and MIN_LEN <= len(es) <= MAX_LEN):
            continue
        out.append(f"{en} :: {es}\n")
    return out


# --- Main ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen2.5-1.5B-Instruct-4bit",
                    help="HF model id (mlx-lm-compatible). Default is "
                    "the 4-bit MLX-quantized Qwen2.5-1.5B-Instruct (~1.0 GB).")
    ap.add_argument("--target-mb", type=float, default=200.0,
                    help="Stop generating once output reaches this size in MB")
    ap.add_argument("--out", default="data/teacher_corpus.txt")
    ap.add_argument("--pairs-per-prompt", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=2048,
                    help="Max tokens per generation (~20 pairs ≈ 1500-2000 tokens)")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--unary-fraction", type=float, default=0.05)
    args = ap.parse_args()

    print(f"loading teacher: {args.model}", flush=True)
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    model, tokenizer = load(args.model)
    print(f"  loaded.", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    bytes_target = int(args.target_mb * 2**20)
    bytes_written = 0
    n_pairs = 0
    n_unary = 0
    n_attempts = 0

    sampler = make_sampler(temp=args.temperature, top_p=0.95)

    with open(out_path, "w", encoding="utf-8") as fout:
        while bytes_written < bytes_target:
            n_attempts += 1
            topic = rng.choice(TOPICS)
            template = rng.choice(TOPIC_TEMPLATES)
            user = template.format(n=args.pairs_per_prompt, topic=topic)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user},
            ]
            prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            text = generate(
                model, tokenizer,
                prompt=prompt,
                max_tokens=args.max_tokens,
                sampler=sampler,
                verbose=False,
            )
            pairs = parse_pairs(text)

            for pair in pairs:
                fout.write(pair)
                bytes_written += len(pair.encode("utf-8"))
                n_pairs += 1
                if rng.random() < args.unary_fraction:
                    n = rng.randint(1, 30)
                    u = "*" * n + ":" + "a" * n + "\n"
                    fout.write(u)
                    bytes_written += len(u.encode("utf-8"))
                    n_unary += 1

            if n_attempts % 5 == 0 or bytes_written >= bytes_target:
                fout.flush()
                pct = 100 * bytes_written / bytes_target
                print(f"  attempts={n_attempts}  pairs={n_pairs}  unary={n_unary}  "
                      f"size={bytes_written/2**20:.1f}/{args.target_mb} MB  "
                      f"({pct:.1f}%)", flush=True)

    print(f"\ndone: {out_path} ({bytes_written/2**20:.1f} MB, "
          f"{n_pairs:,} pairs + {n_unary:,} unary across {n_attempts} prompts)")

    # Archive to the HF bucket (no-op without HF_TOKEN).
    try:
        import time as _time
        from .cloud_archive import CloudArchive
        a = CloudArchive(
            experiment_kind="corpus",
            run_name=f"teacher-local-qwen-{_time.strftime('%Y-%m-%d')}",
            local_dir=str(out_path.parent),
        )
        a.complete()
    except ImportError:
        pass


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
