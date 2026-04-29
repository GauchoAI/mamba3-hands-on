"""Generate paired (text, thought_trajectory) records for JEPA-Cortex.

W1.1 of the JEPA-Cortex plan. Differs from make_teacher_corpus.py in two
ways: (1) CUDA + transformers instead of MLX (we deploy on rented GPU),
(2) writes a binary record stream that includes the teacher's last-layer
hidden state, snapshotted at byte-aligned positions in the generated
response. That trajectory is what the student's thought_head will learn
to predict.

On-disk format (one record per generation, sequentially appended):

    uint32  magic = 0x4A455054                # "JEPT"
    uint32  question_len_bytes
    uint32  response_len_bytes
    uint32  n_thoughts
    uint32  d_teacher
    bytes   question[question_len_bytes]      # natural-language user prompt
    bytes   response[response_len_bytes]      # what the teacher generated
    int32   thought_byte_pos[n_thoughts]      # offset into `response`
    float16 thoughts[n_thoughts, d_teacher]   # teacher hidden at each pos

Sidecar index (`*.idx`):

    int64   n_records
    int64   offsets[n_records + 1]            # byte offsets into .bin

Run on the rented box:

    python make_teacher_thoughts.py \\
        --model Qwen/Qwen2.5-1.5B-Instruct \\
        --target-mb 80 \\
        --out data/teacher_thoughts

This is a one-shot job. Resume is handled by the `--resume` flag, which
seeks to end-of-file and continues appending.
"""
from __future__ import annotations
import argparse
import os
import random
import re
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Prompt templates — same diversity as make_teacher_corpus.py so the JEPA
# corpus distribution matches the byte-only Path A baseline. SIGreg needs a
# wide topic distribution to enforce intent diversity; if every prompt is
# the same template, the latent collapses regardless of what we regularize.
# ---------------------------------------------------------------------------
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

# Counting prompts (W4.1): natural-language requests that exercise the
# CounterPrimitive's gates against real text, not synthetic byte literals.
COUNTING_TEMPLATES_EN = [
    "Count from 1 to {n}, one number per line, then stop.",
    "List the integers 1 through {n}, one per line.",
    "Number from 1 up to {n}, line by line, no extras.",
]
COUNTING_TEMPLATES_ES = [
    "Cuenta de 1 a {n}, un número por línea, luego detente.",
    "Enumera los enteros del 1 al {n}, uno por línea.",
    "Numera del 1 hasta {n}, línea por línea, sin extras.",
]

MAGIC = 0x4A455054  # "JEPT"
RECORD_HEADER_FMT = "<IIIII"  # magic, q_len, r_len, n_thoughts, d_teacher
RECORD_HEADER_SIZE = struct.calcsize(RECORD_HEADER_FMT)


# ---------------------------------------------------------------------------
# Hidden-state capture
# ---------------------------------------------------------------------------
def thought_positions(response_token_ids: list[int],
                      tokenizer,
                      stride_bytes: int) -> tuple[list[int], list[int]]:
    """Map response tokens to byte offsets, then pick every stride_bytes mark.

    Returns:
        token_indices: list[int] — for each stride mark, the index of the
            FIRST token whose decoded prefix covers that mark
        byte_positions: list[int] — the stride mark itself (k * stride_bytes)
            for each captured thought, NOT the token-end byte. Aligning on
            the mark keeps the trainer's predict-the-next-thought shift
            (`src = byte_pos - stride`) well-defined and on a regular grid.

    Why incremental decode: BPE tokens may merge subwords / multi-byte UTF-8
    characters, so token-i's byte length is only well-defined as
    `len(decode(ids[:i+1])) - len(decode(ids[:i]))`. Accents, em-dashes, and
    em-spaces in Spanish output are the most common offenders.

    A single long token can cross multiple stride marks; we emit one capture
    per mark, all pointing at the same teacher token, so the student is
    trained to predict that single hidden from multiple consecutive byte
    positions in its own residual stream.
    """
    token_indices, byte_positions = [], []
    prev_bytes = 0
    next_mark = stride_bytes
    for i in range(len(response_token_ids)):
        prefix = tokenizer.decode(response_token_ids[: i + 1],
                                  skip_special_tokens=True)
        cur_bytes = len(prefix.encode("utf-8"))
        while cur_bytes >= next_mark and prev_bytes < next_mark:
            token_indices.append(i)
            byte_positions.append(next_mark)
            next_mark += stride_bytes
        prev_bytes = cur_bytes
    return token_indices, byte_positions


def capture_thoughts(model,
                     tokenizer,
                     full_ids: torch.Tensor,
                     prompt_token_len: int,
                     stride_bytes: int) -> tuple[np.ndarray, np.ndarray, str]:
    """Run a forward on (prompt + response) with output_hidden_states and
    return (thoughts[K, D], byte_positions[K], decoded_response).

    Two-pass design (generate, then re-forward): simpler and more reliable
    than hooking incremental decode. Costs ~1.5× compute but gives us
    cleanly aligned per-token hidden states in one shot.
    """
    with torch.no_grad():
        out = model(full_ids, output_hidden_states=True, use_cache=False)
    # Last layer's hidden state. Shape: (1, T, D).
    last = out.hidden_states[-1][0]  # (T, D)

    response_ids = full_ids[0, prompt_token_len:].tolist()
    decoded = tokenizer.decode(response_ids, skip_special_tokens=True)

    tok_idx, byte_pos = thought_positions(response_ids, tokenizer, stride_bytes)
    if not tok_idx:
        return None, None, decoded

    # Convert response-local token indices to absolute positions in `last`
    abs_idx = [prompt_token_len + j for j in tok_idx]
    thoughts = last[abs_idx].to(torch.float16).cpu().numpy()  # (K, D)
    byte_pos_arr = np.array(byte_pos, dtype=np.int32)
    return thoughts, byte_pos_arr, decoded


# ---------------------------------------------------------------------------
# Record I/O
# ---------------------------------------------------------------------------
def write_record(fout, question: str, response: str,
                 thoughts: np.ndarray, byte_positions: np.ndarray) -> int:
    """Append one record. Returns bytes written (caller updates idx)."""
    q_bytes = question.encode("utf-8")
    r_bytes = response.encode("utf-8")
    K, D = thoughts.shape
    header = struct.pack(RECORD_HEADER_FMT, MAGIC, len(q_bytes), len(r_bytes), K, D)
    fout.write(header)
    fout.write(q_bytes)
    fout.write(r_bytes)
    fout.write(byte_positions.astype(np.int32, copy=False).tobytes())
    fout.write(thoughts.astype(np.float16, copy=False).tobytes())
    return RECORD_HEADER_SIZE + len(q_bytes) + len(r_bytes) + 4 * K + 2 * K * D


# ---------------------------------------------------------------------------
# Generation loop
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--target-mb", type=float, default=80.0,
                    help="Stop once .bin reaches this size in MB")
    ap.add_argument("--out", default="data/teacher_thoughts",
                    help="Output basename; writes .bin and .idx")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stride-bytes", type=int, default=16)
    ap.add_argument("--counting-fraction", type=float, default=0.05,
                    help="Fraction of prompts that are natural-language counting")
    ap.add_argument("--pairs-per-prompt", type=int, default=12)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--resume", action="store_true",
                    help="Append to existing files instead of truncating")
    args = ap.parse_args()

    print(f"loading teacher: {args.model}", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map=args.device,
    ).eval()
    d_teacher = model.config.hidden_size
    print(f"  loaded. hidden_size={d_teacher}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bin_path = out_path.with_suffix(".bin")
    idx_path = out_path.with_suffix(".idx")

    # idx file holds a header (n_records as int64) followed by offsets.
    # On resume we read existing offsets and seek both files to end.
    offsets: list[int] = [0]
    if args.resume and bin_path.exists() and idx_path.exists():
        with open(idx_path, "rb") as f:
            n_existing = int(np.frombuffer(f.read(8), dtype=np.int64)[0])
            offsets = list(np.frombuffer(f.read(), dtype=np.int64))
            assert len(offsets) == n_existing + 1
        bin_mode = "ab"
        print(f"  resuming from {n_existing} records, {offsets[-1]} bytes", flush=True)
    else:
        bin_mode = "wb"

    rng = random.Random(args.seed)
    target_bytes = int(args.target_mb * 2**20)
    n_attempts = 0
    n_records = len(offsets) - 1

    bin_f = open(bin_path, bin_mode)
    t_start = time.time()

    try:
        while offsets[-1] < target_bytes:
            n_attempts += 1
            # Mix counting prompts into the standard topic prompts.
            if rng.random() < args.counting_fraction:
                n = rng.randint(3, 30)
                if rng.random() < 0.5:
                    user = rng.choice(COUNTING_TEMPLATES_EN).format(n=n)
                else:
                    user = rng.choice(COUNTING_TEMPLATES_ES).format(n=n)
            else:
                user = rng.choice(TOPIC_TEMPLATES).format(
                    n=args.pairs_per_prompt, topic=rng.choice(TOPICS),
                )

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ]
            prompt_text = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False,
            )
            prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(args.device)
            prompt_token_len = prompt_ids.shape[1]

            gen = model.generate(
                prompt_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tokenizer.eos_token_id,
            )

            # If generation degenerates to immediate EOS, skip cheaply.
            if gen.shape[1] - prompt_token_len < 4:
                continue

            thoughts, byte_pos, response = capture_thoughts(
                model, tokenizer, gen, prompt_token_len, args.stride_bytes,
            )
            if thoughts is None or len(response.encode("utf-8")) < args.stride_bytes:
                # Too short to give even one stride mark — drop.
                continue

            written = write_record(bin_f, user, response, thoughts, byte_pos)
            offsets.append(offsets[-1] + written)
            n_records += 1

            if n_attempts % 5 == 0 or offsets[-1] >= target_bytes:
                bin_f.flush()
                # Rewrite idx atomically; cheap (a few KB).
                idx_tmp = idx_path.with_suffix(".idx.tmp")
                with open(idx_tmp, "wb") as f:
                    f.write(np.array([n_records], dtype=np.int64).tobytes())
                    f.write(np.array(offsets, dtype=np.int64).tobytes())
                os.replace(idx_tmp, idx_path)
                pct = 100 * offsets[-1] / target_bytes
                rate = (offsets[-1] - offsets[0]) / max(1.0, time.time() - t_start) / 1024
                print(f"  records={n_records:5d}  attempts={n_attempts:5d}  "
                      f"size={offsets[-1]/2**20:6.2f}/{args.target_mb} MB  "
                      f"({pct:5.1f}%)  rate={rate:.1f} KB/s",
                      flush=True)
    finally:
        bin_f.close()

    print(f"\ndone: {bin_path} ({offsets[-1]/2**20:.1f} MB, {n_records} records, "
          f"{n_attempts} attempts)", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
