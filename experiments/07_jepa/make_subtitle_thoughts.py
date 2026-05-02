"""Generate (prompt, response, teacher_response_hidden) records for conv-JEPA.

Adjacent OpenSubtitles lines are scene-coherent — line N often answers,
reacts to, or continues line N+1's situation. We use them as cheap
(prompt, response) pairs for conversational distillation: the trainer
reads `<line_n>\\n<line_n+1>` as a single byte stream where the model
has to predict the response-side teacher hidden state from its own
end-of-prompt residual.

Lighter than make_teacher_thoughts.py:
  - No generation. The text already exists.
  - One teacher thought per record: the last-layer hidden at the BPE
    token whose decoded prefix ends at len(response) bytes — the
    *response-end* hidden. That's the target conv_jepa_loss matches.
  - Reuses the JEPT on-disk format with n_thoughts=1 so the existing
    TeacherThoughtsDataset / TeacherIterator load it for free; the
    trainer just dispatches by file path.

Run on the box:

    python make_subtitle_thoughts.py \\
        --model Qwen/Qwen2.5-1.5B-Instruct \\
        --pairs 200000 \\
        --out data/subtitle_thoughts
"""
from __future__ import annotations
import argparse
import os
import struct
import time
from pathlib import Path

import numpy as np
import torch

from make_teacher_thoughts import MAGIC, RECORD_HEADER_FMT, RECORD_HEADER_SIZE, write_record


def iter_subtitle_pairs(path: Path, min_len: int, max_len: int):
    """Yield (line_n, line_n+1) byte pairs from opensubtitles.txt.

    Filters: drop the unary `***:aaa` lines that some loaders mixed in,
    drop empty / pathological lines, drop pairs where either side falls
    outside [min_len, max_len] bytes. Pairs are non-overlapping (we
    consume two lines at a time) so each line appears at most once."""
    prev = None
    with open(path, "rb") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                prev = None
                continue
            if line.startswith(b"*") and b":" in line[:32]:
                # the unary `*****:aaa` synthetic mixed in earlier
                prev = None
                continue
            if not (min_len <= len(line) <= max_len):
                prev = None
                continue
            if prev is not None:
                yield prev, line
                prev = None
            else:
                prev = line


def find_token_at_byte(tokenizer, input_ids: list[int],
                        target_bytes: int) -> int:
    """Return token index whose decoded prefix first reaches `target_bytes`.

    Two-pointer walk: decode incremental prefixes until the byte length of
    the decoded text covers the target. Used for both end-of-prompt and
    end-of-response marks."""
    for i in range(len(input_ids)):
        prefix = tokenizer.decode(input_ids[: i + 1], skip_special_tokens=True)
        if len(prefix.encode("utf-8")) >= target_bytes:
            return i
    return len(input_ids) - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--pairs", type=int, default=200_000,
                    help="Stop after this many records written")
    ap.add_argument("--target-mb", type=float, default=400.0,
                    help="Or stop once .bin reaches this size in MB")
    ap.add_argument("--corpus", default="data/opensubtitles.txt")
    ap.add_argument("--out", default="data/subtitle_thoughts")
    ap.add_argument("--target", default="response_end",
                    choices=["response_end", "prompt_end"],
                    help="Which teacher hidden to capture per record. "
                         "response_end (round 5/6 default) was refuted; "
                         "prompt_end is plain hidden-state distillation, "
                         "the round-7 lever.")
    ap.add_argument("--min-line", type=int, default=8)
    ap.add_argument("--max-line", type=int, default=200)
    ap.add_argument("--max-input-tokens", type=int, default=128,
                    help="Skip pairs whose tokenization exceeds this; the "
                         "long-tail saves us from a few outliers and keeps "
                         "the forward pass fixed-cost.")
    ap.add_argument("--batch", type=int, default=16,
                    help="Pairs per teacher forward. Padded to longest in batch.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    print(f"loading teacher: {args.model}", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map=args.device,
    ).eval()
    d_teacher = model.config.hidden_size
    print(f"  loaded. hidden_size={d_teacher}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bin_path = out_path.with_suffix(".bin")
    idx_path = out_path.with_suffix(".idx")

    offsets: list[int] = [0]
    if args.resume and bin_path.exists() and idx_path.exists():
        with open(idx_path, "rb") as f:
            n_existing = int(np.frombuffer(f.read(8), dtype=np.int64)[0])
            offsets = list(np.frombuffer(f.read(), dtype=np.int64))
            assert len(offsets) == n_existing + 1
        bin_mode = "ab"
        print(f"  resuming from {n_existing} records, {offsets[-1]} bytes",
              flush=True)
    else:
        bin_mode = "wb"

    target_bytes = int(args.target_mb * 2**20)
    n_records = len(offsets) - 1
    n_attempts = 0
    bin_f = open(bin_path, bin_mode)
    t_start = time.time()

    pairs_iter = iter_subtitle_pairs(Path(args.corpus), args.min_line, args.max_line)

    # Batch up pairs and run the teacher in groups for throughput.
    pending: list[tuple[bytes, bytes]] = []
    pending_inputs: list[list[int]] = []
    pending_target_tok: list[int] = []

    def flush_batch():
        nonlocal n_records
        if not pending:
            return
        max_len = max(len(t) for t in pending_inputs)
        ids = np.full((len(pending), max_len), tokenizer.pad_token_id,
                      dtype=np.int64)
        attn = np.zeros((len(pending), max_len), dtype=np.int64)
        for i, t in enumerate(pending_inputs):
            ids[i, :len(t)] = t
            attn[i, :len(t)] = 1
        ids_t = torch.from_numpy(ids).to(args.device)
        attn_t = torch.from_numpy(attn).to(args.device)
        with torch.no_grad():
            out = model(ids_t, attention_mask=attn_t,
                        output_hidden_states=True, use_cache=False)
        last = out.hidden_states[-1]                         # (B, T, D)
        for i, ((q, r), tok_idx) in enumerate(zip(pending, pending_target_tok)):
            h = last[i, tok_idx].to(torch.float16).cpu().numpy()  # (D,)
            thoughts = h.reshape(1, -1)                      # K=1
            # byte_pos is unused by conv_jepa_loss but written for
            # collate_records compatibility. 0 keeps it inside any window.
            byte_pos = np.array([0], dtype=np.int32)
            written = write_record(bin_f, q.decode("utf-8"),
                                   r.decode("utf-8"), thoughts, byte_pos)
            offsets.append(offsets[-1] + written)
            n_records += 1
        pending.clear()
        pending_inputs.clear()
        pending_target_tok.clear()

    try:
        for q, r in pairs_iter:
            if n_records >= args.pairs or offsets[-1] >= target_bytes:
                break
            n_attempts += 1
            joined = (q + b"\n" + r).decode("utf-8", errors="replace")
            prompt_text = (q + b"\n").decode("utf-8", errors="replace")
            ids = tokenizer(joined, add_special_tokens=False)["input_ids"]
            if len(ids) < 2 or len(ids) > args.max_input_tokens:
                continue
            if args.target == "prompt_end":
                target_bytes_pos = len(prompt_text.encode("utf-8"))
            else:
                target_bytes_pos = len(joined.encode("utf-8"))
            tok_idx = find_token_at_byte(tokenizer, ids, target_bytes_pos)
            pending.append((q, r))
            pending_inputs.append(ids)
            pending_target_tok.append(tok_idx)
            if len(pending) >= args.batch:
                flush_batch()
                if n_attempts % (args.batch * 10) == 0:
                    bin_f.flush()
                    idx_tmp = idx_path.with_suffix(".idx.tmp")
                    with open(idx_tmp, "wb") as f:
                        f.write(np.array([n_records], dtype=np.int64).tobytes())
                        f.write(np.array(offsets, dtype=np.int64).tobytes())
                    os.replace(idx_tmp, idx_path)
                    rate = n_records / max(1.0, time.time() - t_start)
                    pct = 100 * offsets[-1] / target_bytes
                    print(f"  records={n_records:7d} attempts={n_attempts:7d} "
                          f"size={offsets[-1]/2**20:6.2f}/{args.target_mb} MB "
                          f"({pct:5.1f}%) rate={rate:.1f} pairs/s", flush=True)
        flush_batch()
        # Final idx write.
        bin_f.flush()
        with open(idx_path, "wb") as f:
            f.write(np.array([n_records], dtype=np.int64).tobytes())
            f.write(np.array(offsets, dtype=np.int64).tobytes())
    finally:
        bin_f.close()

    print(f"\ndone: {bin_path} ({offsets[-1]/2**20:.1f} MB, "
          f"{n_records} records, {n_attempts} attempts)", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
