"""Generate per-byte teacher distributions for logit-projection KD.

The long-postponed real distillation, queued since findings §5 → memory
entry feedback_logit_distillation_gap.md. The full sequence-level
pseudo-distillation × loss × target grid has been eliminated; this is
the only distillation flavor with byte-level signal density.

Design:
  Read consecutive within-movie pairs from data/movie_pairs_clean.txt.
  For each pair, build the joined byte stream "<line_n>\\n<line_n+1>".
  BPE-tokenize via Qwen-2.5-1.5B-Instruct, run a forward pass over the
  full sequence with output_hidden_states=False (we only need logits).

  At each BPE-token boundary:
    - Take softmax over vocab to get next-token distribution.
    - For top-K tokens (default K=256), decode each and read its FIRST
      byte. Marginalize: next_byte_dist[b] = sum_{tok: first_byte(tok)==b}
      P(tok). This is the "logit-projection to byte distribution"
      described in feedback_logit_distillation_gap.md.
    - Quantize to uint8 via simple linear scaling (probabilities ∈ [0,1]
      → ints ∈ [0, 255]). The trainer dequantizes for KL.

  At positions NOT at BPE boundaries: don't store anything — the next
  byte is determinate (just the next byte of the current BPE token), so
  no auxiliary signal is needed.

  Per-record storage: ~K_marks × 260 bytes ≈ 13 KB/record at typical
  pair lengths. 130k records → 1.7 GB. Same order as the existing
  subtitle_thoughts corpora.

On-disk format (binary, sequential, one record per pair):

    uint32  magic = 0x4B445450               # "KDTP" (KD-T-P, T=teacher P=projection)
    uint32  question_len_bytes
    uint32  response_len_bytes
    uint32  n_marks
    bytes   question[question_len_bytes]
    bytes   response[response_len_bytes]
    int32   mark_byte_pos[n_marks]           # student byte index of each mark
    uint8   mark_byte_dist[n_marks, 256]     # quantized next-byte distribution

Sidecar .idx: int64 n_records; int64 offsets[n_records + 1]

Run on the box (after corpus generation):

    cd /workspace/mamba3-hands-on
    source .venv/bin/activate
    CUDA_VISIBLE_DEVICES=N python jepa/make_logit_kd_corpus.py \\
        --pairs 100000 --batch 16 \\
        --corpus data/movie_pairs_clean.txt \\
        --out data/kd_logit_clean

This is the first generator that actually delivers byte-level
prompt-conditional gradient — the loss target depends on every byte of
the prompt because teacher logits change with each input byte. By the
input-shuffle test from feedback_loss_target_input_dependence.md, this
should pass: shuffled inputs change the teacher's logits, so the loss
can't be fit by a corpus-mean projector.
"""
from __future__ import annotations
import argparse
import os
import struct
import time
from pathlib import Path

import numpy as np
import torch

MAGIC = 0x4B445450  # "KDTP"
RECORD_HEADER_FMT = "<IIII"
RECORD_HEADER_SIZE = struct.calcsize(RECORD_HEADER_FMT)


def iter_within_movie_pairs(path: Path, min_len: int, max_len: int):
    """Yield (line_n, line_n+1) byte pairs, treating blank lines as movie
    boundaries (the format produced by extract_movie_pairs.py)."""
    prev = None
    with open(path, "rb") as f:
        for raw in f:
            line = raw.strip()
            if not line:
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


def find_token_byte_ends(tokenizer, input_ids: list[int]) -> list[int]:
    """For each input token i, return the byte index of its last byte
    in the full decoded string. Walks decoded prefixes; same logic as
    make_subtitle_thoughts.find_token_at_byte but inverted (we want
    every boundary, not a single target)."""
    ends = []
    prev_bytes = 0
    for i in range(len(input_ids)):
        prefix = tokenizer.decode(input_ids[: i + 1], skip_special_tokens=True)
        cur_bytes = len(prefix.encode("utf-8"))
        ends.append(cur_bytes - 1 if cur_bytes > 0 else 0)
        prev_bytes = cur_bytes
    return ends


def project_logits_to_bytes(logits: torch.Tensor, tokenizer,
                             top_k: int) -> np.ndarray:
    """Project a single position's vocab logits onto a 256-D next-byte
    distribution by marginalizing over BPE tokens that share a first
    byte. Returns uint8-quantized array of length 256.

    logits: (V,) fp32 — vocab-sized logits at one position.
    """
    probs = torch.softmax(logits, dim=-1)
    top_p, top_t = torch.topk(probs, k=min(top_k, probs.numel()))
    top_p = top_p.cpu().numpy()
    top_t = top_t.cpu().tolist()
    # Decode each candidate; tokenizer can give an empty string for
    # special/space tokens — skip those (they contribute no first byte).
    byte_dist = np.zeros(256, dtype=np.float32)
    for tok_id, p in zip(top_t, top_p):
        s = tokenizer.decode([tok_id], skip_special_tokens=True)
        b = s.encode("utf-8")
        if not b:
            continue
        byte_dist[b[0]] += float(p)
    s = byte_dist.sum()
    if s > 0:
        byte_dist = byte_dist / s
    # uint8 quantization: scale [0, 1] → [0, 255], rounded.
    return np.clip((byte_dist * 255.0).round(), 0, 255).astype(np.uint8)


def write_record(fout, q: bytes, r: bytes,
                  mark_byte_pos: np.ndarray,
                  mark_byte_dist: np.ndarray) -> int:
    K = mark_byte_pos.shape[0]
    header = struct.pack(RECORD_HEADER_FMT, MAGIC, len(q), len(r), K)
    fout.write(header)
    fout.write(q)
    fout.write(r)
    fout.write(mark_byte_pos.astype(np.int32, copy=False).tobytes())
    fout.write(mark_byte_dist.astype(np.uint8, copy=False).tobytes())
    return RECORD_HEADER_SIZE + len(q) + len(r) + 4 * K + 256 * K


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--corpus", default="data/movie_pairs_clean.txt")
    ap.add_argument("--out", default="data/kd_logit_clean")
    ap.add_argument("--pairs", type=int, default=100_000)
    ap.add_argument("--target-mb", type=float, default=2000.0)
    ap.add_argument("--top-k", type=int, default=256,
                    help="How many top tokens to marginalize per position. "
                         "256 covers ~99.9% of probability mass on Qwen "
                         "byte-level continuations and matches the byte "
                         "vocab size, so the projection is dense enough.")
    ap.add_argument("--max-input-tokens", type=int, default=128)
    ap.add_argument("--min-line", type=int, default=8)
    ap.add_argument("--max-line", type=int, default=200)
    ap.add_argument("--batch", type=int, default=16)
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
    print(f"  loaded. vocab={model.config.vocab_size}", flush=True)

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
    pairs_iter = iter_within_movie_pairs(Path(args.corpus),
                                          args.min_line, args.max_line)
    bin_f = open(bin_path, bin_mode)
    t_start = time.time()

    pending: list[tuple[bytes, bytes, list[int], int]] = []

    def flush_batch():
        nonlocal n_records
        if not pending:
            return
        max_len = max(len(t[2]) for t in pending)
        ids = np.full((len(pending), max_len), tokenizer.pad_token_id,
                      dtype=np.int64)
        attn = np.zeros((len(pending), max_len), dtype=np.int64)
        for i, t in enumerate(pending):
            ids[i, :len(t[2])] = t[2]
            attn[i, :len(t[2])] = 1
        ids_t = torch.from_numpy(ids).to(args.device)
        attn_t = torch.from_numpy(attn).to(args.device)
        with torch.no_grad():
            out = model(ids_t, attention_mask=attn_t, use_cache=False)
        # logits: (B, T, V). We want, for each token position t at the
        # end of a BPE boundary, the next-byte distribution = projection
        # of softmax(logits[t]) to bytes.
        for i, (q, r, t_ids, prompt_tok_len) in enumerate(pending):
            byte_ends = find_token_byte_ends(tokenizer, t_ids)
            joined_len = len(q) + 1 + len(r)
            marks_pos: list[int] = []
            marks_dist: list[np.ndarray] = []
            # Skip the prompt-half tokens — we only want supervision in
            # the response. Boundaries inside the prompt are seen by the
            # student but the teacher's loss there is just learning to
            # echo the prompt (we want next-byte conditional prediction
            # for the *response* given the prompt).
            for tk in range(prompt_tok_len, len(t_ids)):
                end_byte = byte_ends[tk]
                if end_byte >= joined_len - 1:
                    break  # past response end
                # At byte position end_byte, the teacher predicts the
                # next byte. logits at position tk give next-token
                # distribution.
                dist = project_logits_to_bytes(
                    out.logits[i, tk].float(), tokenizer, args.top_k,
                )
                marks_pos.append(end_byte)
                marks_dist.append(dist)
            if not marks_pos:
                continue
            mp = np.array(marks_pos, dtype=np.int32)
            md = np.stack(marks_dist, axis=0)
            written = write_record(bin_f, q, r, mp, md)
            offsets.append(offsets[-1] + written)
            n_records += 1
        pending.clear()

    try:
        for q, r in pairs_iter:
            if n_records >= args.pairs or offsets[-1] >= target_bytes:
                break
            n_attempts += 1
            joined = q + b"\n" + r
            joined_text = joined.decode("utf-8", errors="replace")
            prompt_text = (q + b"\n").decode("utf-8", errors="replace")
            ids = tokenizer(joined_text,
                             add_special_tokens=False)["input_ids"]
            if len(ids) < 4 or len(ids) > args.max_input_tokens:
                continue
            # Find which token covers the last byte of the prompt — we
            # only generate KD signal for response-side tokens.
            prompt_bytes_target = len(prompt_text.encode("utf-8"))
            prompt_tok_len = 0
            cur_bytes = 0
            for ti in range(len(ids)):
                cur_bytes = len(tokenizer.decode(
                    ids[: ti + 1], skip_special_tokens=True
                ).encode("utf-8"))
                prompt_tok_len = ti + 1
                if cur_bytes >= prompt_bytes_target:
                    break
            pending.append((q, r, ids, prompt_tok_len))
            if len(pending) >= args.batch:
                flush_batch()
                if n_attempts % (args.batch * 10) == 0:
                    bin_f.flush()
                    idx_tmp = idx_path.with_suffix(".idx.tmp")
                    with open(idx_tmp, "wb") as f:
                        f.write(np.array([n_records],
                                         dtype=np.int64).tobytes())
                        f.write(np.array(offsets,
                                         dtype=np.int64).tobytes())
                    os.replace(idx_tmp, idx_path)
                    rate = n_records / max(1.0, time.time() - t_start)
                    pct = 100 * offsets[-1] / target_bytes
                    print(f"  records={n_records:7d} "
                          f"attempts={n_attempts:7d} "
                          f"size={offsets[-1]/2**20:7.2f}/{args.target_mb} "
                          f"MB ({pct:5.1f}%) rate={rate:.1f} pairs/s",
                          flush=True)
        flush_batch()
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
