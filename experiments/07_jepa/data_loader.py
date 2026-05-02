"""Memmap loader for the JEPT record stream produced by make_teacher_thoughts.py.

W1.2 of the JEPA-Cortex plan. Random-access reads via the .idx sidecar; per
record we return:

    question:        bytes (the natural-language user prompt)
    response:        bytes (what the teacher generated; what the student trains on)
    thoughts:        np.ndarray[K, D]   float16  (teacher last-layer hiddens)
    thought_byte_pos: np.ndarray[K]     int32    (offsets into `response`)

The trainer's `collate` function turns these into right-padded torch tensors
plus byte/thought masks. Keeping the raw record format in NumPy here means
the loader is independent of MLX vs. PyTorch — either trainer can use it.
"""
from __future__ import annotations
import os
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

MAGIC = 0x4A455054  # "JEPT"
RECORD_HEADER_FMT = "<IIIII"
RECORD_HEADER_SIZE = struct.calcsize(RECORD_HEADER_FMT)

# Logit-projection KD record (from make_logit_kd_corpus.py)
KDTP_MAGIC = 0x4B445450  # "KDTP"
KDTP_HEADER_FMT = "<IIII"
KDTP_HEADER_SIZE = struct.calcsize(KDTP_HEADER_FMT)


@dataclass
class Record:
    question: bytes
    response: bytes
    thoughts: np.ndarray         # (K, D) float16
    thought_byte_pos: np.ndarray  # (K,) int32


@dataclass
class KDTPRecord:
    """Logit-projection KD record: per-byte teacher next-byte distributions
    at BPE boundary positions in the response."""
    question: bytes
    response: bytes
    mark_byte_pos: np.ndarray   # (K,) int32 — student byte position of each mark
    mark_byte_dist: np.ndarray  # (K, 256) uint8 — quantized teacher distribution


class TeacherThoughtsDataset:
    """Random-access memmap reader over a .bin/.idx pair."""

    def __init__(self, basename: str | Path):
        base = Path(basename)
        self.bin_path = base.with_suffix(".bin")
        self.idx_path = base.with_suffix(".idx")
        with open(self.idx_path, "rb") as f:
            self.n = int(np.frombuffer(f.read(8), dtype=np.int64)[0])
            self.offsets = np.frombuffer(f.read(), dtype=np.int64).copy()
        assert len(self.offsets) == self.n + 1, "idx/offsets mismatch"
        # Memmap the bin once; per-record reads are cheap views into it.
        self._mm = np.memmap(self.bin_path, dtype=np.uint8, mode="r")

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> Record:
        if i < 0:
            i += self.n
        if not 0 <= i < self.n:
            raise IndexError(i)
        start = int(self.offsets[i])
        end = int(self.offsets[i + 1])
        buf = bytes(self._mm[start:end])
        magic, q_len, r_len, K, D = struct.unpack(
            RECORD_HEADER_FMT, buf[:RECORD_HEADER_SIZE]
        )
        assert magic == MAGIC, f"bad magic at record {i}: {magic:#x}"
        p = RECORD_HEADER_SIZE
        question = buf[p:p + q_len]; p += q_len
        response = buf[p:p + r_len]; p += r_len
        bp = np.frombuffer(buf[p:p + 4 * K], dtype=np.int32).copy(); p += 4 * K
        th = np.frombuffer(buf[p:p + 2 * K * D], dtype=np.float16).copy().reshape(K, D)
        return Record(question, response, th, bp)


class KDTPDataset:
    """Random-access memmap reader for KDTP records (round-10 logit-KD).

    Same shape as TeacherThoughtsDataset but for the KDTP format:
    per-byte teacher next-byte distributions at BPE boundary positions.
    """

    def __init__(self, basename: str | Path):
        base = Path(basename)
        self.bin_path = base.with_suffix(".bin")
        self.idx_path = base.with_suffix(".idx")
        with open(self.idx_path, "rb") as f:
            self.n = int(np.frombuffer(f.read(8), dtype=np.int64)[0])
            self.offsets = np.frombuffer(f.read(), dtype=np.int64).copy()
        assert len(self.offsets) == self.n + 1, "idx/offsets mismatch"
        self._mm = np.memmap(self.bin_path, dtype=np.uint8, mode="r")

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> KDTPRecord:
        if i < 0:
            i += self.n
        if not 0 <= i < self.n:
            raise IndexError(i)
        start = int(self.offsets[i])
        end = int(self.offsets[i + 1])
        buf = bytes(self._mm[start:end])
        magic, q_len, r_len, K = struct.unpack(
            KDTP_HEADER_FMT, buf[:KDTP_HEADER_SIZE]
        )
        assert magic == KDTP_MAGIC, f"bad KDTP magic at record {i}: {magic:#x}"
        p = KDTP_HEADER_SIZE
        question = buf[p:p + q_len]; p += q_len
        response = buf[p:p + r_len]; p += r_len
        mp = np.frombuffer(buf[p:p + 4 * K], dtype=np.int32).copy(); p += 4 * K
        md = np.frombuffer(buf[p:p + 256 * K],
                            dtype=np.uint8).copy().reshape(K, 256)
        return KDTPRecord(question, response, mp, md)


# ---------------------------------------------------------------------------
# Collate: pack a list of Records into right-padded torch tensors
# ---------------------------------------------------------------------------
@dataclass
class Batch:
    tokens:           torch.Tensor   # (B, L_max)  long       — bytes the student reads
    byte_pad_mask:    torch.Tensor   # (B, L_max)  bool       — True where real, False where pad
    prompt_lens:      torch.Tensor   # (B,)        long       — where each response begins (== len(question)+1 for the '\n' join)
    teacher_thoughts: torch.Tensor   # (B, K_max, D)  float32 — teacher hiddens
    thought_byte_pos: torch.Tensor   # (B, K_max)  long       — student byte index for each thought (in the joined byte stream)
    thought_pad_mask: torch.Tensor   # (B, K_max)  bool
    # KDTP-only fields (round 10). Empty (size-0 K dim) for non-KD batches.
    kd_byte_pos:      torch.Tensor = None  # (B, K_kd_max) long
    kd_byte_dist:     torch.Tensor = None  # (B, K_kd_max, 256) float32 (dequantized)
    kd_pad_mask:      torch.Tensor = None  # (B, K_kd_max) bool


def collate_records(records: list[Record],
                    max_bytes: int = 512,
                    join_sep: bytes = b"\n") -> Batch:
    """Build training tensors from a list of dataset records.

    Layout per sample (before truncation):
        <question_bytes><sep><response_bytes>
    Thoughts are reported relative to `response`; we shift them by
    `len(question_bytes) + len(sep)` so they align with the joined byte stream.

    Truncation policy: if the joined stream exceeds max_bytes we right-truncate
    and drop any thoughts whose target byte position falls past the end. Most
    records fit; the long tail is mostly very chatty Qwen outputs we don't
    need to memorize verbatim.
    """
    B = len(records)
    seqs: list[np.ndarray] = []
    prompt_lens: list[int] = []
    thought_byte_pos_list: list[np.ndarray] = []
    thoughts_list: list[np.ndarray] = []
    D = records[0].thoughts.shape[1] if records else 0

    for r in records:
        q = r.question
        joined = q + join_sep + r.response
        if len(joined) > max_bytes:
            joined = joined[:max_bytes]
        seqs.append(np.frombuffer(joined, dtype=np.uint8).copy())
        prompt_lens.append(len(q) + len(join_sep))
        # Shift thought byte positions into joined coords.
        bp = r.thought_byte_pos.astype(np.int64) + (len(q) + len(join_sep))
        keep = bp < len(joined)
        thought_byte_pos_list.append(bp[keep])
        thoughts_list.append(r.thoughts[keep])

    L_max = max((len(s) for s in seqs), default=0)
    K_max = max((len(bp) for bp in thought_byte_pos_list), default=0)

    tokens = np.zeros((B, L_max), dtype=np.int64)
    byte_pad = np.zeros((B, L_max), dtype=bool)
    for i, s in enumerate(seqs):
        tokens[i, :len(s)] = s
        byte_pad[i, :len(s)] = True

    teacher_thoughts = np.zeros((B, max(K_max, 1), D), dtype=np.float32)
    thought_byte_pos = np.zeros((B, max(K_max, 1)), dtype=np.int64)
    thought_pad = np.zeros((B, max(K_max, 1)), dtype=bool)
    for i, (bp, th) in enumerate(zip(thought_byte_pos_list, thoughts_list)):
        k = len(bp)
        if k > 0:
            teacher_thoughts[i, :k] = th.astype(np.float32, copy=False)
            thought_byte_pos[i, :k] = bp
            thought_pad[i, :k] = True

    return Batch(
        tokens=torch.from_numpy(tokens),
        byte_pad_mask=torch.from_numpy(byte_pad),
        prompt_lens=torch.tensor(prompt_lens, dtype=torch.long),
        teacher_thoughts=torch.from_numpy(teacher_thoughts),
        thought_byte_pos=torch.from_numpy(thought_byte_pos),
        thought_pad_mask=torch.from_numpy(thought_pad),
    )


# ---------------------------------------------------------------------------
# Iterators: random batch sampler + bilingual mixin
# ---------------------------------------------------------------------------
class TeacherIterator:
    """Infinite random batch iterator over the dataset."""

    def __init__(self, ds: TeacherThoughtsDataset, batch_size: int,
                 max_bytes: int = 512, seed: int = 0):
        self.ds = ds
        self.batch_size = batch_size
        self.max_bytes = max_bytes
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        return self

    def __next__(self) -> Batch:
        idx = self.rng.integers(0, len(self.ds), size=self.batch_size)
        recs = [self.ds[int(i)] for i in idx]
        return collate_records(recs, max_bytes=self.max_bytes)


# ---------------------------------------------------------------------------
# KDTP collate + iterator (round 10 — logit-projection KD)
# ---------------------------------------------------------------------------
def collate_kdtp_records(records: list[KDTPRecord],
                          max_bytes: int = 512,
                          join_sep: bytes = b"\n") -> Batch:
    """Build a Batch with kd_* fields populated. Same byte-stream layout
    as collate_records (question + sep + response); marks are shifted
    into joined coords. uint8 dist is dequantized to float32 / 255."""
    B = len(records)
    seqs: list[np.ndarray] = []
    prompt_lens: list[int] = []
    pos_list: list[np.ndarray] = []
    dist_list: list[np.ndarray] = []

    for r in records:
        q = r.question
        joined = q + join_sep + r.response
        if len(joined) > max_bytes:
            joined = joined[:max_bytes]
        seqs.append(np.frombuffer(joined, dtype=np.uint8).copy())
        prompt_lens.append(len(q) + len(join_sep))
        # Marks are stored relative to response coords inside the
        # response. Shift into joined coords.
        bp = r.mark_byte_pos.astype(np.int64) + (len(q) + len(join_sep))
        keep = bp < len(joined)
        pos_list.append(bp[keep])
        dist_list.append(r.mark_byte_dist[keep])

    L_max = max((len(s) for s in seqs), default=0)
    K_max = max((len(bp) for bp in pos_list), default=0)
    if K_max == 0:
        K_max = 1  # placeholder shape

    tokens = np.zeros((B, L_max), dtype=np.int64)
    byte_pad = np.zeros((B, L_max), dtype=bool)
    for i, s in enumerate(seqs):
        tokens[i, :len(s)] = s
        byte_pad[i, :len(s)] = True

    kd_pos = np.zeros((B, K_max), dtype=np.int64)
    kd_dist = np.zeros((B, K_max, 256), dtype=np.float32)
    kd_pad = np.zeros((B, K_max), dtype=bool)
    for i, (bp, dist) in enumerate(zip(pos_list, dist_list)):
        k = len(bp)
        if k > 0:
            kd_pos[i, :k] = bp
            # Dequantize uint8 → float32 in [0, 1]; renormalize each row
            # so it sums to 1 (uint8 quantization can drift).
            d = dist.astype(np.float32) / 255.0
            d = d / np.clip(d.sum(axis=-1, keepdims=True), 1e-6, None)
            kd_dist[i, :k] = d
            kd_pad[i, :k] = True

    return Batch(
        tokens=torch.from_numpy(tokens),
        byte_pad_mask=torch.from_numpy(byte_pad),
        prompt_lens=torch.tensor(prompt_lens, dtype=torch.long),
        # Empty thought fields — KDTP records don't carry thoughts.
        teacher_thoughts=torch.zeros(B, 1, 1),
        thought_byte_pos=torch.zeros(B, 1, dtype=torch.long),
        thought_pad_mask=torch.zeros(B, 1, dtype=torch.bool),
        kd_byte_pos=torch.from_numpy(kd_pos),
        kd_byte_dist=torch.from_numpy(kd_dist),
        kd_pad_mask=torch.from_numpy(kd_pad),
    )


class KDTPIterator:
    """Infinite random batch iterator over the KDTP dataset."""

    def __init__(self, ds: KDTPDataset, batch_size: int,
                 max_bytes: int = 512, seed: int = 0):
        self.ds = ds
        self.batch_size = batch_size
        self.max_bytes = max_bytes
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        return self

    def __next__(self) -> Batch:
        idx = self.rng.integers(0, len(self.ds), size=self.batch_size)
        recs = [self.ds[int(i)] for i in idx]
        return collate_kdtp_records(recs, max_bytes=self.max_bytes)


class BilingualByteIterator:
    """Plain byte iterator over a text file (e.g. data/bilingual.txt).

    Used as the safety-belt 25% mixin in the JEPA trainer. No thoughts —
    we only compute byte CE on these batches. Keeps the model from
    over-fitting the teacher distribution.
    """

    def __init__(self, path: str | Path, batch_size: int, seq_len: int = 256,
                 seed: int = 0):
        self.data = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        return self

    def __next__(self) -> Batch:
        starts = self.rng.integers(0, len(self.data) - self.seq_len - 1,
                                   size=self.batch_size)
        tokens = np.stack(
            [self.data[s:s + self.seq_len].astype(np.int64) for s in starts]
        )
        B, L = tokens.shape
        return Batch(
            tokens=torch.from_numpy(tokens),
            byte_pad_mask=torch.ones(B, L, dtype=torch.bool),
            prompt_lens=torch.zeros(B, dtype=torch.long),
            teacher_thoughts=torch.zeros(B, 1, 1),     # unused for biling batches
            thought_byte_pos=torch.zeros(B, 1, dtype=torch.long),
            thought_pad_mask=torch.zeros(B, 1, dtype=torch.bool),
        )


class CountingByteIterator:
    """Synthetic unary `***:aaa\\n` iterator. Same format as cortex_counting.py."""

    def __init__(self, batch_size: int, n_min: int = 1, n_max: int = 30,
                 seq_len: int = 80, seed: int = 0):
        self.batch_size = batch_size
        self.n_min, self.n_max = n_min, n_max
        self.seq_len = seq_len
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        return self

    def __next__(self) -> Batch:
        B, L = self.batch_size, self.seq_len
        tokens = np.zeros((B, L), dtype=np.int64)
        mask = np.zeros((B, L), dtype=bool)
        for i in range(B):
            n = int(self.rng.integers(self.n_min, self.n_max + 1))
            s = "*" * n + ":" + "a" * n + "\n"
            b = s.encode("utf-8")[:L]
            tokens[i, :len(b)] = np.frombuffer(b, dtype=np.uint8)
            mask[i, :len(b)] = True
        return Batch(
            tokens=torch.from_numpy(tokens),
            byte_pad_mask=torch.from_numpy(mask),
            prompt_lens=torch.zeros(B, dtype=torch.long),
            teacher_thoughts=torch.zeros(B, 1, 1),
            thought_byte_pos=torch.zeros(B, 1, dtype=torch.long),
            thought_pad_mask=torch.zeros(B, 1, dtype=torch.bool),
        )
