"""Binary batch writer — the Python side of the wire protocol with ptxd.

Mirrors `engine/ptx/src/batch_format.rs`. Layout (little-endian):

    [magic: u32 = 0x42544348]   ('BTCH')
    [version: u32 = 1 or 2]
    [n_examples: u32]
    [flags: u32]                bit 0 = has_teacher_logits (v2 only)
    for each example:
      [n_tokens: u32]
      [tokens:   u32 * n_tokens]
      [targets:  u32 * n_tokens]   (0xFFFFFFFF = ignore)
      if has_teacher_logits:
        [vocab_size: u32]
        [n_supervised: u32]
        [pos: u32]                 ┐ for each supervised position,
        [logits: f32 * vocab_size] ┘ in left-to-right token order

v2 carries optional teacher logits per supervised position — input
to Loss::CeKd. v1 readers/writers (no teacher logits) keep working.
"""
import struct
from pathlib import Path

BATCH_MAGIC               = 0x42544348
BATCH_VERSION_V1          = 1
BATCH_VERSION_V2          = 2
BATCH_VERSION             = BATCH_VERSION_V2
FLAG_HAS_TEACHER_LOGITS   = 1 << 0
IGNORE                    = 0xFFFFFFFF


def write_examples(path, examples, teacher_logits=None, vocab_size=None):
    """Write a list of (tokens, targets) tuples to a binary batch file.

    `tokens` and `targets` are integer iterables of equal length. Use
    `IGNORE` (= u32::MAX) at any position you don't want supervised.

    `teacher_logits` (optional): list parallel to `examples`. Each
    entry is a list of (pos, logits_for_vocab) tuples — one per
    supervised position. When provided, writes v2 with the teacher_logits
    flag set; `vocab_size` must also be provided.

    Returns: number of bytes written.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    has_teacher = teacher_logits is not None
    if has_teacher:
        assert vocab_size is not None, "vocab_size required when teacher_logits provided"
        assert len(teacher_logits) == len(examples), "teacher_logits len mismatch"
        version = BATCH_VERSION_V2
        flags = FLAG_HAS_TEACHER_LOGITS
    else:
        version = BATCH_VERSION_V1
        flags = 0
    with open(p, "wb") as f:
        f.write(struct.pack("<4I", BATCH_MAGIC, version, len(examples), flags))
        for i, (tokens, targets) in enumerate(examples):
            assert len(tokens) == len(targets), \
                f"len mismatch: tokens={len(tokens)} targets={len(targets)}"
            n = len(tokens)
            f.write(struct.pack("<I", n))
            f.write(struct.pack(f"<{n}I", *tokens))
            f.write(struct.pack(f"<{n}I", *targets))
            if has_teacher:
                slots = teacher_logits[i]
                f.write(struct.pack("<2I", vocab_size, len(slots)))
                for pos, logits in slots:
                    assert len(logits) == vocab_size
                    f.write(struct.pack("<I", pos))
                    f.write(struct.pack(f"<{vocab_size}f", *logits))
    return p.stat().st_size


def read_examples(path):
    """Read a batch file back. Inverse of `write_examples`. Used for
    testing the round-trip; the Rust reader is what production uses.
    Returns a list of (tokens, targets) tuples for v1, or a list of
    (tokens, targets, teacher_slots) tuples for v2 with teacher logits."""
    with open(path, "rb") as f:
        magic, version, n_examples, flags = struct.unpack("<4I", f.read(16))
        assert magic == BATCH_MAGIC, f"bad magic: 0x{magic:x}"
        assert version in (BATCH_VERSION_V1, BATCH_VERSION_V2), f"bad version: {version}"
        has_teacher = (version >= BATCH_VERSION_V2) and (flags & FLAG_HAS_TEACHER_LOGITS) != 0
        examples = []
        for _ in range(n_examples):
            (n,) = struct.unpack("<I", f.read(4))
            tokens  = list(struct.unpack(f"<{n}I", f.read(4 * n)))
            targets = list(struct.unpack(f"<{n}I", f.read(4 * n)))
            if has_teacher:
                v, n_sup = struct.unpack("<2I", f.read(8))
                slots = []
                for _ in range(n_sup):
                    (pos,) = struct.unpack("<I", f.read(4))
                    logits = list(struct.unpack(f"<{v}f", f.read(4 * v)))
                    slots.append((pos, logits))
                examples.append((tokens, targets, slots))
            else:
                examples.append((tokens, targets))
    return examples


if __name__ == "__main__":
    # Self-test: round-trip without and with teacher logits.
    import tempfile, os
    examples = [
        ([256, 48, 32, 49, 258, 83, 257],
         [IGNORE, IGNORE, IGNORE, IGNORE, 83, IGNORE, IGNORE]),
        ([256, 49, 32, 48, 258, 68, 257],
         [IGNORE, IGNORE, IGNORE, IGNORE, 68, IGNORE, IGNORE]),
    ]
    # v1 (no teacher logits)
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        write_examples(tmp.name, examples)
        rt = read_examples(tmp.name)
        os.unlink(tmp.name)
    assert rt == examples, "v1 round-trip mismatch"
    print(f"PASS v1 — round-tripped {len(examples)} examples (no teacher)")

    # v2 (with teacher logits)
    V = 8  # tiny vocab for test
    teacher_logits = [
        [(4, [0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7])],   # ex0: one supervised pos
        [(4, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5])],   # ex1: one supervised pos
    ]
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        write_examples(tmp.name, examples, teacher_logits=teacher_logits, vocab_size=V)
        rt = read_examples(tmp.name)
        os.unlink(tmp.name)
    assert len(rt) == 2 and len(rt[0]) == 3, "v2 should return (tokens, targets, teacher) tuples"
    for orig_ex, orig_tch, got in zip(examples, teacher_logits, rt):
        assert got[0] == orig_ex[0]
        assert got[1] == orig_ex[1]
        assert len(got[2]) == len(orig_tch)
        for (op, ol), (gp, gl) in zip(orig_tch, got[2]):
            assert op == gp
            assert all(abs(a - b) < 1e-6 for a, b in zip(ol, gl))
    print(f"PASS v2 — round-tripped {len(examples)} examples WITH teacher logits")
