"""Binary batch writer — the Python side of the wire protocol with ptxd.

Mirrors `engine/ptx/src/batch_format.rs`. Layout (little-endian):

    [magic: u32 = 0x42544348]   ('BTCH')
    [version: u32 = 1]
    [n_examples: u32]
    [flags: u32 = 0]            (reserved)
    for each example:
      [n_tokens: u32]
      [tokens:   u32 * n_tokens]
      [targets:  u32 * n_tokens]   (0xFFFFFFFF = ignore)

Used by ptxd_specialist.py and any future GA worker that wants ptxd
to train on a particular set of examples. Keeping the format dead-
simple — no compression, no chunking — because cycle-sized files are
~10 MB and disk I/O is not the bottleneck (kernel launches are).
"""
import struct
from pathlib import Path

BATCH_MAGIC   = 0x42544348
BATCH_VERSION = 1
IGNORE        = 0xFFFFFFFF


def write_examples(path, examples):
    """Write a list of (tokens, targets) tuples to a binary batch file.

    `tokens` and `targets` are integer iterables of equal length. Use
    `IGNORE` (= u32::MAX) at any position you don't want supervised.

    Returns: number of bytes written.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        f.write(struct.pack("<4I", BATCH_MAGIC, BATCH_VERSION, len(examples), 0))
        for tokens, targets in examples:
            assert len(tokens) == len(targets), \
                f"len mismatch: tokens={len(tokens)} targets={len(targets)}"
            n = len(tokens)
            f.write(struct.pack("<I", n))
            f.write(struct.pack(f"<{n}I", *tokens))
            f.write(struct.pack(f"<{n}I", *targets))
    return p.stat().st_size


def read_examples(path):
    """Read a batch file back. Inverse of `write_examples`. Used for
    testing the round-trip; the Rust reader is what production uses."""
    with open(path, "rb") as f:
        magic, version, n_examples, _flags = struct.unpack("<4I", f.read(16))
        assert magic == BATCH_MAGIC, f"bad magic: 0x{magic:x}"
        assert version == BATCH_VERSION, f"bad version: {version}"
        examples = []
        for _ in range(n_examples):
            (n,) = struct.unpack("<I", f.read(4))
            tokens  = list(struct.unpack(f"<{n}I", f.read(4 * n)))
            targets = list(struct.unpack(f"<{n}I", f.read(4 * n)))
            examples.append((tokens, targets))
    return examples


if __name__ == "__main__":
    # Self-test: round-trip a tiny example set.
    import tempfile, os
    examples = [
        ([256, 48, 32, 49, 258, 83, 257],
         [IGNORE, IGNORE, IGNORE, IGNORE, 83, IGNORE, IGNORE]),
        ([256, 49, 32, 48, 258, 68, 257],
         [IGNORE, IGNORE, IGNORE, IGNORE, 68, IGNORE, IGNORE]),
    ]
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        write_examples(tmp.name, examples)
        roundtripped = read_examples(tmp.name)
        os.unlink(tmp.name)
    assert roundtripped == examples, "round-trip mismatch"
    print(f"PASS — round-tripped {len(examples)} examples")
