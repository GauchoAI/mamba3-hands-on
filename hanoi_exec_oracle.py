"""hanoi_exec_oracle — generate per-step register-execution traces
for the Tower of Hanoi.

Used by the training task to teach a small Mamba-3 + RegisterBank to
EXECUTE the Hanoi algorithm (not just emit the final answer count).

Register layout (indices 0..n-1 hold disk pegs; reg N holds move
counter; rest are scratch):

    reg[0..n-1] : peg of disk k+1   (0=A, 1=B, 2=C)
    reg[n]      : current move number (0..2^n-1)

For each move (k, src, dst), the model executes one ATOMIC STEP of:
  - read reg[k-1] (verify the peg state of disk k)
  - emit the move tokens "k src dst\n"
  - write reg[k-1] = dst

Token emission is byte-by-byte (small vocab), but the register I/O
is bundled into ONE step to keep the trace compact. The trace is a
list of (action_kind, payload) tuples per timestep:

    ('emit_token', byte)
    ('emit_token', byte) ... (one per byte of "k src dst\\n")
    ('register_op', read_addr, write_addr, write_val)  # at the LAST byte of the move

Total len = 6*M tokens per move (assuming single-digit n). For n=3
that's 6*7 = 42 tokens. For n=10 that's 6*1023 ≈ 6k tokens.

This is the simplest-correct encoding; we can refine the per-step
distribution of read/write later (e.g., interleave writes through
the move emission). For now: write happens at the end of each move.
"""
from typing import List, Tuple

# Peg labels
PEG_NAMES = ['A', 'B', 'C']
NO_OP = -1  # used for "no-read" / "no-write" in the trace


def hanoi_moves(n: int):
    """Recursive Hanoi: yield (disk, src_peg, dst_peg) tuples.
    pegs are 0=A, 1=B, 2=C."""
    moves = []
    def rec(n_, src, dst, aux):
        if n_ == 1:
            moves.append((1, src, dst))
            return
        rec(n_ - 1, src, aux, dst)
        moves.append((n_, src, dst))
        rec(n_ - 1, aux, dst, src)
    rec(n, 0, 2, 1)  # always solve A -> C with B as aux
    return moves


def encode_move(k: int, src: int, dst: int) -> List[int]:
    """Bytes for one move line: e.g. '3 A C\\n' -> [51, 32, 65, 32, 67, 10].
    Always 6 bytes (single-digit k for n<=9; for n>9 it's 7 bytes).
    """
    s = f"{k} {PEG_NAMES[src]} {PEG_NAMES[dst]}\n"
    return list(s.encode("ascii"))


def gen_exec_trace(n: int, n_registers: int = 16):
    """Build the full per-timestep training trace for Hanoi(n).

    Returns a list of timestep records. Each record is a dict:
        token: int (byte to predict at this position)
        read_addr: int  (which register to read for next step;
                         n_registers means no-read)
        write_addr: int  (which register to write THIS step;
                          n_registers means no-write)
        write_val: int   (value to write; meaningful only if
                          write_addr < n_registers)
        registers_after: list[int]  (expected register state
                                     AFTER this step, length n_registers)

    The trace is designed so that:
      - On the LAST byte of each move, the register write happens
        (peg of disk k := dst).
      - The READ of register k happens on the FIRST byte of each move
        (so that on the second byte the model has the read result and
        can use it to choose the move output).
    """
    moves = hanoi_moves(n)
    NO_REG = n_registers  # "no-op" address index

    # Initialise registers: all disks on peg 0 (A); move counter at 0.
    registers = [0] * n_registers
    # reg[0..n-1] are disk pegs; reg[n] is move counter.

    trace = []

    for move_idx, (k, src, dst) in enumerate(moves):
        bytes_ = encode_move(k, src, dst)
        for byte_idx, b in enumerate(bytes_):
            # Read register k-1 on the FIRST byte (offers the model the
            # current peg of disk k as feature input on the NEXT byte).
            read_addr = (k - 1) if byte_idx == 0 else NO_REG
            # Write register k-1 := dst on the LAST byte (commits the
            # move to state after the model has emitted the full record).
            if byte_idx == len(bytes_) - 1:
                write_addr = k - 1
                write_val = dst
            else:
                write_addr = NO_REG
                write_val = 0
            # Apply the write to our running register snapshot AFTER
            # this step (so registers_after reflects post-step state).
            if write_addr < NO_REG:
                registers = list(registers)
                registers[write_addr] = write_val
            trace.append({
                "token": b,
                "read_addr": read_addr,
                "write_addr": write_addr,
                "write_val": write_val,
                "registers_after": list(registers),
            })

    return trace, moves


# ───────────────── Smoke test ─────────────────

def _smoke():
    moves = hanoi_moves(3)
    print(f"Hanoi(3) moves ({len(moves)}):")
    for m in moves:
        k, s, d = m
        print(f"  disk {k}: {PEG_NAMES[s]} -> {PEG_NAMES[d]}")

    trace, _ = gen_exec_trace(3, n_registers=16)
    print(f"\nTrace length: {len(trace)}")
    print(f"First 12 records:")
    for i, r in enumerate(trace[:12]):
        ch = chr(r["token"]) if 32 <= r["token"] < 127 else f"\\x{r['token']:02x}"
        ra = r["read_addr"]
        wa = r["write_addr"]
        ra_str = f"r{ra}" if ra < 16 else "—"
        wa_str = f"r{wa}={r['write_val']}" if wa < 16 else "—"
        print(f"  [{i:2d}]  emit {ch!r:5}  read={ra_str:5}  write={wa_str:8}  regs={r['registers_after'][:5]}")

    # Sanity: total bytes for n=3 should be 7 moves * 6 bytes = 42
    assert len(trace) == 7 * 6, f"expected 42, got {len(trace)}"

    # End state: disk 1 on peg C (reg[0]=2), disk 2 on peg C (reg[1]=2), disk 3 on peg C (reg[2]=2)
    final_regs = trace[-1]["registers_after"]
    assert final_regs[:3] == [2, 2, 2], f"final reg[0:3]={final_regs[:3]} != [2,2,2]"
    print("\n✓ Trace structurally correct, final state has all disks on peg C")


if __name__ == "__main__":
    _smoke()
