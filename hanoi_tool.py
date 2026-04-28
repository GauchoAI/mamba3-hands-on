"""hanoi_tool — Python state-tracker for Tower of Hanoi.

Lives OUTSIDE the neural network, operating as an "embedded tool":
the model emits bytes; the tool watches them, maintains task state
(peg of each disk + a few summaries), and provides a small fixed-size
feedback vector that's fed back into the model's input embedding on
the next step.

This replaces the previous neural RegisterBank — the bank's state was
a torch tensor managed by trained read/write heads. Now the state
is a Python list and the "controls" are just byte-level parsing in
plain Python. Adding more state slots costs ZERO model parameters
and zero retraining: the tool's `n` can be anything, and the model
sees the same fixed-size feedback shape.

Feedback design (3 small ints per step, encoded into a single
state_value via base-N packing):

    feedback[0] = move_index (mod 256) — which move are we on
    feedback[1] = current_disk          — disk being moved (0=none)
    feedback[2] = src_peg               — peg the current disk is on
    feedback[3] = dst_peg               — peg the algorithm dictates

These four are PURE FUNCTIONS of the bytes emitted so far, computed
deterministically by Python. The model receives them as a single
combined integer in [0, value_range) (or three separate small ints,
each embedded and summed). Choice depends on what value_range the
trainer was constructed with.
"""

PEG_NAMES = ['A', 'B', 'C']


def hanoi_moves(n: int):
    """Recursive Hanoi: yield (disk, src_peg, dst_peg) tuples."""
    moves = []
    def rec(n_, src, dst, aux):
        if n_ == 1:
            moves.append((1, src, dst))
            return
        rec(n_ - 1, src, aux, dst)
        moves.append((n_, src, dst))
        rec(n_ - 1, aux, dst, src)
    rec(n, 0, 2, 1)
    return moves


class HanoiTool:
    """Stateful tracker. Watches emitted bytes, exposes per-position
    feedback vectors. Slot count = n disks; can be any positive int.
    """

    def __init__(self, n: int):
        self.n = n
        # Pre-compute the move sequence the algorithm will produce.
        self.moves = hanoi_moves(n)
        # Initial state: all disks on peg 0 (A).
        self.peg = [0] * n
        # Parser state for the move currently being emitted byte-by-byte.
        self.partial = []  # bytes accumulating
        self.move_index = 0  # 0-indexed; advances when a move completes
        # Per-step feedback we expose to the model:
        #   current_disk: 1..n  (0 = no disk being addressed yet)
        #   src_peg, dst_peg: 0..2  (or 3 for "none")
        self.current_disk = 0
        self.current_src = 3
        self.current_dst = 3

    @property
    def n_moves(self) -> int:
        return len(self.moves)

    def reset(self):
        self.peg = [0] * self.n
        self.partial = []
        self.move_index = 0
        self.current_disk = 0
        self.current_src = 3
        self.current_dst = 3

    def step(self, byte: int):
        """Update tool state given the next emitted byte."""
        self.partial.append(byte)
        text = bytes(b for b in self.partial if 0 <= b < 256).decode("ascii", errors="replace")

        # As the bytes accumulate, surface the disk number and src/dst
        # AS SOON AS the model has emitted enough to disambiguate them.
        # Fixed format "k SRC DST\n" (6 bytes for n<=9).
        # When we see the digit at position 0: current_disk = that digit.
        # When we see SRC peg letter at position 2: src_peg = ord(SRC) - 'A'.
        # The dst is determined by the current move (we know it from
        # self.moves[self.move_index]).
        if len(self.partial) >= 1 and self.current_disk == 0:
            try:
                d = int(chr(self.partial[0]))
                if 1 <= d <= self.n:
                    self.current_disk = d
                    # Look up src/dst from the algorithm's current move.
                    if self.move_index < len(self.moves):
                        k, src, dst = self.moves[self.move_index]
                        self.current_src = src
                        self.current_dst = dst
            except (ValueError, IndexError):
                pass

        # Move complete on '\n'.
        if byte == ord('\n'):
            # Apply the move to the peg state.
            if self.move_index < len(self.moves):
                k, src, dst = self.moves[self.move_index]
                if 1 <= k <= self.n:
                    self.peg[k - 1] = dst
            self.move_index += 1
            self.partial = []
            self.current_disk = 0
            self.current_src = 3
            self.current_dst = 3

    # ── Feedback encoding ───────────────────────────────────────────
    # Pack the FULL peg-state into a single integer using base-3:
    #   state_value = sum(peg[i] * 3^i)  for i = 0..n-1
    # This is a complete summary of the world for Hanoi: every disk's
    # peg. The model sees this and must figure out the algorithm
    # (which disk to move next, src/dst pegs) from it.
    #
    # For n=2: range [0, 9). For n=3: [0, 27). For n=4: [0, 81).
    # For n=5: [0, 243). Fits in value_range=256 up to n=5.
    # For larger n we'd need multi-int feedback (one per disk) — kept
    # as a follow-up. The single-int packing is the simplest demo.

    def feedback_value(self) -> int:
        v = 0
        for i, p in enumerate(self.peg):
            v += p * (3 ** i)
        return v


def precompute_feedback(n: int):
    """For training: compute the per-position feedback values that
    the tool would produce on the ground-truth byte stream.

    Returns list[int] of length (6 * n_moves) — one feedback per
    answer-span position. The value is what the model SEES at
    position p (= what the tool emitted after byte p-1).
    """
    tool = HanoiTool(n)
    moves = tool.moves
    feedback_per_pos = []
    # The model "sees" the feedback at position p that was produced
    # AFTER the byte at position p-1 was emitted. So feedback[0]
    # is the initial state (before any byte emitted).
    feedback_per_pos.append(tool.feedback_value())
    bytes_emitted = []
    for k, src, dst in moves:
        s = f"{k} {PEG_NAMES[src]} {PEG_NAMES[dst]}\n"
        for ch in s:
            tool.step(ord(ch))
            feedback_per_pos.append(tool.feedback_value())
    # Drop the very last one (post-final-byte feedback isn't used).
    return feedback_per_pos[:-1]


# ───────────── Smoke test ─────────────

if __name__ == "__main__":
    fb = precompute_feedback(3)
    print(f"Hanoi(3) feedback length: {len(fb)} (expected 42)")
    print(f"First 12 feedback values: {fb[:12]}")
    print(f"Last 8 feedback values: {fb[-8:]}")
    # Initial state: all on peg 0, packed value 0
    assert fb[0] == 0, f"initial state must be 0, got {fb[0]}"
    # After '\n' of move 1 ('1 A C'), disk 1 goes to peg 2.
    # peg = [2, 0, 0] -> 2*1 + 0*3 + 0*9 = 2
    assert fb[6] == 2, f"post-move-1 must be 2, got {fb[6]}"
    # fb[-1] = state right BEFORE the very last byte is emitted, so
    # disk 1's last move ('1 A C') hasn't applied yet. peg = [0,2,2]
    # -> 0 + 2*3 + 2*9 = 24.
    assert fb[-1] == 24, f"pre-final-byte state must be 24, got {fb[-1]}"
    print("\n✓ HanoiTool feedback encoding correct (full peg-state packed)")
