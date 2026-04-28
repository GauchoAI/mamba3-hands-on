"""orchestrator — composes step-function specialists into solvers.

The Lego principle: each specialist is a tiny f(state) → action.
The orchestrator is a plain Python loop that:
  1. Parses the input task descriptor
  2. Picks the right specialist
  3. Runs it step-by-step against a state tool
  4. Returns the result

No neural composition layer is needed for task DISPATCH — string
matching plus a Python switch handles it. The "composition" that
matters is in the COMPOSED tasks like GCDHANOI below: solving them
requires using multiple specialists in sequence.

This demonstrates:
  - Specialists snap together (frozen, no retraining required)
  - New compositions of existing specialists are free
  - Per-task cost: parsing + a few specialist steps
"""
import sys, time, re
sys.path.insert(0, ".")

import torch
from train_step_function import StepFunctionMLP, step_through_autoregressive
from train_gcd_step import GCDStepMLP, step_through_ar
from hanoi_tool import HanoiTool
from hanoi_step_function import (
    state_for_step as hanoi_state_for_step,
    ACTIONS as HANOI_ACTIONS, ACTION_TO_IDX as HANOI_A2I,
)
from gcd_step_function import (
    state_for_pair, gcd_step, gcd_trajectory,
    ACTIONS as GCD_ACTIONS, ACTION_TO_IDX as GCD_A2I,
)


# ── Specialist registry ─────────────────────────────────────────

class SpecialistRegistry:
    def __init__(self, device="cpu"):
        self.device = device
        self._cache = {}

    def hanoi(self):
        if "hanoi" not in self._cache:
            ck = torch.load("checkpoints/specialists/hanoi_step_fn.pt",
                            map_location=self.device, weights_only=False)
            cfg = ck["config"]
            m = StepFunctionMLP(max_disk=cfg["max_disk"], d_emb=cfg["d_emb"],
                                 d_hidden=cfg["d_hidden"]).to(self.device)
            m.load_state_dict(ck["model"])
            m.eval()
            self._cache["hanoi"] = (m, cfg)
        return self._cache["hanoi"]

    def gcd(self):
        if "gcd" not in self._cache:
            ck = torch.load("checkpoints/specialists/gcd_step.pt",
                            map_location=self.device, weights_only=False)
            m = GCDStepMLP(**ck["config"]).to(self.device)
            m.load_state_dict(ck["model"])
            m.eval()
            self._cache["gcd"] = (m, ck["config"])
        return self._cache["gcd"]


# ── Specialist runners ──────────────────────────────────────────

def run_hanoi(n: int, registry: SpecialistRegistry) -> tuple[list[tuple[int, int, int]], int]:
    """Solve HANOI(n) by running the specialist step-by-step.
    Returns (moves, n_moves). Each move is (disk_id, src_peg, dst_peg)."""
    model, cfg = registry.hanoi()
    tool = HanoiTool(n)
    moves = []
    expected_total = (2 ** n) - 1
    for _ in range(expected_total + 1):  # +1 safety
        s = hanoi_state_for_step(tool)
        s_t = torch.tensor([list(s)], dtype=torch.long, device=registry.device)
        s_t[0, 2:].clamp_(0, cfg["max_disk"])
        with torch.no_grad():
            logits = model(s_t)
        action_idx = int(logits[0].argmax().item())
        src, dst = HANOI_ACTIONS[action_idx]
        # Determine which disk moved (smallest on src)
        moved_disk = None
        for k_idx, p in enumerate(tool.peg):
            if p == src:
                if moved_disk is None or (k_idx + 1) < moved_disk:
                    moved_disk = k_idx + 1
        if moved_disk is None:
            break
        moves.append((moved_disk, src, dst))
        tool.peg[moved_disk - 1] = dst
        tool.move_index += 1
        if tool.move_index >= expected_total:
            break
    return moves, len(moves)


def run_gcd(a: int, b: int, registry: SpecialistRegistry,
            max_steps: int = 2_000_000) -> int:
    """Solve GCD(a, b) by running the specialist step-by-step. Returns gcd."""
    model, cfg = registry.gcd()
    cur_a, cur_b = a, b
    for step in range(max_steps):
        s = state_for_pair(cur_a, cur_b)
        s_t = torch.tensor([list(s)], dtype=torch.long, device=registry.device)
        with torch.no_grad():
            logits = model(s_t)
        act = int(logits[0].argmax().item())
        if act == GCD_A2I["done"]:
            return cur_a
        if act == GCD_A2I["sub_b_from_a"]:
            cur_a -= cur_b
        else:
            cur_b -= cur_a
    return -1  # didn't terminate


# ── The orchestrator (string switch) ────────────────────────────

HANOI_RE = re.compile(r"^HANOI\s+(\d+)\s*$")
GCD_RE = re.compile(r"^GCD\s+(\d+)\s+(\d+)\s*$")
# Composite: GCD of move-counts of HANOI(n) and HANOI(m).
GCDHANOI_RE = re.compile(r"^GCDHANOI\s+(\d+)\s+(\d+)\s*$")


def solve(task: str, registry: SpecialistRegistry) -> dict:
    """Dispatch a task descriptor to the right specialist(s)."""
    task = task.strip()

    m = HANOI_RE.match(task)
    if m:
        n = int(m.group(1))
        moves, count = run_hanoi(n, registry)
        return {"task": task, "n_moves": count,
                "expected_moves": (2 ** n) - 1,
                "first_3_moves": moves[:3], "last_3_moves": moves[-3:]}

    m = GCD_RE.match(task)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        result = run_gcd(a, b, registry)
        from math import gcd as math_gcd
        return {"task": task, "gcd": result, "expected": math_gcd(a, b)}

    m = GCDHANOI_RE.match(task)
    if m:
        n, mn = int(m.group(1)), int(m.group(2))
        # Run HANOI specialist for both n and m to get move counts.
        _, count_n = run_hanoi(n, registry)
        _, count_m = run_hanoi(mn, registry)
        # Then run GCD specialist on the counts.
        result = run_gcd(count_n, count_m, registry)
        from math import gcd as math_gcd
        expected = math_gcd((2 ** n) - 1, (2 ** mn) - 1)
        return {"task": task, "result": result, "expected": expected,
                "intermediate": {"hanoi_n_moves": count_n,
                                 "hanoi_m_moves": count_m}}

    return {"task": task, "error": "unknown task"}


if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    reg = SpecialistRegistry(device=device)
    test_tasks = [
        "HANOI 3",
        "HANOI 7",
        "HANOI 10",
        "GCD 12 8",
        "GCD 100 75",
        "GCD 12345 67890",
        # Composite — uses BOTH specialists
        "GCDHANOI 4 6",   # gcd(15, 63) = 3
        "GCDHANOI 5 7",   # gcd(31, 127) = 1
        "GCDHANOI 6 9",   # gcd(63, 511) = 1
        "GCDHANOI 6 12",  # gcd(63, 4095) = 63
    ]
    for t in test_tasks:
        t0 = time.time()
        result = solve(t, reg)
        dt = time.time() - t0
        print(f"  {t!r:25}  {dt*1000:>6.1f}ms  {result}", flush=True)
