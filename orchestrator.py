"""orchestrator — composes step-function specialists into solvers.

Currently registered Legos:
    HANOI    n            -> Tower of Hanoi: emit move sequence
    GCD      a b          -> Euclidean GCD by subtraction
    CONWAY   "<grid>"     -> apply one Game-of-Life step to a grid
    BUBBLE   "1,3,2,…"    -> bubble sort the list
    MAZE     x0 y0 x1 y1  -> greedy navigation start -> goal

Composites (demonstrate combining frozen specialists):
    GCDHANOI    n m        -> gcd(2^n - 1, 2^m - 1)
    CONWAYSTABLE "<grid>"  -> iterate Conway until grid is unchanged
    SORTHANOI   n          -> Hanoi(n)'s move-counts sorted
    MAZESTEPS   x0 y0 x1 y1 -> path length the maze specialist takes
    GCDSORTED   "1,3,2,…"  -> bubble sort, then GCD of (max, min)

The composites use ZERO new training — they're pure Python plumbing
over frozen step-function specialists. This is the "lazy composition"
that scales as the library grows.
"""
import sys, time, re
sys.path.insert(0, ".")

import torch

# --- specialist module imports ---
from train_step_function import StepFunctionMLP
from train_gcd_step import GCDStepMLP, step_through_ar as gcd_run
from train_conway_step import ConwayStepMLP
from train_bubble_step import BubbleStepMLP
from train_maze_step import MazeStepMLP

# --- specialist runtime helpers ---
from hanoi_tool import HanoiTool
from hanoi_step_function import (
    state_for_step as hanoi_state_for_step,
    ACTIONS as HANOI_ACTIONS, ACTION_TO_IDX as HANOI_A2I,
)
from gcd_step_function import (
    state_for_pair, ACTIONS as GCD_ACTIONS, ACTION_TO_IDX as GCD_A2I,
)
from conway_step_function import step_grid as conway_ref_step
from bubble_step_function import bubble_sort_with_step
from maze_step_function import navigate as maze_navigate, ACTIONS as MAZE_ACTIONS


# ── Specialist registry (lazy load) ─────────────────────────────

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
            m.load_state_dict(ck["model"]); m.eval()
            self._cache["hanoi"] = (m, cfg)
        return self._cache["hanoi"]

    def gcd(self):
        if "gcd" not in self._cache:
            ck = torch.load("checkpoints/specialists/gcd_step.pt",
                            map_location=self.device, weights_only=False)
            m = GCDStepMLP(**ck["config"]).to(self.device)
            m.load_state_dict(ck["model"]); m.eval()
            self._cache["gcd"] = (m, ck["config"])
        return self._cache["gcd"]

    def conway(self):
        if "conway" not in self._cache:
            ck = torch.load("checkpoints/specialists/conway_step.pt",
                            map_location=self.device, weights_only=False)
            m = ConwayStepMLP(**ck["config"]).to(self.device)
            m.load_state_dict(ck["model"]); m.eval()
            self._cache["conway"] = (m, ck["config"])
        return self._cache["conway"]

    def bubble(self):
        if "bubble" not in self._cache:
            ck = torch.load("checkpoints/specialists/bubble_step.pt",
                            map_location=self.device, weights_only=False)
            m = BubbleStepMLP(**ck["config"]).to(self.device)
            m.load_state_dict(ck["model"]); m.eval()
            self._cache["bubble"] = (m, ck["config"])
        return self._cache["bubble"]

    def maze(self):
        if "maze" not in self._cache:
            ck = torch.load("checkpoints/specialists/maze_step.pt",
                            map_location=self.device, weights_only=False)
            m = MazeStepMLP(**ck["config"]).to(self.device)
            m.load_state_dict(ck["model"]); m.eval()
            self._cache["maze"] = (m, ck["config"])
        return self._cache["maze"]


# ── Specialist runners ──────────────────────────────────────────

def run_hanoi(n: int, reg: SpecialistRegistry):
    model, cfg = reg.hanoi()
    tool = HanoiTool(n)
    moves = []
    expected_total = (2 ** n) - 1
    for _ in range(expected_total + 1):
        s = hanoi_state_for_step(tool)
        s_t = torch.tensor([list(s)], dtype=torch.long, device=reg.device)
        s_t[0, 2:].clamp_(0, cfg["max_disk"])
        with torch.no_grad():
            logits = model(s_t)
        action_idx = int(logits[0].argmax().item())
        src, dst = HANOI_ACTIONS[action_idx]
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
    return moves


def run_gcd(a: int, b: int, reg: SpecialistRegistry, max_steps=2_000_000):
    model, _ = reg.gcd()
    cur_a, cur_b = a, b
    for _ in range(max_steps):
        s_t = torch.tensor([list(state_for_pair(cur_a, cur_b))],
                           dtype=torch.long, device=reg.device)
        with torch.no_grad():
            logits = model(s_t)
        act = int(logits[0].argmax().item())
        if act == GCD_A2I["done"]:
            return cur_a
        if act == GCD_A2I["sub_b_from_a"]:
            cur_a -= cur_b
        else:
            cur_b -= cur_a
    return -1


def run_conway(grid, reg: SpecialistRegistry):
    model, _ = reg.conway()
    h, w = len(grid), len(grid[0])
    batch = []
    for r in range(h):
        for c in range(w):
            n = sum(
                grid[(r + dr) % h][(c + dc) % w]
                for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                if not (dr == 0 and dc == 0)
            )
            batch.append((grid[r][c], n))
    bt = torch.tensor(batch, dtype=torch.long, device=reg.device)
    with torch.no_grad():
        preds = model(bt).argmax(-1).cpu().tolist()
    return [preds[r * w:(r + 1) * w] for r in range(h)]


def run_bubble(arr, reg: SpecialistRegistry):
    model, _ = reg.bubble()
    sorted_arr, swaps = bubble_sort_with_step(arr, model, device=reg.device)
    return sorted_arr, swaps


def run_maze(start, goal, reg: SpecialistRegistry):
    model, _ = reg.maze()
    final, hist, n = maze_navigate(start, goal, model, device=reg.device,
                                    max_steps=200000)
    return final, hist, n


# ── Task dispatcher ─────────────────────────────────────────────

HANOI_RE     = re.compile(r"^HANOI\s+(\d+)\s*$")
GCD_RE       = re.compile(r"^GCD\s+(\d+)\s+(\d+)\s*$")
GCDHANOI_RE  = re.compile(r"^GCDHANOI\s+(\d+)\s+(\d+)\s*$")
CONWAY_RE    = re.compile(r"^CONWAY\s+(.+?)\s*$")
CONWAYS_RE   = re.compile(r"^CONWAYSTABLE\s+(.+?)\s*$")
BUBBLE_RE    = re.compile(r"^BUBBLE\s+([\d,\s\-]+)\s*$")
MAZE_RE      = re.compile(r"^MAZE\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s*$")
MAZESTEPS_RE = re.compile(r"^MAZESTEPS\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s*$")
SORTHANOI_RE = re.compile(r"^SORTHANOI\s+(\d+)\s*$")
GCDSORTED_RE = re.compile(r"^GCDSORTED\s+([\d,\s\-]+)\s*$")


def parse_grid(text: str):
    """Grid encoded as e.g. '0,1,0;1,0,1;0,1,0' (rows separated by ';')."""
    rows = text.strip().split(";")
    return [[int(x) for x in row.split(",")] for row in rows]


def fmt_grid(g):
    return ";".join(",".join(str(c) for c in r) for r in g)


def solve(task: str, reg: SpecialistRegistry):
    task = task.strip()

    m = HANOI_RE.match(task)
    if m:
        n = int(m.group(1))
        moves = run_hanoi(n, reg)
        return {"task": task, "n_moves": len(moves),
                "expected": (2 ** n) - 1,
                "first": moves[:3], "last": moves[-3:]}

    m = GCD_RE.match(task)
    if m:
        from math import gcd as math_gcd
        a, b = int(m.group(1)), int(m.group(2))
        return {"task": task, "gcd": run_gcd(a, b, reg), "expected": math_gcd(a, b)}

    m = GCDHANOI_RE.match(task)
    if m:
        from math import gcd as math_gcd
        n, mn = int(m.group(1)), int(m.group(2))
        cn = len(run_hanoi(n, reg))
        cm = len(run_hanoi(mn, reg))
        result = run_gcd(cn, cm, reg)
        return {"task": task, "result": result,
                "expected": math_gcd((2**n)-1, (2**mn)-1),
                "intermediate": {"hanoi_n": cn, "hanoi_m": cm}}

    m = CONWAY_RE.match(task)
    if m and not task.startswith("CONWAYSTABLE"):
        grid = parse_grid(m.group(1))
        out = run_conway(grid, reg)
        return {"task": task, "result": fmt_grid(out)}

    m = CONWAYS_RE.match(task)
    if m:
        grid = parse_grid(m.group(1))
        prev = None
        steps = 0
        for steps in range(1, 1000):
            new = run_conway(grid, reg)
            if new == grid:
                return {"task": task, "stable_after_steps": steps - 1,
                        "result": fmt_grid(new)}
            grid = new
        return {"task": task, "stable_after_steps": -1, "result": fmt_grid(grid)}

    m = BUBBLE_RE.match(task)
    if m:
        arr = [int(x) for x in m.group(1).split(",") if x.strip()]
        sorted_arr, swaps = run_bubble(arr, reg)
        return {"task": task, "sorted": sorted_arr, "swaps": swaps}

    m = MAZE_RE.match(task)
    if m:
        x0, y0, x1, y1 = (int(g) for g in m.groups())
        final, hist, n = run_maze((x0, y0), (x1, y1), reg)
        return {"task": task, "reached": final == (x1, y1),
                "steps": n, "first_5_moves": [MAZE_ACTIONS[a] for a in hist[:5]]}

    m = MAZESTEPS_RE.match(task)
    if m:
        x0, y0, x1, y1 = (int(g) for g in m.groups())
        final, hist, n = run_maze((x0, y0), (x1, y1), reg)
        return {"task": task, "steps_to_reach_goal": n}

    m = SORTHANOI_RE.match(task)
    if m:
        # Hanoi(n)'s moves, sorted by disk number (which is already
        # the first element). Show as a Lego composition.
        n = int(m.group(1))
        moves = run_hanoi(n, reg)
        # extract disk numbers, sort them
        disk_seq = [d for d, _, _ in moves]
        sorted_disks, _ = run_bubble(disk_seq, reg)
        return {"task": task, "moves_used": len(moves),
                "sorted_first_10": sorted_disks[:10]}

    m = GCDSORTED_RE.match(task)
    if m:
        from math import gcd as math_gcd
        arr = [int(x) for x in m.group(1).split(",") if x.strip()]
        sorted_arr, _ = run_bubble(arr, reg)
        if len(sorted_arr) < 2:
            return {"task": task, "error": "need at least 2 numbers"}
        small = sorted_arr[0] if sorted_arr[0] > 0 else sorted_arr[1]
        big = sorted_arr[-1]
        result = run_gcd(big, small, reg) if small > 0 else big
        return {"task": task, "sorted_min": small, "sorted_max": big,
                "gcd_of_min_max": result, "expected": math_gcd(small, big)}

    return {"task": task, "error": "unknown task"}


if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    reg = SpecialistRegistry(device=device)
    test_tasks = [
        # Single-specialist tasks
        "HANOI 5",
        "GCD 100 75",
        "CONWAY 0,0,0,0,0;0,0,0,0,0;0,1,1,1,0;0,0,0,0,0;0,0,0,0,0",  # blinker
        "BUBBLE 5,3,8,1,9,2,7,4,6",
        "MAZE 0 0 5 -3",
        # Composites — use multiple frozen specialists, ZERO retraining
        "GCDHANOI 6 9",
        "CONWAYSTABLE 0,0,0,0;0,1,1,0;0,1,1,0;0,0,0,0",  # block: stable in 0 steps
        "MAZESTEPS 0 0 100 -50",
        "SORTHANOI 4",
        "GCDSORTED 12,18,8,30,15",
    ]
    for t in test_tasks:
        t0 = time.time()
        result = solve(t, reg)
        dt = time.time() - t0
        # Pretty-print
        cmd = t[:40] + ("..." if len(t) > 40 else "")
        print(f"  [{dt*1000:>6.1f}ms] {cmd}")
        for k, v in result.items():
            if k == "task":
                continue
            sval = str(v)
            if len(sval) > 80:
                sval = sval[:77] + "..."
            print(f"      {k}: {sval}")
        print()
