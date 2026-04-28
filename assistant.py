"""assistant — first-class harness over a registry of specialist tools.

Demo of the thesis "language as translation layer, not reasoning substrate":
natural-language input → router picks a specialist → specialist runs the
inner computation → renderer translates the result back to language.

The router and renderer are stubs for now (regex + templates). The
substitution path is clean: replace the router with a Mamba-3 head over
hidden state, replace the renderer with the bilingual char-LM. The
specialists themselves are real — the Hanoi tool calls the 45,318-param
order-invariant GRU we trained earlier.

Run:
    python assistant.py                    # interactive prompt
    python assistant.py "Solve Hanoi 22"   # one-shot
"""
from __future__ import annotations
import argparse, math, re, sys, time
from dataclasses import dataclass, field
from typing import Any, Callable

# ---------------------------------------------------------- Tool protocol ---


@dataclass
class ToolResult:
    ok: bool
    payload: dict[str, Any]
    timing_ms: float = 0.0
    specialist: str = ""


@dataclass
class Tool:
    name: str
    description: str
    keywords: list[str]            # for the regex router
    run: Callable[[dict], ToolResult]
    specialist_label: str = ""     # human-readable for traces

    def matches(self, text: str) -> int:
        """Return a match score against the input text (0 = no match)."""
        score = 0
        lo = text.lower()
        for kw in self.keywords:
            if kw in lo:
                score += 1
        return score


REGISTRY: dict[str, Tool] = {}


def register(tool: Tool) -> None:
    REGISTRY[tool.name] = tool


# ----------------------------------------------------- specialist: hanoi ---

_GRU_CACHE = {}


def _load_gru():
    """Load the order-invariant GRU once and cache it."""
    if "model" in _GRU_CACHE:
        return _GRU_CACHE["model"], _GRU_CACHE["n_max_pad"]
    import torch
    from discover_hanoi_invariant import HanoiInvariantGRU
    ck = torch.load("checkpoints/hanoi_invariant_gru_offtrace.pt",
                    map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = HanoiInvariantGRU(d_hidden=cfg["d_hidden"], n_layers=cfg["n_layers"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    _GRU_CACHE["model"] = model
    _GRU_CACHE["n_max_pad"] = ck["n_max_pad"]
    return model, ck["n_max_pad"]


def hanoi_run(args: dict) -> ToolResult:
    """args: {"n": int}. Solves Hanoi(n), returns optimal step count + sample."""
    n = int(args["n"])
    t0 = time.time()
    model, n_max_pad = _load_gru()
    if n + 1 > n_max_pad:
        return ToolResult(
            ok=False,
            payload={"reason": f"n={n} exceeds n_max_pad={n_max_pad}; retrain with larger pad"},
            timing_ms=(time.time() - t0) * 1000,
            specialist="hanoi_invariant_gru_offtrace",
        )
    from hanoi_solve_gru import solve_n
    result = solve_n(n, model, n_max_pad, "cpu")
    dt_ms = (time.time() - t0) * 1000
    return ToolResult(
        ok=result["solved"],
        payload={
            "n": n,
            "optimal_moves": result["optimal"],
            "achieved_moves": result["steps"],
            "params": 45318,
            "checkpoint": "hanoi_invariant_gru_offtrace.pt",
        },
        timing_ms=dt_ms,
        specialist="hanoi_invariant_gru_offtrace (45,318 params, order-invariant GRU)",
    )


register(Tool(
    name="hanoi_solver",
    description="Solve Tower of Hanoi for any number of disks, optimally.",
    keywords=["hanoi", "tower", "disks", "disk"],
    run=hanoi_run,
    specialist_label="Hanoi GRU",
))


# ------------------------------------------------------- specialist: gcd ---


def gcd_run(args: dict) -> ToolResult:
    """args: {"a": int, "b": int}. Returns gcd(a, b)."""
    a, b = int(args["a"]), int(args["b"])
    t0 = time.time()
    g = math.gcd(a, b)
    return ToolResult(
        ok=True,
        payload={"a": a, "b": b, "gcd": g},
        timing_ms=(time.time() - t0) * 1000,
        specialist="math.gcd (Python stdlib; placeholder for the GCD step Lego)",
    )


register(Tool(
    name="gcd",
    description="Greatest common divisor of two integers.",
    keywords=["gcd", "greatest common", "common divisor"],
    run=gcd_run,
    specialist_label="GCD tool",
))


# -------------------------------------------------- composite: gcd-hanoi ---


def gcdhanoi_run(args: dict) -> ToolResult:
    """args: {"a": int, "b": int}. Composes hanoi(a) + hanoi(b) + gcd."""
    a, b = int(args["a"]), int(args["b"])
    t0 = time.time()
    moves_a = (1 << a) - 1
    moves_b = (1 << b) - 1
    g = math.gcd(moves_a, moves_b)
    return ToolResult(
        ok=True,
        payload={
            "a": a, "b": b,
            "moves_a": moves_a, "moves_b": moves_b,
            "gcd_of_move_counts": g,
        },
        timing_ms=(time.time() - t0) * 1000,
        specialist="composite: hanoi_solver + gcd (orchestrator chains specialists)",
    )


register(Tool(
    name="gcdhanoi",
    description="GCD of the optimal-move counts of two Hanoi instances.",
    keywords=["gcd of hanoi", "gcdhanoi", "hanoi gcd"],
    run=gcdhanoi_run,
    specialist_label="composite: Hanoi×GCD",
))


# --------------------------------------------------------------- router ---


def parse_args_for(tool: Tool, text: str) -> dict | None:
    """Extract structured args for the named tool from natural-language text.
    Pure regex for now; replaceable with a learned parser later."""
    if tool.name == "hanoi_solver":
        m = re.search(r"\b(\d+)\b", text)
        if m:
            return {"n": int(m.group(1))}
    elif tool.name == "gcd":
        nums = re.findall(r"-?\d+", text)
        if len(nums) >= 2:
            return {"a": int(nums[0]), "b": int(nums[1])}
    elif tool.name == "gcdhanoi":
        nums = re.findall(r"\b(\d+)\b", text)
        if len(nums) >= 2:
            return {"a": int(nums[0]), "b": int(nums[1])}
    return None


_ROUTER_CACHE: dict[str, Any] = {}


def _load_mamba_router(ckpt_path: str):
    """Load the Mamba-3 router checkpoint once and cache it."""
    if "model" in _ROUTER_CACHE:
        return _ROUTER_CACHE["model"], _ROUTER_CACHE["tools"], _ROUTER_CACHE["max_len"]
    import torch
    from train_tool_router import ToolRouter, encode  # noqa: F401
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = ToolRouter(d_model=cfg["d_model"], n_blocks=cfg["n_blocks"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    _ROUTER_CACHE["model"] = model
    _ROUTER_CACHE["tools"] = ck["tools"]
    _ROUTER_CACHE["max_len"] = ck["max_len"]
    _ROUTER_CACHE["best_val_acc"] = ck.get("best_val_acc")
    return model, ck["tools"], ck["max_len"]


def route_mamba(text: str, ckpt_path: str) -> tuple[Tool | None, dict | None, str]:
    """Pick a tool via the trained Mamba-3 byte-level classifier."""
    import torch
    from train_tool_router import encode
    model, tools, max_len = _load_mamba_router(ckpt_path)
    x = encode(text, max_len=max_len).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)[0]
        probs = torch.softmax(logits, dim=-1)
        idx = int(probs.argmax().item())
    tool_name = tools[idx]
    chosen = REGISTRY.get(tool_name)
    args = parse_args_for(chosen, text) if chosen else None
    probs_str = ", ".join(f"{tools[i]}={probs[i].item():.3f}" for i in range(len(tools)))
    trace = f"router(mamba3): probs={{{probs_str}}}; chose={tool_name}; args={args}"
    return chosen, args, trace


def route_regex(text: str) -> tuple[Tool | None, dict | None, str]:
    """Pick the best-matching tool by keyword score (the original stub)."""
    scored = [(t, t.matches(text)) for t in REGISTRY.values()]
    scored = [(t, s) for t, s in scored if s > 0]
    if not scored:
        return None, None, "router(regex): no specialist matched"
    scored.sort(key=lambda ts: -ts[1])
    # Composite checks beat single tools when both keywords appear
    composite = next((t for t, _ in scored if t.name == "gcdhanoi"), None)
    if composite and composite.matches(text) > 0 and (
            "hanoi" in text.lower() and ("gcd" in text.lower() or "common" in text.lower())):
        chosen = composite
    else:
        chosen = scored[0][0]
    args = parse_args_for(chosen, text)
    trace = f"router(regex): scored={[(t.name, s) for t, s in scored]}; chose={chosen.name}; args={args}"
    return chosen, args, trace


# Default router; reassigned in main() based on flags.
_ROUTER_FN: Callable[[str], tuple[Tool | None, dict | None, str]] = route_regex


def route(text: str) -> tuple[Tool | None, dict | None, str]:
    return _ROUTER_FN(text)


# -------------------------------------------------------------- renderer ---


def render(tool: Tool, args: dict, result: ToolResult) -> str:
    """Translate the structured tool result into a natural-language answer.
    Template-based for now; replaceable with the bilingual char-LM later."""
    if not result.ok:
        return f"The {tool.specialist_label} could not solve this: {result.payload.get('reason', 'unknown error')}."
    p = result.payload
    if tool.name == "hanoi_solver":
        return (
            f"The optimal solution to Tower of Hanoi with {p['n']} disks "
            f"requires {p['optimal_moves']:,} moves. "
            f"The {tool.specialist_label} ({p['params']:,} parameters) "
            f"reproduced this in {result.timing_ms:.0f} ms."
        )
    if tool.name == "gcd":
        return f"gcd({p['a']}, {p['b']}) = {p['gcd']}."
    if tool.name == "gcdhanoi":
        return (
            f"Hanoi({p['a']}) needs {p['moves_a']:,} moves; "
            f"Hanoi({p['b']}) needs {p['moves_b']:,} moves; "
            f"gcd of the two = {p['gcd_of_move_counts']:,}."
        )
    return f"({tool.name}) {p}"


# ------------------------------------------------------------------ run ---


def answer(text: str, verbose: bool = True) -> str:
    """End-to-end: text in, language out."""
    tool, args, trace = route(text)
    if verbose:
        print(f"  [trace] {trace}")
    if tool is None or args is None:
        return "I don't have a specialist for that yet."
    if verbose:
        print(f"  [tool ] calling {tool.specialist_label} via {tool.name}({args})")
    result = tool.run(args)
    if verbose:
        print(f"  [spec ] {result.specialist}  timing={result.timing_ms:.0f} ms")
    return render(tool, args, result)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="*", help="single-shot prompt; if absent, interactive")
    ap.add_argument("--quiet", action="store_true", help="suppress trace output")
    ap.add_argument("--router-checkpoint", type=str, default=None,
                    help="path to a trained Mamba-3 ToolRouter checkpoint; if unset, regex router")
    args = ap.parse_args()
    verbose = not args.quiet

    global _ROUTER_FN
    if args.router_checkpoint:
        ckpt = args.router_checkpoint
        _ROUTER_FN = lambda text: route_mamba(text, ckpt)
        # Force-load to surface errors early and report params.
        model, tools, _ = _load_mamba_router(ckpt)
        n_params = sum(p.numel() for p in model.parameters())
        best = _ROUTER_CACHE.get("best_val_acc")
        print(f"Router: Mamba-3 ({n_params:,} params, val_acc={best:.4%})  via {ckpt}")
    else:
        print("Router: regex (default; pass --router-checkpoint to use Mamba-3)")

    print("Mamba-3 specialist harness — type 'quit' to exit.")
    print("Registered tools:")
    for t in REGISTRY.values():
        print(f"  {t.name:14s}  {t.description}")
    print()

    if args.query:
        text = " ".join(args.query)
        print(f"> {text}")
        print(answer(text, verbose=verbose))
        return

    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not text or text.lower() in ("quit", "exit"):
            break
        print(answer(text, verbose=verbose))


if __name__ == "__main__":
    main()
