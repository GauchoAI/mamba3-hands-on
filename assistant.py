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


# ----------------------------------------------------- specialist: fibonacci ---


def fibonacci_run(args: dict) -> ToolResult:
    """args: {"n": int}. Returns F(n) (0-indexed: F(0)=0, F(1)=1)."""
    n = int(args["n"])
    t0 = time.time()
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return ToolResult(
        ok=True,
        payload={"n": n, "fibonacci": a},
        timing_ms=(time.time() - t0) * 1000,
        specialist="fibonacci tool (Python iterative; placeholder for the FIB Mamba-3 LM)",
    )


register(Tool(
    name="fibonacci",
    description="Nth Fibonacci number (F(0)=0, F(1)=1).",
    keywords=["fibonacci", "fib(", "fib ", "ésimo de fibonacci", "número de fibonacci"],
    run=fibonacci_run,
    specialist_label="Fibonacci tool",
))


# ----------------------------------------------------- specialist: factorial ---


def factorial_run(args: dict) -> ToolResult:
    """args: {"n": int}. Returns n!."""
    n = int(args["n"])
    t0 = time.time()
    r = 1
    for i in range(2, n + 1):
        r *= i
    return ToolResult(
        ok=True,
        payload={"n": n, "factorial": r},
        timing_ms=(time.time() - t0) * 1000,
        specialist="factorial tool (Python iterative)",
    )


register(Tool(
    name="factorial",
    description="Factorial n! of a non-negative integer.",
    keywords=["factorial", "!", "factorial de"],
    run=factorial_run,
    specialist_label="Factorial tool",
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
    elif tool.name == "fibonacci":
        m = re.search(r"\b(\d+)\b", text)
        if m:
            return {"n": int(m.group(1))}
    elif tool.name == "factorial":
        m = re.search(r"\b(\d+)\b", text)
        if m:
            return {"n": int(m.group(1))}
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


def render_template(tool: Tool, args: dict, result: ToolResult) -> str:
    """Translate the structured tool result into a natural-language answer.
    Template-based stub; the Mamba-3 renderer below is the learned version."""
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
    if tool.name == "fibonacci":
        return f"F({p['n']}) = {p['fibonacci']:,}."
    if tool.name == "factorial":
        return f"{p['n']}! = {p['factorial']:,}."
    return f"({tool.name}) {p}"


# ---- Mamba-3 renderer: structured payload → natural-language sentence ----

_RENDERER_CACHE: dict[str, Any] = {}


def _load_renderer(ckpt_path: str):
    if "model" in _RENDERER_CACHE:
        return (_RENDERER_CACHE["model"], _RENDERER_CACHE["tokens"],
                _RENDERER_CACHE["kind"])
    import torch
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    # Detect which renderer architecture this checkpoint is for:
    # CopyMamba3LM has 'attn_dim'; plain Mamba3LM has 'd_state', 'expand', etc.
    if "attn_dim" in cfg:
        from train_tool_renderer_copy import CopyMamba3LM
        model = CopyMamba3LM(**cfg)
        kind = "copy"
    else:
        from mamba3_lm import Mamba3LM, LMConfig
        model = Mamba3LM(LMConfig(**cfg))
        kind = "lm"
    model.load_state_dict(ck["state_dict"])
    model.eval()
    _RENDERER_CACHE["model"] = model
    _RENDERER_CACHE["tokens"] = ck["tokens"]
    _RENDERER_CACHE["max_len"] = ck["max_len"]
    _RENDERER_CACHE["kind"] = kind
    _RENDERER_CACHE["best_val_loss"] = ck.get("best_val_loss")
    return model, ck["tokens"], kind


def _payload_string(tool: Tool, result: ToolResult) -> str | None:
    """Canonical pipe-delimited payload that matches the renderer's training format."""
    p = result.payload
    if tool.name == "hanoi_solver":
        return f"hanoi_solver|n={p['n']}|optimal={p['optimal_moves']}|params={p['params']}|timing={int(result.timing_ms)}"
    if tool.name == "gcd":
        return f"gcd|a={p['a']}|b={p['b']}|gcd={p['gcd']}"
    if tool.name == "gcdhanoi":
        return f"gcdhanoi|a={p['a']}|b={p['b']}|moves_a={p['moves_a']}|moves_b={p['moves_b']}|gcd={p['gcd_of_move_counts']}"
    if tool.name == "fibonacci":
        return f"fibonacci|n={p['n']}|fibonacci={p['fibonacci']}"
    if tool.name == "factorial":
        return f"factorial|n={p['n']}|factorial={p['factorial']}"
    return None


# Slot map: how each tool's placeholder names resolve to payload values, and
# how those values should be formatted when substituted. Keeping this declarative
# means the renderer LM only needs to learn the *language form* (the template
# skeleton); precise number copying is deterministic post-processing in the
# orchestrator. This is the data-to-text / pointer-mechanism pattern, applied
# at the orchestrator boundary.
SLOT_MAP: dict[str, dict[str, tuple[str, str]]] = {
    # tool_name -> { placeholder -> (payload_key, format_spec) }
    "hanoi_solver": {
        "$N":       ("n",             ""),
        "$OPTIMAL": ("optimal_moves", ","),
        "$PARAMS":  ("params",        ","),
    },
    "gcd": {
        "$A":   ("a",   ""),
        "$B":   ("b",   ""),
        "$GCD": ("gcd", ""),
    },
    "gcdhanoi": {
        "$A":       ("a",                  ""),
        "$B":       ("b",                  ""),
        "$MOVES_A": ("moves_a",            ","),
        "$MOVES_B": ("moves_b",            ","),
        "$GCD":     ("gcd_of_move_counts", ","),
    },
    "fibonacci": {
        "$N":      ("n",         ""),
        "$RESULT": ("fibonacci", ","),
    },
    "factorial": {
        "$N":      ("n",         ""),
        "$RESULT": ("factorial", ","),
    },
}


def _substitute(template: str, tool: Tool, result: ToolResult) -> tuple[str, list[str]]:
    """Replace every placeholder in `template` with values from `result.payload`.
    Returns (substituted_text, list_of_unfilled_placeholders).
    """
    mapping = SLOT_MAP.get(tool.name, {})
    text = template
    for slot, (key, fmt) in mapping.items():
        if slot not in text:
            continue
        val = result.payload[key]
        formatted = format(val, fmt) if fmt else str(val)
        text = text.replace(slot, formatted)
    leftover = [s for s in mapping if s in text]
    return text, leftover


def render_mamba(tool: Tool, args: dict, result: ToolResult, ckpt_path: str) -> str:
    """Generate a templated sentence with the Mamba-3 LM, then substitute the
    actual numeric values from the payload via deterministic post-processing.

    The LM is trained on targets like:
        "The optimal solution to Tower of Hanoi with $N disks requires $OPTIMAL moves."

    so the small SSM never has to copy specific digits from the prefix. It only
    learns the *language form* — which it does easily. The orchestrator does
    the literal substitution. The boundary is clean: LM owns shape, orchestrator
    owns values.

    Guard: if the LM output is missing required placeholders for this tool,
    fall back to the template renderer and log what was missing.
    """
    if not result.ok:
        return f"The {tool.specialist_label} could not solve this: {result.payload.get('reason', 'unknown error')}."
    payload_str = _payload_string(tool, result)
    if payload_str is None:
        return render_template(tool, args, result)
    model, tokens, kind = _load_renderer(ckpt_path)
    prefix = list(payload_str.encode("utf-8")) + [tokens["BOA"]]

    if kind == "copy":
        # CopyMamba3LM: model handles digit-copy internally via the pointer
        # mechanism. Output is the final natural-language sentence — no slot
        # substitution needed. Apply payload-fidelity guard against the raw
        # required numerics; fall back to template if the LM dropped any.
        gen, trace = model.generate(prefix, max_new=200, temperature=0.1, top_k=1, return_trace=True)
        eos = tokens["EOS"]
        if eos in gen:
            gen = gen[:gen.index(eos)]
        text = bytes([b for b in gen if 32 <= b < 256]).decode("utf-8", errors="ignore")
        # Guard: required numbers must appear verbatim
        required = []
        p = result.payload
        if tool.name == "hanoi_solver":
            required = [str(p["n"]), f"{p['optimal_moves']:,}"]
        elif tool.name == "gcd":
            required = [str(p["a"]), str(p["b"]), str(p["gcd"])]
        elif tool.name == "gcdhanoi":
            required = [str(p["a"]), str(p["b"]), f"{p['moves_a']:,}",
                        f"{p['moves_b']:,}", str(p["gcd_of_move_counts"])]
        missing = [n for n in required if n not in text]
        if missing:
            fallback = render_template(tool, args, result)
            return f"{fallback}  [renderer-guard: copy LM dropped {missing}; used template]"
        return text
    # Plain LM: template-with-placeholders path (kind == "lm")
    gen = model.generate(prefix, max_new=200, temperature=0.1, top_k=1)
    eos = tokens["EOS"]
    if eos in gen:
        gen = gen[:gen.index(eos)]
    template_text = bytes([b for b in gen if 32 <= b < 256]).decode("utf-8", errors="ignore")

    # Guard: the LM must have emitted *at least one* known placeholder for this
    # tool. If it emitted none, it's just generating freeform text without
    # structure → fall back to template.
    known_slots = list(SLOT_MAP.get(tool.name, {}).keys())
    if not any(s in template_text for s in known_slots):
        fallback = render_template(tool, args, result)
        return f"{fallback}  [renderer-guard: LM emitted no known $-slot; used template]"

    substituted, leftover = _substitute(template_text, tool, result)
    if leftover:
        fallback = render_template(tool, args, result)
        return f"{fallback}  [renderer-guard: substitution left {leftover}; used template]"
    return substituted


_RENDER_FN: Callable[[Tool, dict, ToolResult], str] = render_template


def render(tool: Tool, args: dict, result: ToolResult) -> str:
    return _RENDER_FN(tool, args, result)


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
    ap.add_argument("--renderer-checkpoint", type=str, default=None,
                    help="path to a trained Mamba-3 renderer checkpoint; if unset, template renderer")
    args = ap.parse_args()
    verbose = not args.quiet

    global _ROUTER_FN, _RENDER_FN
    if args.router_checkpoint:
        ckpt = args.router_checkpoint
        _ROUTER_FN = lambda text: route_mamba(text, ckpt)
        # Force-load to surface errors early and report params.
        model, tools, _ = _load_mamba_router(ckpt)
        n_params = sum(p.numel() for p in model.parameters())
        best = _ROUTER_CACHE.get("best_val_acc")
        print(f"Router:   Mamba-3 ({n_params:,} params, val_acc={best:.4%})  via {ckpt}")
    else:
        print("Router:   regex (default; pass --router-checkpoint to use Mamba-3)")

    if args.renderer_checkpoint:
        rckpt = args.renderer_checkpoint
        _RENDER_FN = lambda tool, args_, result: render_mamba(tool, args_, result, rckpt)
        rmodel, _, kind = _load_renderer(rckpt)
        n_params = sum(p.numel() for p in rmodel.parameters())
        if kind == "copy":
            print(f"Renderer: CopyMamba3LM ({n_params:,} params, byte-level AR with copy mechanism)  via {rckpt}")
        else:
            print(f"Renderer: Mamba-3 LM ({n_params:,} params, byte-level AR)  via {rckpt}")
    else:
        print("Renderer: templates (default; pass --renderer-checkpoint to use Mamba-3 LM)")

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
