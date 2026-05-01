"""train_tool_router — tiny Mamba-3 classifier over bytes that picks a tool.

Replaces the regex scoring in assistant.py with a learned head. Input is
a natural-language prompt as bytes (vocab=256), output is a 3-way
softmax over {hanoi_solver, gcd, gcdhanoi}.

Argument extraction stays as regex in assistant.py — picking the tool
is the part that benefits from learning (semantic phrasings); pulling
integers out of the text doesn't.

Architecture: byte embedding → 1 Mamba3Block (d=64) → mean-pool → linear.

Trained on synthetic data: many phrasings per tool, mixed casing,
punctuation, decoy words. Should reach >99 % on a held-out split with
~5k synthesized examples and ~3000 steps on CPU.
"""
from __future__ import annotations
import _path_shim  # noqa: F401  (adds repo root to sys.path)
import argparse, random, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

from lab_platform.mamba3_minimal import Mamba3Block, Mamba3Config


TOOLS = ["hanoi_solver", "gcd", "gcdhanoi", "fibonacci", "factorial"]
TOOL_TO_IDX = {t: i for i, t in enumerate(TOOLS)}
N_TOOLS = len(TOOLS)
VOCAB = 256
PAD = 0


# ----------------------------------------------------- synthetic prompts ---


HANOI_TEMPLATES = [
    "solve tower of hanoi with {n} disks",
    "solve hanoi with {n} disks",
    "what is the optimal solution to hanoi {n}?",
    "how many moves does hanoi({n}) take?",
    "compute the minimum number of moves for tower of hanoi with {n} disks",
    "i need the move count for {n}-disk hanoi",
    "what's the answer to tower of hanoi with {n} disks?",
    "hanoi {n}",
    "tower of hanoi {n}",
    "{n}-disk tower of hanoi",
    "give me hanoi for n = {n}",
    "find optimal hanoi at n={n}",
    "Tower of Hanoi with {n} disks please",
    "Solve Hanoi for {n} disks.",
    "How long is the optimal Hanoi trace at n = {n}?",
    "minimum moves hanoi {n}",
    "optimal hanoi {n} disks",
    "the tower problem with {n} disks",
    "resuelve la torre de hanoi con {n} discos",
    "torre de hanoi {n}",
]

GCD_TEMPLATES = [
    "what is the gcd of {a} and {b}?",
    "compute gcd({a}, {b})",
    "greatest common divisor of {a} and {b}",
    "gcd({a}, {b})",
    "gcd of {a} and {b}",
    "find the greatest common divisor of {a} and {b}",
    "what's gcd({a}, {b})?",
    "gcd {a} {b}",
    "the greatest common divisor of {a}, {b}",
    "give me gcd of {a} and {b}",
    "GCD({a}, {b}) please",
    "Compute the GCD of {a} and {b}.",
    "máximo común divisor de {a} y {b}",
    "mcd de {a} y {b}",
    "common divisor of {a} {b}",
    "gcd for the integers {a} and {b}",
]

GCDHANOI_TEMPLATES = [
    "what is the gcd of hanoi {a} and hanoi {b}?",
    "compute the gcd of hanoi({a}) and hanoi({b})",
    "gcd of hanoi {a} and hanoi {b}",
    "greatest common divisor of hanoi {a} and hanoi {b}",
    "find gcd(hanoi({a}), hanoi({b}))",
    "hanoi {a} gcd hanoi {b}",
    "gcd of the move counts of hanoi {a} and hanoi {b}",
    "what's the gcd of hanoi at {a} disks and hanoi at {b} disks?",
    "common divisor of hanoi {a} and hanoi {b}",
    "compose hanoi {a} hanoi {b} via gcd",
    "GCD of Hanoi({a}) and Hanoi({b}) please",
    "the gcd of Hanoi {a} and Hanoi {b}",
    "mcd de hanoi {a} y hanoi {b}",
]

FIBONACCI_TEMPLATES = [
    "what is fibonacci {n}?",
    "fibonacci {n}",
    "compute F({n})",
    "what is F({n})?",
    "the {n}th fibonacci number",
    "fib({n})",
    "fib {n}",
    "give me F({n})",
    "{n}th fibonacci",
    "Fibonacci of {n}",
    "el {n}-ésimo número de fibonacci",
    "número de fibonacci {n}",
    "fibonacci de {n}",
]

FACTORIAL_TEMPLATES = [
    "what is {n}!?",
    "{n}!",
    "compute {n}!",
    "factorial of {n}",
    "the factorial of {n}",
    "{n} factorial",
    "{n}-factorial",
    "compute the factorial of {n}",
    "what is the factorial of {n}?",
    "factorial({n})",
    "factorial de {n}",
    "el factorial de {n}",
]

# Decoys — phrases that contain the keywords but with different intent.
# Adding these lets the model learn it's not just keyword matching.
DECOY_TEMPLATES = [
    ("hanoi_solver", "tower of hanoi {n}"),
    ("hanoi_solver", "the {n}-disk tower puzzle"),
    ("gcd", "compute the gcd of {a} and {b} integers"),
]


def random_n() -> int:
    # Mix small and larger n's — model should not learn a "small number" heuristic.
    if random.random() < 0.3:
        return random.randint(2, 9)
    return random.randint(10, 23)


def random_int() -> int:
    return random.randint(2, 9999)


def random_case(s: str) -> str:
    r = random.random()
    if r < 0.6:
        return s
    if r < 0.85:
        return s.capitalize()
    return s.upper()


def maybe_punct(s: str) -> str:
    if random.random() < 0.5:
        return s
    return s + random.choice([".", "!", "?", "...", " thanks", " please"])


def gen_example(rng: random.Random) -> tuple[str, int]:
    tool = rng.choice(TOOLS)
    if tool == "hanoi_solver":
        tpl = rng.choice(HANOI_TEMPLATES)
        text = tpl.format(n=random_n())
    elif tool == "gcd":
        tpl = rng.choice(GCD_TEMPLATES)
        text = tpl.format(a=random_int(), b=random_int())
    elif tool == "gcdhanoi":
        tpl = rng.choice(GCDHANOI_TEMPLATES)
        text = tpl.format(a=random_n(), b=random_n())
    elif tool == "fibonacci":
        tpl = rng.choice(FIBONACCI_TEMPLATES)
        text = tpl.format(n=rng.randint(0, 30))
    else:  # factorial
        tpl = rng.choice(FACTORIAL_TEMPLATES)
        text = tpl.format(n=rng.randint(0, 12))
    text = random_case(text)
    text = maybe_punct(text)
    return text, TOOL_TO_IDX[tool]


def encode(text: str, max_len: int = 96) -> torch.Tensor:
    b = text.encode("utf-8", errors="ignore")[:max_len]
    out = [0] * max_len
    for i, c in enumerate(b):
        out[i] = c
    return torch.tensor(out, dtype=torch.long)


# -------------------------------------------------------------- model ---


class ToolRouter(nn.Module):
    """Mamba-3 byte-level classifier over a 96-byte prompt."""

    def __init__(self, d_model: int = 64, n_blocks: int = 1):
        super().__init__()
        cfg = Mamba3Config(d_model=d_model)
        self.emb = nn.Embedding(VOCAB, d_model, padding_idx=PAD)
        self.blocks = nn.ModuleList([Mamba3Block(cfg) for _ in range(n_blocks)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, N_TOOLS)

    def forward(self, byte_ids: torch.Tensor) -> torch.Tensor:
        # byte_ids: (B, L)
        x = self.emb(byte_ids)
        for block in self.blocks:
            x = x + block(x)
        x = self.norm(x)
        # mean-pool over non-pad positions
        mask = (byte_ids != PAD).float().unsqueeze(-1)  # (B, L, 1)
        x = (x * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return self.head(x)


# ------------------------------------------------------------ training ---


def train(steps: int, batch: int, lr: float, device: str, seed: int = 42,
          val_size: int = 1024, save_to: str = "checkpoints/tool_router_mamba3.pt"):
    rng = random.Random(seed)
    torch.manual_seed(seed)

    model = ToolRouter().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.05)

    val_texts = [gen_example(rng) for _ in range(val_size)]
    val_x = torch.stack([encode(t) for t, _ in val_texts]).to(device)
    val_y = torch.tensor([y for _, y in val_texts], device=device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"ToolRouter params: {n_params}")
    t0 = time.time()
    best_acc = 0.0
    best_state = None

    for step in range(steps):
        examples = [gen_example(rng) for _ in range(batch)]
        x = torch.stack([encode(t) for t, _ in examples]).to(device)
        y = torch.tensor([y for _, y in examples], device=device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        if (step + 1) % 200 == 0:
            with torch.no_grad():
                vlogits = model(val_x)
                vacc = (vlogits.argmax(-1) == val_y).float().mean().item()
            if vacc > best_acc:
                best_acc = vacc
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            print(f"  step {step+1}/{steps}  loss={loss.item():.4f}  val_acc={vacc:.4%}  "
                  f"best={best_acc:.4%}  elapsed={time.time()-t0:.0f}s")

    if best_state is not None:
        model.load_state_dict(best_state)
    Path(save_to).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "tools": TOOLS,
        "max_len": 96,
        "config": {"d_model": 64, "n_blocks": 1},
        "best_val_acc": best_acc,
    }, save_to)
    print(f"\nSaved → {save_to}  (best val acc = {best_acc:.4%})")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--save-to", default="checkpoints/tool_router_mamba3.pt")
    args = ap.parse_args()
    print(f"Device: {args.device}")
    train(args.steps, args.batch, args.lr, args.device, save_to=args.save_to)


if __name__ == "__main__":
    main()
