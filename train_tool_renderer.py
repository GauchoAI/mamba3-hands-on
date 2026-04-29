"""train_tool_renderer — Mamba-3 byte-level conditional LM that renders
structured tool results as natural-language sentences.

This is the third Mamba-3-class model in the harness chain:
  router (which tool?) → specialist (do the math) → renderer (say it)

Training format is prefix-LM:
  <payload>\\x01<sentence>\\x02

  - byte 0 (\\x00) = PAD
  - byte 1 (\\x01) = BOA (begin-of-answer); separates structured payload
                     from the natural-language sentence to generate.
  - byte 2 (\\x02) = EOS (end-of-sentence); generation stops here.

Payload format: tool|key=val|key=val|...
  e.g. hanoi_solver|n=12|optimal=4095|params=45318

Target: any of several diverse phrasings per tool. The model learns to
pick a phrasing conditioned on the payload, but for inference we'll
greedily decode (deterministic, single best sentence).

Trained on synthetic data, ~5000 examples, fast on CPU.
"""
from __future__ import annotations
import argparse, math, random, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba3_lm import Mamba3LM, LMConfig


# ---- Lion optimizer (Chen et al. 2023) — sign-of-momentum updates ----
# Faster convergence than AdamW on many language tasks at lower compute.
# Inline implementation, no extra dep. ~15 lines.
class Lion(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]; b1, b2 = group["betas"]; wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None: continue
                g = p.grad
                state = self.state[p]
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(p)
                m = state["exp_avg"]
                # update direction = sign(b1 * m + (1 - b1) * g)
                update = (m.mul(b1).add(g, alpha=1 - b1)).sign_()
                # decoupled weight decay
                if wd != 0: p.mul_(1 - lr * wd)
                p.add_(update, alpha=-lr)
                # momentum: m = b2 * m + (1 - b2) * g
                m.mul_(b2).add_(g, alpha=1 - b2)
        return loss


PAD = 0
BOA = 1
EOS = 2

MAX_LEN = 256


# ---------------------------------------------------- synthetic targets ---


def _pick_lang(rng: random.Random) -> str:
    return rng.choice(["en", "es"])


def hanoi_payload_and_sentences(rng: random.Random) -> tuple[str, list[str]]:
    n = rng.randint(2, 23)
    optimal = (1 << n) - 1
    params = 45318
    timing = rng.randint(50, 5000)
    lang = _pick_lang(rng)
    payload = f"hanoi_solver|n={n}|optimal={optimal}|params={params}|timing={timing}|lang={lang}"
    # Templates per language. The trailing lang field in the payload tells
    # the renderer which output language to emit. Same content, two forms.
    if lang == "en":
        sentences = ["The optimal solution to Tower of Hanoi with $N disks requires $OPTIMAL moves."]
    else:
        sentences = ["La solución óptima de la Torre de Hanoi con $N discos requiere $OPTIMAL movimientos."]
    return payload, sentences


def gcd_payload_and_sentences(rng: random.Random) -> tuple[str, list[str]]:
    a = rng.randint(2, 9999)
    b = rng.randint(2, 9999)
    g = math.gcd(a, b)
    lang = _pick_lang(rng)
    payload = f"gcd|a={a}|b={b}|gcd={g}|lang={lang}"
    if lang == "en":
        sentences = ["The greatest common divisor of $A and $B is $GCD."]
    else:
        sentences = ["El máximo común divisor de $A y $B es $GCD."]
    return payload, sentences


def gcdhanoi_payload_and_sentences(rng: random.Random) -> tuple[str, list[str]]:
    a = rng.randint(2, 18)
    b = rng.randint(2, 18)
    ma = (1 << a) - 1
    mb = (1 << b) - 1
    g = math.gcd(ma, mb)
    lang = _pick_lang(rng)
    payload = f"gcdhanoi|a={a}|b={b}|moves_a={ma}|moves_b={mb}|gcd={g}|lang={lang}"
    if lang == "en":
        sentences = ["Hanoi($A) needs $MOVES_A moves; Hanoi($B) needs $MOVES_B moves; their gcd is $GCD."]
    else:
        sentences = ["Hanoi($A) requiere $MOVES_A movimientos; Hanoi($B) requiere $MOVES_B movimientos; su mcd es $GCD."]
    return payload, sentences


def fibonacci_payload_and_sentences(rng: random.Random) -> tuple[str, list[str]]:
    n = rng.randint(0, 30)
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    lang = _pick_lang(rng)
    payload = f"fibonacci|n={n}|fibonacci={a}|lang={lang}"
    if lang == "en":
        sentences = ["The $N-th Fibonacci number is $RESULT."]
    else:
        sentences = ["El $N-ésimo número de Fibonacci es $RESULT."]
    return payload, sentences


def factorial_payload_and_sentences(rng: random.Random) -> tuple[str, list[str]]:
    n = rng.randint(0, 12)
    r = 1
    for i in range(2, n + 1):
        r *= i
    lang = _pick_lang(rng)
    payload = f"factorial|n={n}|factorial={r}|lang={lang}"
    if lang == "en":
        sentences = ["The factorial of $N is $RESULT."]
    else:
        sentences = ["El factorial de $N es $RESULT."]
    return payload, sentences


GENERATORS = [
    hanoi_payload_and_sentences, gcd_payload_and_sentences, gcdhanoi_payload_and_sentences,
    fibonacci_payload_and_sentences, factorial_payload_and_sentences,
]


def gen_example(rng: random.Random) -> tuple[bytes, bytes]:
    """Pick a tool, generate a payload and a random matching sentence."""
    payload, sentences = rng.choice(GENERATORS)(rng)
    sentence = rng.choice(sentences)
    return payload.encode("utf-8"), sentence.encode("utf-8")


def encode_pair(payload_b: bytes, sentence_b: bytes, max_len: int = MAX_LEN) -> torch.Tensor:
    """Concatenate payload + BOA + sentence + EOS, pad to max_len."""
    tokens = [b for b in payload_b] + [BOA] + [b for b in sentence_b] + [EOS]
    tokens = tokens[:max_len]
    while len(tokens) < max_len:
        tokens.append(PAD)
    return torch.tensor(tokens, dtype=torch.long)


def build_target_mask(tokens: torch.Tensor) -> torch.Tensor:
    """1 on positions that are part of the answer (after BOA, including EOS),
    0 on payload + BOA + PAD. We only train next-byte loss inside the answer."""
    boa_pos = (tokens == BOA).nonzero(as_tuple=True)[0]
    if len(boa_pos) == 0:
        return torch.zeros_like(tokens, dtype=torch.bool)
    boa_idx = int(boa_pos[0])
    mask = torch.zeros_like(tokens, dtype=torch.bool)
    mask[boa_idx + 1:] = True
    mask[tokens == PAD] = False
    return mask


# -------------------------------------------------------------- training ---


def train(steps: int, batch: int, lr: float, device: str, seed: int = 42,
          val_size: int = 256, save_to: str = "checkpoints/tool_renderer_mamba3.pt",
          optimizer: str = "lion"):
    rng = random.Random(seed)
    torch.manual_seed(seed)

    cfg = LMConfig(
        n_layers=2, d_model=64, d_state=16, expand=2, headdim=16,
        vocab_size=256, max_seq_len=MAX_LEN,
        batch_size=batch, lr=lr, total_steps=steps,
    )
    model = Mamba3LM(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Mamba3LM renderer params: {n_params}")

    if optimizer == "lion":
        # Lion's effective LR is ~3-10× lower than AdamW for the same training
        # dynamics; the paper recommends scaling lr down proportionally.
        opt = Lion(model.parameters(), lr=lr / 3.0, weight_decay=0.1)
        print(f"Optimizer: Lion (lr={lr/3.0:.4f})")
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
        print(f"Optimizer: AdamW (lr={lr:.4f})")
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.05)

    val_pairs = [gen_example(rng) for _ in range(val_size)]
    val_tokens = torch.stack([encode_pair(p, s) for p, s in val_pairs]).to(device)
    val_masks = torch.stack([build_target_mask(t) for t in val_tokens]).to(device)

    t0 = time.time()
    best_loss = float("inf")
    best_state = None

    for step in range(steps):
        examples = [gen_example(rng) for _ in range(batch)]
        tokens = torch.stack([encode_pair(p, s) for p, s in examples]).to(device)
        masks = torch.stack([build_target_mask(t) for t in tokens]).to(device)

        # Standard prefix-LM training: predict next byte at every answer position
        x = tokens[:, :-1]
        y = tokens[:, 1:]
        m = masks[:, 1:]
        logits = model(x)  # (B, L-1, V)
        # Only score the answer positions
        loss = F.cross_entropy(
            logits[m], y[m]
        )
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        if (step + 1) % 100 == 0:
            with torch.no_grad():
                vx, vy = val_tokens[:, :-1], val_tokens[:, 1:]
                vm = val_masks[:, 1:]
                vlogits = model(vx)
                vloss = F.cross_entropy(vlogits[vm], vy[vm]).item()
            if vloss < best_loss:
                best_loss = vloss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                # Periodic save: never lose work to a kill mid-training.
                Path(save_to).parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "state_dict": best_state,
                    "config": cfg.__dict__,
                    "max_len": MAX_LEN,
                    "best_val_loss": best_loss,
                    "tokens": {"PAD": PAD, "BOA": BOA, "EOS": EOS},
                }, save_to)
            print(f"  step {step+1}/{steps}  train_loss={loss.item():.4f}  "
                  f"val_loss={vloss:.4f}  best_val={best_loss:.4f}  "
                  f"elapsed={time.time()-t0:.0f}s")

    if best_state is not None:
        model.load_state_dict(best_state)
    Path(save_to).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "config": cfg.__dict__,
        "max_len": MAX_LEN,
        "best_val_loss": best_loss,
        "tokens": {"PAD": PAD, "BOA": BOA, "EOS": EOS},
    }, save_to)
    print(f"\nSaved → {save_to}  (best val loss = {best_loss:.4f})")

    # Quick sample
    print("\nSamples:")
    rng2 = random.Random(seed + 1000)
    for _ in range(3):
        payload, _ = rng2.choice(GENERATORS)(rng2)
        prefix = list(payload.encode("utf-8")) + [BOA]
        gen = model.generate(prefix, max_new=200, temperature=0.1, top_k=1)
        # Stop at EOS
        if EOS in gen:
            gen = gen[:gen.index(EOS)]
        text = bytes([b for b in gen if 32 <= b < 256]).decode("utf-8", errors="ignore")
        print(f"  payload: {payload}")
        print(f"  → {text}\n")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--save-to", default="checkpoints/tool_renderer_mamba3.pt")
    ap.add_argument("--optimizer", default="lion", choices=["lion", "adamw"])
    args = ap.parse_args()
    print(f"Device: {args.device}")
    train(args.steps, args.batch, args.lr, args.device, save_to=args.save_to,
          optimizer=args.optimizer)


if __name__ == "__main__":
    main()
