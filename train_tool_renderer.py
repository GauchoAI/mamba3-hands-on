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


PAD = 0
BOA = 1
EOS = 2

MAX_LEN = 256


# ---------------------------------------------------- synthetic targets ---


def hanoi_payload_and_sentences(rng: random.Random) -> tuple[str, list[str]]:
    n = rng.randint(2, 23)
    optimal = (1 << n) - 1
    params = 45318
    timing = rng.randint(50, 5000)
    payload = f"hanoi_solver|n={n}|optimal={optimal}|params={params}|timing={timing}"
    # One canonical phrasing per tool. The Mamba-3 LM is small (74k params)
    # and synthetic data is easy; ambiguity over phrasings was the dominant
    # source of irreducible loss. Single template → near-zero loss → clean
    # numeric copy at greedy decoding.
    sentences = [
        f"The optimal solution to Tower of Hanoi with {n} disks requires {optimal:,} moves.",
    ]
    return payload, sentences


def gcd_payload_and_sentences(rng: random.Random) -> tuple[str, list[str]]:
    a = rng.randint(2, 9999)
    b = rng.randint(2, 9999)
    g = math.gcd(a, b)
    payload = f"gcd|a={a}|b={b}|gcd={g}"
    sentences = [
        f"The greatest common divisor of {a} and {b} is {g}.",
    ]
    return payload, sentences


def gcdhanoi_payload_and_sentences(rng: random.Random) -> tuple[str, list[str]]:
    a = rng.randint(2, 18)
    b = rng.randint(2, 18)
    ma = (1 << a) - 1
    mb = (1 << b) - 1
    g = math.gcd(ma, mb)
    payload = f"gcdhanoi|a={a}|b={b}|moves_a={ma}|moves_b={mb}|gcd={g}"
    sentences = [
        f"Hanoi({a}) needs {ma:,} moves; Hanoi({b}) needs {mb:,} moves; their gcd is {g}.",
    ]
    return payload, sentences


GENERATORS = [hanoi_payload_and_sentences, gcd_payload_and_sentences, gcdhanoi_payload_and_sentences]


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
          val_size: int = 256, save_to: str = "checkpoints/tool_renderer_mamba3.pt"):
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

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
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
    args = ap.parse_args()
    print(f"Device: {args.device}")
    train(args.steps, args.batch, args.lr, args.device, save_to=args.save_to)


if __name__ == "__main__":
    main()
