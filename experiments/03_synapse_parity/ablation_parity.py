"""Ablation: which of the two innovations (RoPE, trapezoidal) is responsible
for the parity win?

Four variants, same hyperparameters:
  - full:      RoPE + trapezoidal  (Mamba-3)
  - no-trap:   RoPE only
  - no-rope:   trapezoidal only
  - neither:   == Mamba-2-like in spirit
"""
import torch, torch.nn as nn, torch.nn.functional as F
from lab_platform.mamba3_minimal import Mamba3Block, Mamba3Config
from parity_experiment import make_parity_batch, ParityModel


def train_eval(use_rope, use_trap, device, steps=400, L=16, lr=3e-3):
    torch.manual_seed(0)
    cfg = Mamba3Config(d_model=32, d_state=16, expand=2, headdim=16)
    block_factory = lambda c: Mamba3Block(c, use_rope=use_rope, use_trap=use_trap)
    m = ParityModel(block_factory, cfg).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    for _ in range(steps):
        bits, tgt = make_parity_batch(64, L, device)
        loss = F.cross_entropy(m(bits).reshape(-1, 2), tgt.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    m.eval()
    with torch.no_grad():
        bits, tgt = make_parity_batch(2048, L, device)
        preds = m(bits).argmax(-1)
        return {
            "acc_all":  (preds == tgt).float().mean().item(),
            "acc_last": (preds[:, -1] == tgt[:, -1]).float().mean().item(),
            "loss":     loss.item(),
        }


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}\nParity L=16, 400 steps\n")

    variants = [
        ("full (RoPE+trap)",  True,  True),
        ("no trapezoidal",    True,  False),
        ("no RoPE",           False, True),
        ("neither",           False, False),
    ]
    print(f"{'variant':20s}  {'acc_all':>8s}  {'acc_last':>9s}  {'loss':>7s}")
    print("-" * 48)
    for name, rope, trap in variants:
        r = train_eval(rope, trap, device)
        print(f"{name:20s}  {r['acc_all']*100:7.1f}%  {r['acc_last']*100:8.1f}%  {r['loss']:7.3f}")

    print("\nInterpretation cheatsheet:")
    print("  full    == Mamba-3")
    print("  no-rope == parity-blind, state-tracking-blind (Mamba-2 behaviour)")
    print("  no-trap == just the complex-dynamics fix; is it enough?")


if __name__ == "__main__":
    main()
