"""
Parity task: compare Mamba-3 vs Mamba-2-like on the state-tracking task
that Mamba-2 famously fails.

Input:  sequence of bits in {0, 1}, length L
Target: running XOR (parity) at each position

Theory: a real-scalar SSM cannot solve parity (needs rotational eigenvalues).
Mamba-3's data-dependent RoPE should give it this capability.

Paper reports: Mamba-2 ~0.9% (random), Mamba-3 100% on parity.
We expect a similar qualitative gap at much smaller scale.
"""
from __future__ import annotations
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_platform.mamba3_minimal import Mamba3Block, Mamba2LikeBlock, Mamba3Config


def make_parity_batch(batch: int, L: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (bits, running_parity). bits: (B, L) in {0,1}; targets same shape."""
    bits = torch.randint(0, 2, (batch, L), device=device)
    parity = torch.cumsum(bits, dim=1) % 2
    return bits, parity


class ParityModel(nn.Module):
    """Tiny wrapper: embed bit -> SSM block -> classifier."""
    def __init__(self, block_cls, cfg: Mamba3Config):
        super().__init__()
        self.embed = nn.Embedding(2, cfg.d_model)
        self.block = block_cls(cfg)
        self.norm  = nn.LayerNorm(cfg.d_model)
        self.head  = nn.Linear(cfg.d_model, 2)

    def forward(self, bits: torch.Tensor) -> torch.Tensor:
        u = self.embed(bits)
        y = self.block(u) + u            # residual
        y = self.norm(y)
        return self.head(y)              # (B, L, 2) logits


def train_and_eval(
    name: str,
    block_cls,
    cfg: Mamba3Config,
    device,
    steps: int = 400,
    batch: int = 64,
    L: int = 16,
    lr: float = 3e-3,
) -> dict:
    torch.manual_seed(0)
    model = ParityModel(block_cls, cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    n_params = sum(p.numel() for p in model.parameters())

    model.train()
    t0 = time.time()
    losses = []
    for step in range(steps):
        bits, target = make_parity_batch(batch, L, device)
        logits = model(bits)
        loss = F.cross_entropy(logits.reshape(-1, 2), target.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    t_train = time.time() - t0

    # Eval on fresh data
    model.eval()
    with torch.no_grad():
        bits, target = make_parity_batch(2048, L, device)
        logits = model(bits)
        preds = logits.argmax(dim=-1)
        acc_all    = (preds == target).float().mean().item()
        acc_last   = (preds[:, -1] == target[:, -1]).float().mean().item()
        # position-wise accuracy
        pos_acc = (preds == target).float().mean(dim=0).tolist()

    return {
        "name": name,
        "n_params": n_params,
        "final_loss": losses[-1],
        "mean_last_10_loss": sum(losses[-10:]) / 10,
        "acc_all_positions": acc_all,
        "acc_last_position": acc_last,
        "pos_acc": pos_acc,
        "train_time_s": t_train,
    }


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    cfg = Mamba3Config(d_model=32, d_state=16, expand=2, headdim=16)
    L = 16

    results = []
    for name, block in [("Mamba-2-like", Mamba2LikeBlock),
                        ("Mamba-3",      Mamba3Block)]:
        print(f"\n--- Training {name} ---")
        r = train_and_eval(name, block, cfg, device, steps=400, batch=64, L=L, lr=3e-3)
        results.append(r)
        print(f"{name:15s}  params={r['n_params']:,}"
              f"  final_loss={r['final_loss']:.4f}"
              f"  acc_all={r['acc_all_positions']:.3f}"
              f"  acc_last={r['acc_last_position']:.3f}"
              f"  time={r['train_time_s']:.1f}s")

    print("\n=== Position-wise accuracy ===")
    print(f"pos:  " + " ".join(f"{i:>5d}" for i in range(L)))
    for r in results:
        print(f"{r['name']:14s} " + " ".join(f"{a:.2f}" for a in r['pos_acc']))

    print("\n=== Summary ===")
    for r in results:
        print(f"  {r['name']:14s}  acc@all={r['acc_all_positions']*100:5.1f}%"
              f"  acc@L-1={r['acc_last_position']*100:5.1f}%"
              f"  loss={r['final_loss']:.3f}")

    # Random baseline
    print(f"  Random          acc@all= 50.0%  acc@L-1= 50.0%")


if __name__ == "__main__":
    main()
