"""Length generalization: train on L=16, test on L=16, 32, 64, 128.
A real state-tracker should extrapolate; a hack that just memorized short patterns won't.
"""
import torch, torch.nn as nn, torch.nn.functional as F
from lab_platform.mamba3_minimal import Mamba3Block, Mamba2LikeBlock, Mamba3Config
from parity_experiment import make_parity_batch, ParityModel

device = "mps" if torch.backends.mps.is_available() else "cpu"
cfg = Mamba3Config(d_model=32, d_state=16, expand=2, headdim=16)

def train(block_cls, L_train=16, steps=600, lr=3e-3):
    torch.manual_seed(0)
    m = ParityModel(block_cls, cfg).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    for _ in range(steps):
        bits, tgt = make_parity_batch(64, L_train, device)
        loss = F.cross_entropy(m(bits).reshape(-1,2), tgt.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    return m

def eval_at(model, L, N=1024):
    model.eval()
    with torch.no_grad():
        bits, tgt = make_parity_batch(N, L, device)
        preds = model(bits).argmax(-1)
        acc_last = (preds[:, -1] == tgt[:, -1]).float().mean().item()
        acc_all  = (preds == tgt).float().mean().item()
    return acc_all, acc_last

print(f"Training both on L=16, testing on {[16, 32, 64, 128]}")
print(f"Device: {device}")

for name, cls in [("Mamba-2-like", Mamba2LikeBlock), ("Mamba-3", Mamba3Block)]:
    print(f"\n--- Training {name} ---")
    m = train(cls)
    for L in [16, 32, 64, 128]:
        acc_all, acc_last = eval_at(m, L)
        print(f"  {name:14s}  L={L:4d}  acc_all={acc_all*100:5.1f}%  acc_last={acc_last*100:5.1f}%")
