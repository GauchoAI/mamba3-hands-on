"""Test parity convergence with different register sizes."""
import torch
import time
from mamba3_minimal import Mamba3Block, Mamba3Config
from parity_experiment import make_parity_batch, ParityModel, train_and_eval

device = "cuda" if torch.cuda.is_available() else "cpu"

configs = [
    ("tiny d=16 dS=2 hd=4",    16,  2,  4),
    ("tiny d=16 dS=4 hd=4",    16,  4,  4),
    ("small d=32 dS=8 hd=8",   32,  8,  8),
    ("orig d=32 dS=16 hd=16",  32, 16, 16),
    ("current d=64 dS=16",     64, 16, 16),
]

print(f"Parity register size experiment on {device}")
print(f"{'config':30s} {'params':>7s} {'acc_all':>8s} {'acc_last':>9s} {'time':>6s} {'registers':>10s}")
print("-" * 75)

for name, d, ds, hd in configs:
    cfg = Mamba3Config(d_model=d, d_state=ds, expand=2, headdim=hd)
    nheads = (d * 2) // hd
    regs = nheads * hd * ds
    r = train_and_eval(name, Mamba3Block, cfg, device, steps=400, batch=64, L=16, lr=3e-3)
    acc_a = r["acc_all_positions"] * 100
    acc_l = r["acc_last_position"] * 100
    t = r["train_time_s"]
    p = r["n_params"]
    print(f"{name:30s} {p:>7,d} {acc_a:>7.1f}% {acc_l:>8.1f}% {t:>5.1f}s {regs:>10,d}")

print("\nDone.")
