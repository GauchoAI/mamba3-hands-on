"""Test parity with forced backend: triton vs jit on CUDA."""
import torch
import ssm_triton
from mamba3_minimal import Mamba3Block, Mamba3Config
from parity_experiment import make_parity_batch, ParityModel, train_and_eval

cfg = Mamba3Config(d_model=32, d_state=16, expand=2, headdim=16)
device = "cuda"

print("Parity — forced backend comparison on CUDA")
print()

for backend in ["triton", "jit"]:
    ssm_triton.FORCE_BACKEND = backend
    r = train_and_eval(backend, Mamba3Block, cfg, device, steps=400, batch=64, L=16, lr=3e-3)
    acc = r["acc_all_positions"] * 100
    loss = r["final_loss"]
    t = r["train_time_s"]
    print(f"  {backend:7s}: acc={acc:.1f}% loss={loss:.4f} time={t:.1f}s")

ssm_triton.FORCE_BACKEND = None
