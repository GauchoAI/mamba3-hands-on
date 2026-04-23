"""Compare parity on CPU vs CUDA — same config, same seed."""
import torch
from mamba3_minimal import Mamba3Block, Mamba3Config
from parity_experiment import make_parity_batch, ParityModel, train_and_eval

cfg = Mamba3Config(d_model=32, d_state=16, expand=2, headdim=16)

print("Exact same config (d=32, dS=16, hd=16), seed=0, 400 steps")
print()

for device_name in ["cpu", "cuda"]:
    device = device_name if torch.cuda.is_available() or device_name == "cpu" else "cpu"
    r = train_and_eval("Mamba-3", Mamba3Block, cfg, device, steps=400, batch=64, L=16, lr=3e-3)
    acc_a = r["acc_all_positions"] * 100
    acc_l = r["acc_last_position"] * 100
    t = r["train_time_s"]
    print(f"  {device_name}: acc_all={acc_a:.1f}% acc_last={acc_l:.1f}% time={t:.1f}s loss={r['final_loss']:.4f}")
