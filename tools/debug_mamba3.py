"""Quick diagnostic: what's happening inside Mamba-3 on parity?"""
import torch, torch.nn as nn, torch.nn.functional as F
from lab_platform.mamba3_minimal import Mamba3Block, Mamba3Config

device = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(0)

cfg = Mamba3Config(d_model=32, d_state=16, expand=2, headdim=16)
blk = Mamba3Block(cfg).to(device)

# Inspect intermediate values for a single batch of bits
bits = torch.tensor([[0, 1, 1, 0, 1, 0, 0, 1]], device=device)
emb = nn.Embedding(2, cfg.d_model).to(device)
u = emb(bits)

proj = blk.in_proj(u)
splits = [blk.d_inner, blk.d_inner, cfg.d_state, cfg.d_state, blk.nheads, blk.nheads, blk.nheads, blk.num_rope_angles]
z, x, Bp, Cp, dd_dt, dd_A, trap_raw, angles = torch.split(proj, splits, dim=-1)
DT = F.softplus(dd_dt + blk.dt_bias)
A  = -F.softplus(dd_A).clamp(max=-cfg.A_floor)
ADT = A * DT
decay = torch.exp(ADT)

print("DT stats:      ", DT.mean().item(),  DT.min().item(),  DT.max().item())
print("A stats:       ", A.mean().item(),   A.min().item(),   A.max().item())
print("decay stats:   ", decay.mean().item(),decay.min().item(), decay.max().item())
print("angles stats:  ", angles.mean().item(), angles.min().item(), angles.max().item())
print("trap sig stats:", torch.sigmoid(trap_raw).mean().item(),
                          torch.sigmoid(trap_raw).min().item(),
                          torch.sigmoid(trap_raw).max().item())

DT_mean = DT.mean(dim=-1, keepdim=True)
phase = torch.cumsum(angles * DT_mean, dim=1)
print("phase (cum):   ", phase[0].detach().cpu().numpy())
