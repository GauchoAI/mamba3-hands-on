"""Open the black box: what rotations did the trained Mamba-3 actually learn?

Theory predicts: bit=1 -> ~pi per step (flip), bit=0 -> ~0 per step (identity).
Let's check.
"""
import math, torch, torch.nn as nn, torch.nn.functional as F
from mamba_platform.mamba3_minimal import Mamba3Block, Mamba3Config
from parity_experiment import make_parity_batch, ParityModel

device = "mps" if torch.backends.mps.is_available() else "cpu"
cfg = Mamba3Config(d_model=32, d_state=16, expand=2, headdim=16)

# Re-train (cheap; same seed as parity_experiment)
torch.manual_seed(0)
model = ParityModel(Mamba3Block, cfg).to(device)
opt   = torch.optim.AdamW(model.parameters(), lr=3e-3)
for _ in range(600):
    bits, tgt = make_parity_batch(64, 16, device)
    loss = F.cross_entropy(model(bits).reshape(-1,2), tgt.reshape(-1))
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()

# Confirm it learned
model.eval()
with torch.no_grad():
    bits, tgt = make_parity_batch(512, 16, device)
    acc = (model(bits).argmax(-1) == tgt).float().mean().item()
print(f"Trained Mamba-3 parity acc at L=16: {acc*100:.1f}%\n")

# --- Probe: what phase does each bit induce? ---
# Feed a single constant bit sequence and inspect the per-step phase increment.
blk = model.block
emb = model.embed

with torch.no_grad():
    for bit_value in [0, 1]:
        bits = torch.full((1, 8), bit_value, dtype=torch.long, device=device)
        u = emb(bits)
        proj = blk.in_proj(u)
        splits = [blk.d_inner, blk.d_inner, cfg.d_state, cfg.d_state,
                  blk.nheads, blk.nheads, blk.nheads, blk.num_rope_angles]
        z, x, Bp, Cp, dd_dt, dd_A, trap_raw, angles = torch.split(proj, splits, dim=-1)
        DT = F.softplus(dd_dt + blk.dt_bias)                     # (1, 8, H)
        DT_mean = DT.mean(dim=-1, keepdim=True)                  # (1, 8, 1)
        phase_step = (angles * DT_mean)[0]                       # (8, d_state/2)

        print(f"=== bit = {bit_value} ===")
        print(f"  DT (mean per step):       {DT_mean.squeeze().cpu().numpy().round(4).tolist()}")
        print(f"  angle * DT (per step, avg over components): "
              f"{phase_step.mean(dim=-1).cpu().numpy().round(4).tolist()}")
        print(f"  angle * DT (per step, std over components): "
              f"{phase_step.std(dim=-1).cpu().numpy().round(4).tolist()}")
        # Average phase step across all 8 tokens (stationary input)
        avg_step = phase_step.mean().item()
        print(f"  average per-step phase across components & steps: {avg_step:+.4f} rad"
              f"  ({math.degrees(avg_step):+.1f} deg)")
        print(f"  |avg step| / pi:                                 {abs(avg_step)/math.pi:.3f}")
        print()

# Compare the difference: bit=1 should induce a LARGER magnitude rotation
# per step than bit=0 (ideally close to pi).
with torch.no_grad():
    def phase_for_bit(v):
        bits = torch.full((1, 8), v, dtype=torch.long, device=device)
        u = emb(bits)
        proj = blk.in_proj(u)
        _, _, _, _, dd_dt, _, _, angles = torch.split(
            proj,
            [blk.d_inner, blk.d_inner, cfg.d_state, cfg.d_state,
             blk.nheads, blk.nheads, blk.nheads, blk.num_rope_angles],
            dim=-1)
        DT = F.softplus(dd_dt + blk.dt_bias).mean(-1, keepdim=True)
        return (angles * DT)[0]                                   # (8, d_state/2)

    p0 = phase_for_bit(0)
    p1 = phase_for_bit(1)
    diff = (p1 - p0)                                              # difference per step

    # The "toggle" component: how much MORE phase does bit=1 induce vs bit=0?
    print("=== bit=1 minus bit=0 (the actual 'toggle' rotation) ===")
    print(f"  component-wise mean (per step):  {diff.mean(dim=-1).cpu().numpy().round(4).tolist()}")
    print(f"  overall mean:        {diff.mean().item():+.4f} rad = {math.degrees(diff.mean().item()):+.1f} deg")
    print(f"  |mean| / pi:         {abs(diff.mean().item())/math.pi:.3f}")
    # Also show per-component (8 components of d_state/2 = 8 angles)
    print(f"  per-component mean phase diff (across time):")
    per_comp = diff.mean(dim=0).cpu().numpy()
    for i, v in enumerate(per_comp):
        print(f"    angle[{i}]: {v:+.4f} rad  ({math.degrees(v):+.1f} deg, {v/math.pi:+.3f} * pi)")
