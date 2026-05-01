"""Modular counting: generalise parity to any modulus M.

Task: input is a sequence of integers in {0, 1, ..., M-1}. Target at each position
is cumsum(inputs) mod M. This requires tracking a state in Z_M, which in the
complex-SSM view means being able to learn a rotation of 2*pi/M (or a multiple).

Parity is M=2 (rotation pi). Mod 3 needs 2*pi/3. Mod 5 needs 2*pi/5.

If Mamba-3's RoPE really lets it learn any angle, it should solve all of these.
Mamba-2-like should fail on all of them.
"""
import math
import torch, torch.nn as nn, torch.nn.functional as F
from lab_platform.mamba3_minimal import Mamba3Block, Mamba2LikeBlock, Mamba3Config


def make_modular_batch(batch: int, L: int, M: int, device):
    toks = torch.randint(0, M, (batch, L), device=device)
    tgt  = torch.cumsum(toks, dim=1) % M
    return toks, tgt


class ModularModel(nn.Module):
    def __init__(self, block_cls, cfg: Mamba3Config, M: int):
        super().__init__()
        self.embed = nn.Embedding(M, cfg.d_model)
        self.block = block_cls(cfg)
        self.norm  = nn.LayerNorm(cfg.d_model)
        self.head  = nn.Linear(cfg.d_model, M)

    def forward(self, x):
        u = self.embed(x)
        y = self.block(u) + u
        return self.head(self.norm(y))


def train_eval(block_cls, M, L=16, steps=800, lr=3e-3, device="mps"):
    torch.manual_seed(0)
    cfg = Mamba3Config(d_model=32, d_state=16, expand=2, headdim=16)
    m = ModularModel(block_cls, cfg, M).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    for _ in range(steps):
        x, tgt = make_modular_batch(64, L, M, device)
        loss = F.cross_entropy(m(x).reshape(-1, M), tgt.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    m.eval()
    with torch.no_grad():
        x, tgt = make_modular_batch(2048, L, M, device)
        preds = m(x).argmax(-1)
        acc_all  = (preds == tgt).float().mean().item()
        acc_last = (preds[:, -1] == tgt[:, -1]).float().mean().item()
    return m, acc_all, acc_last


def probe_learned_angles(model, M, device):
    """For each token value in 0..M-1, measure the per-step phase increment
    it induces relative to token 0. We expect an integer multiple of 2*pi/M.
    """
    blk, emb = model.block, model.embed
    cfg = blk.cfg

    with torch.no_grad():
        def phase_per_step_for_token(v):
            toks = torch.full((1, 8), v, dtype=torch.long, device=device)
            u = emb(toks)
            proj = blk.in_proj(u)
            splits = [blk.d_inner, blk.d_inner, cfg.d_state, cfg.d_state,
                      blk.nheads, blk.nheads, blk.nheads, blk.num_rope_angles]
            _, _, _, _, dd_dt, _, _, angles = torch.split(proj, splits, dim=-1)
            DT_mean = F.softplus(dd_dt + blk.dt_bias).mean(-1, keepdim=True)
            return (angles * DT_mean)[0]                         # (8, d_state/2)

        p0 = phase_per_step_for_token(0)
        print(f"  token 0 per-component phase (reference): "
              f"{p0.mean(dim=0).cpu().numpy().round(3).tolist()}")
        for v in range(1, M):
            pv = phase_per_step_for_token(v)
            diff = (pv - p0).mean(dim=0)                         # (d_state/2,)
            # find the component closest to the "ideal" 2*pi*v/M (mod 2*pi)
            ideal_angles = [2 * math.pi * v / M * k for k in range(M)]   # multiples
            best_k, best_err, best_comp = None, math.inf, None
            for c_idx, c_val in enumerate(diff.cpu().numpy()):
                # wrap c_val to (-pi, pi]
                w = ((c_val + math.pi) % (2 * math.pi)) - math.pi
                for k in range(M):
                    ideal = ((ideal_angles[k] + math.pi) % (2 * math.pi)) - math.pi
                    err = min(abs(w - ideal), abs(w - ideal + 2*math.pi), abs(w - ideal - 2*math.pi))
                    if err < best_err:
                        best_err, best_k, best_comp = err, k, c_idx
            w_best = ((diff[best_comp].item() + math.pi) % (2 * math.pi)) - math.pi
            target = 2 * math.pi * v / M
            print(f"  token {v}: best-matching component[{best_comp}] = "
                  f"{w_best:+.3f} rad ({math.degrees(w_best):+.1f}°)  "
                  f"→ closest ideal = {best_k} * 2π/{M} = {best_k * 2 * math.pi / M:+.3f} rad "
                  f"(err {best_err:.3f} rad)")


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}\n")
    for M in [2, 3, 5]:
        print(f"{'='*60}\nModulus M={M}  (ideal per-token rotation = 2π/{M} = {360/M:.1f}°)\n{'='*60}")
        for name, cls in [("Mamba-2-like", Mamba2LikeBlock), ("Mamba-3", Mamba3Block)]:
            m, acc_all, acc_last = train_eval(cls, M, device=device)
            print(f"{name:14s}  acc_all={acc_all*100:5.1f}%  acc_last={acc_last*100:5.1f}%  "
                  f"(random = {100/M:.1f}%)")
            if name == "Mamba-3" and acc_all > 0.8:
                print(f"  [probing learned angles]")
                probe_learned_angles(m, M, device)
        print()


if __name__ == "__main__":
    main()
