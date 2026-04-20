"""Try to crack mod 3 (and mod 5) by scaling the minimal Mamba-3.

Hypothesis: the 8k-param config was capacity-starved, not algorithmically unable.
Bigger d_state / d_inner + more training should find 2*pi/3 the same way it
found -pi for parity.
"""
import math, time
import torch, torch.nn as nn, torch.nn.functional as F
from mamba3_minimal import Mamba3Block, Mamba2LikeBlock, Mamba3Config
from modular_counting import make_modular_batch, ModularModel


def train_eval(block_cls, M, cfg: Mamba3Config, device,
               steps=2000, L=16, batch=64, lr=3e-3, warmup=50):
    torch.manual_seed(0)
    m = ModularModel(block_cls, cfg, M).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda s: min(1.0, (s + 1) / warmup))

    t0 = time.time()
    best_acc = 0.0
    for step in range(steps):
        x, tgt = make_modular_batch(batch, L, M, device)
        loss = F.cross_entropy(m(x).reshape(-1, M), tgt.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); sched.step()
        if (step + 1) % 200 == 0:
            m.eval()
            with torch.no_grad():
                xe, te = make_modular_batch(512, L, M, device)
                acc = (m(xe).argmax(-1) == te).float().mean().item()
            best_acc = max(best_acc, acc)
            print(f"    step {step+1:4d}  loss={loss.item():.3f}  acc={acc*100:5.1f}%  best={best_acc*100:5.1f}%")
            m.train()

    # final eval
    m.eval()
    with torch.no_grad():
        x, tgt = make_modular_batch(2048, L, M, device)
        preds = m(x).argmax(-1)
        acc_all  = (preds == tgt).float().mean().item()
        acc_last = (preds[:, -1] == tgt[:, -1]).float().mean().item()
    return m, acc_all, acc_last, time.time() - t0


def probe_angles(model, M, device):
    """Report the per-step phase diff for each token value vs token 0,
    and check how close any component is to k*2pi/M for some k."""
    blk, emb = model.block, model.embed
    cfg = blk.cfg
    with torch.no_grad():
        def ph(v):
            toks = torch.full((1, 8), v, dtype=torch.long, device=device)
            u = emb(toks)
            proj = blk.in_proj(u)
            splits = [blk.d_inner, blk.d_inner, cfg.d_state, cfg.d_state,
                      blk.nheads, blk.nheads, blk.nheads, blk.num_rope_angles]
            _, _, _, _, dd_dt, _, _, angles = torch.split(proj, splits, dim=-1)
            DT = F.softplus(dd_dt + blk.dt_bias).mean(-1, keepdim=True)
            return (angles * DT)[0].mean(dim=0)                   # (d_state/2,)
        p0 = ph(0)
        for v in range(1, M):
            diff = (ph(v) - p0).cpu().numpy()
            best = None
            for i, val in enumerate(diff):
                w = ((val + math.pi) % (2 * math.pi)) - math.pi
                for k in range(1, M):
                    target = 2 * math.pi * v * k / M
                    tw = ((target + math.pi) % (2 * math.pi)) - math.pi
                    err = min(abs(w - tw), 2*math.pi - abs(w - tw))
                    if best is None or err < best[0]:
                        best = (err, i, k, w, target)
            err, i, k, w, tgt = best
            print(f"    token {v}: best comp[{i}] = {w:+.3f} rad "
                  f"({math.degrees(w):+6.1f}°), closest = {k}*2π·{v}/{M} "
                  f"= {tgt:+.3f} (err {err:.3f})")


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}\n")

    # bigger minimal: 4x params roughly
    cfg_big = Mamba3Config(d_model=64, d_state=32, expand=2, headdim=32)

    for M in [2, 3, 5]:
        print(f"{'='*60}\nModulus M={M}  (d_model={cfg_big.d_model}, d_state={cfg_big.d_state}, "
              f"expand={cfg_big.expand}, headdim={cfg_big.headdim})\n{'='*60}")
        for name, cls in [("Mamba-2-like", Mamba2LikeBlock), ("Mamba-3", Mamba3Block)]:
            print(f"  [{name}]")
            m, a, al, t = train_eval(cls, M, cfg_big, device, steps=2000)
            print(f"    final: acc_all={a*100:5.1f}%  acc_last={al*100:5.1f}%  "
                  f"(random={100/M:.1f}%)  time={t:.1f}s")
            if name == "Mamba-3" and a > 0.6:
                probe_angles(m, M, device)
        print()


if __name__ == "__main__":
    main()
