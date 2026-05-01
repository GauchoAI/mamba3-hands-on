"""Attempt 2: keep the parity-winning config, just train much longer.

Also: use a shorter sequence length during training (easier optimization)
and then evaluate on L=16. If mod 3 is solvable in principle by this model,
a longer training run should find it.
"""
import math, time
import torch, torch.nn as nn, torch.nn.functional as F
from mamba_platform.mamba3_minimal import Mamba3Block, Mamba3Config
from modular_counting import make_modular_batch, ModularModel


def run(M, device, steps=8000, L_train=8, L_eval=16, lr=3e-3):
    torch.manual_seed(0)
    cfg = Mamba3Config(d_model=32, d_state=16, expand=2, headdim=16)
    m = ModularModel(Mamba3Block, cfg, M).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)

    t0 = time.time()
    best = 0.0
    for step in range(steps):
        x, tgt = make_modular_batch(64, L_train, M, device)
        loss = F.cross_entropy(m(x).reshape(-1, M), tgt.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if (step + 1) % 500 == 0:
            m.eval()
            with torch.no_grad():
                xe, te = make_modular_batch(512, L_eval, M, device)
                acc = (m(xe).argmax(-1) == te).float().mean().item()
                xt, tt = make_modular_batch(512, L_train, M, device)
                acc_train_len = (m(xt).argmax(-1) == tt).float().mean().item()
            if acc > best: best = acc
            print(f"    step {step+1:5d}  loss={loss.item():.3f}  "
                  f"acc@L{L_train}={acc_train_len*100:5.1f}%  "
                  f"acc@L{L_eval}={acc*100:5.1f}%  best={best*100:5.1f}%")
            m.train()

    # final
    m.eval()
    with torch.no_grad():
        x, tgt = make_modular_batch(2048, L_eval, M, device)
        preds = m(x).argmax(-1)
        return m, (preds == tgt).float().mean().item(), (preds[:, -1] == tgt[:, -1]).float().mean().item(), time.time()-t0


def probe_angles(model, M, device):
    blk, emb, cfg = model.block, model.embed, model.block.cfg
    with torch.no_grad():
        def ph(v):
            toks = torch.full((1, 8), v, dtype=torch.long, device=device)
            u = emb(toks)
            proj = blk.in_proj(u)
            splits = [blk.d_inner, blk.d_inner, cfg.d_state, cfg.d_state,
                      blk.nheads, blk.nheads, blk.nheads, blk.num_rope_angles]
            _, _, _, _, dd_dt, _, _, angles = torch.split(proj, splits, dim=-1)
            DT = F.softplus(dd_dt + blk.dt_bias).mean(-1, keepdim=True)
            return (angles * DT)[0].mean(dim=0)                  # (d_state/2,)
        p0 = ph(0)
        print("  token 0 per-component phase:", p0.cpu().numpy().round(3).tolist())
        for v in range(1, M):
            diff = (ph(v) - p0).cpu().numpy()
            print(f"  token {v} - token 0 per-component:")
            for i, val in enumerate(diff):
                w = ((val + math.pi) % (2 * math.pi)) - math.pi
                # closest k*2*pi/M
                best_k, best_err = None, math.inf
                for k in range(M):
                    tgt = 2 * math.pi * k / M
                    tw = ((tgt + math.pi) % (2 * math.pi)) - math.pi
                    err = min(abs(w - tw), 2*math.pi - abs(w - tw))
                    if err < best_err:
                        best_err, best_k = err, k
                flag = "  <- close!" if best_err < 0.2 else ""
                print(f"    comp[{i}] = {w:+.3f} rad ({math.degrees(w):+6.1f}°)  "
                      f"closest: {best_k} * 2π/{M} = {2*math.pi*best_k/M:+.3f}  "
                      f"err={best_err:.3f}{flag}")


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}\n")

    for M in [3, 5]:
        print(f"{'='*60}\nMamba-3 on modular counting M={M}\n"
              f"(tiny config, 8000 steps, L_train=8, eval L=16, chance = {100/M:.1f}%)\n{'='*60}")
        m, a, al, t = run(M, device)
        print(f"\n  FINAL @ L=16: acc_all={a*100:5.1f}%  acc_last={al*100:5.1f}%  time={t:.1f}s")
        if a > 0.4:
            print("\n  --- learned per-step phase ---")
            probe_angles(m, M, device)
        print()


if __name__ == "__main__":
    main()
