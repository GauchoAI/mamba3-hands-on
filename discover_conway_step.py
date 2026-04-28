"""discover_conway_step — Conway's Game of Life rule discovery.

Observations: (cell_alive ∈ {0,1}, n_alive_neighbors ∈ {0..8}, next_alive)
Actions: 0=dead, 1=alive

True minimum codebook: 2 codes (one per action). The 18 (state, n) pairs
collapse onto 2 outputs, so the encoder should partition the 18-pair input
space into "next becomes alive" (3 pairs) vs "next becomes dead" (15 pairs).
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def conway_action(alive: int, n_alive: int) -> int:
    if alive == 1 and n_alive in (2, 3):
        return 1
    if alive == 0 and n_alive == 3:
        return 1
    return 0


class DiscoveryConway(nn.Module):
    def __init__(self, K: int = 8, d_emb: int = 4, d_hidden: int = 16):
        super().__init__()
        self.K = K
        # Discrete input: alive ∈ {0,1}, n_alive ∈ {0..8}.
        # Use embeddings rather than scalars — the encoder discovers
        # how to combine them, doesn't get to cheat with arithmetic.
        self.alive_emb = nn.Embedding(2, d_emb)
        self.neigh_emb = nn.Embedding(9, d_emb)
        self.enc = nn.Sequential(
            nn.Linear(2 * d_emb, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, K),
        )
        self.dec = nn.Linear(K, 2)

    def forward(self, alive: torch.Tensor, neigh: torch.Tensor, tau: float = 1.0):
        e = torch.cat([self.alive_emb(alive), self.neigh_emb(neigh)], dim=-1)
        logits = self.enc(e)
        code = F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)
        return self.dec(code), code, logits


def sample_batch(rng, batch):
    alive = rng.integers(0, 2, batch).astype(np.int64)
    neigh = rng.integers(0, 9, batch).astype(np.int64)
    actions = np.array([conway_action(int(a), int(n)) for a, n in zip(alive, neigh)], dtype=np.int64)
    return alive, neigh, actions


def train(K: int, steps: int = 2000, batch: int = 512, lr: float = 1e-2,
          device: str = "cpu", verbose: bool = True):
    rng = np.random.default_rng(0)
    model = DiscoveryConway(K=K).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"K={K}  params={n_params}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    for step in range(steps):
        alive, neigh, actions = sample_batch(rng, batch)
        a_t = torch.tensor(alive, device=device)
        n_t = torch.tensor(neigh, device=device)
        y = torch.tensor(actions, device=device)
        tau = max(0.1, 1.0 * (1.0 - step / steps))
        action_logits, code, _ = model(a_t, n_t, tau=tau)
        loss = F.cross_entropy(action_logits, y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

        if verbose and (step + 1) % 200 == 0:
            with torch.no_grad():
                action_logits, code, _ = model(a_t, n_t, tau=0.01)
                acc = (action_logits.argmax(-1) == y).float().mean().item()
                code_idx = code.argmax(-1)
                n_used = int(torch.unique(code_idx).numel())
            print(f"  step {step+1:>4}  loss={loss.item():.4f}  acc={acc:.1%}  codes_used={n_used}/{K}")

    # Evaluate on the full 18-cell truth table
    print(f"\nFull truth table evaluation (all 18 states):")
    print(f"  {'alive':>5} {'neigh':>5} {'true':>5}  →  {'code':>4}  {'pred':>5}  {'  a→':>4}")
    correct = 0
    code_to_action_map = {}
    code_to_states = {}
    for a in range(2):
        for n in range(9):
            true_action = conway_action(a, n)
            a_t = torch.tensor([a], device=device)
            n_t = torch.tensor([n], device=device)
            with torch.no_grad():
                action_logits, code, _ = model(a_t, n_t, tau=0.01)
                pred = int(action_logits.argmax(-1).item())
                c = int(code.argmax(-1).item())
            mark = "✓" if pred == true_action else "✗"
            if pred == true_action: correct += 1
            print(f"  {a:>5} {n:>5} {true_action:>5}  →  {c:>4}  {pred:>5}  {mark}")
            code_to_action_map.setdefault(c, []).append(pred)
            code_to_states.setdefault(c, []).append((a, n, true_action))

    print(f"\nAccuracy: {correct}/18 ({100*correct/18:.1f}%)")
    print(f"Codes used: {len(code_to_states)}")
    print(f"\nDiscovered partition:")
    for c in sorted(code_to_states):
        states = code_to_states[c]
        actions_in_c = [s[2] for s in states]
        a_pred = code_to_action_map[c][0]
        action_name = ['dead', 'alive'][a_pred]
        print(f"  code {c} → {action_name}  ({len(states)} states):")
        for a, n, true_act in states:
            mark = "✓" if true_act == a_pred else "✗"
            print(f"    (alive={a}, n={n}, true={['dead','alive'][true_act]})  {mark}")
    return correct / 18


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K-list", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()
    print(f"Device: {args.device}")
    for K in args.K_list:
        print(f"\n══════════ Codebook size K = {K} ══════════")
        train(K=K, steps=args.steps, device=args.device)


if __name__ == "__main__":
    main()
