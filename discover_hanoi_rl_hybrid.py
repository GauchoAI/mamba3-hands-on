"""discover_hanoi_rl_hybrid — imitation pretraining + REINFORCE fine-tune.

Phase 1 — behavior cloning: train the policy by copying optimal Hanoi
moves at every state seen during expert rollouts. Same supervised setup
as discover_hanoi_aggregates.py but as a starting point, not the end.

Phase 2 — RL fine-tune: switch to REINFORCE with potential-based reward
shaping. The policy already plays well; RL fine-tunes for the last
few mistakes (especially the held-out parity bit that imitation alone
doesn't capture cleanly).

Phase 3 — Eval: roll out on n_max_train+1, +2, +3 to test held-out.

Designed for m4-mini dispatch — should crack n=4 and beyond because the
imitation seed gives the agent a viable baseline policy, and RL only
needs to refine it rather than discover the algorithm from scratch.
"""
import argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ACTION_PAIRS = [(0,1), (0,2), (1,0), (1,2), (2,0), (2,1)]
ACTION_TO_IDX = {p: i for i, p in enumerate(ACTION_PAIRS)}
N_ACTIONS = len(ACTION_PAIRS)


def hanoi_moves(n, src=0, dst=2, aux=1):
    if n == 0: return
    yield from hanoi_moves(n - 1, src, aux, dst)
    yield (src, dst)
    yield from hanoi_moves(n - 1, aux, dst, src)


def aggregates_for_state(pegs, n_disks, n_max_pad):
    big = n_max_pad
    peg0 = pegs[0]
    peg1 = pegs[1] if n_disks >= 2 else 0
    count0 = sum(1 for p in pegs if p == 0)
    count1 = sum(1 for p in pegs if p == 1)
    count2 = sum(1 for p in pegs if p == 2)
    top_p0 = next((i for i in range(n_max_pad) if pegs[i] == 0), big)
    top_p1 = next((i for i in range(n_max_pad) if pegs[i] == 1), big)
    top_p2 = next((i for i in range(n_max_pad) if pegs[i] == 2), big)
    peg_largest = pegs[max(0, n_disks - 1)]
    return np.array([peg0, peg1, count0, count1, count2,
                     top_p0, top_p1, top_p2, n_disks, peg_largest], dtype=np.int64)


def expert_trace(n, n_max_pad):
    """Optimal trace: (aggs, actions) for n-disk Hanoi."""
    pegs = [0] * n_max_pad
    for i in range(n, n_max_pad): pegs[i] = -1
    aggs_seq, actions = [], []
    for src, dst in hanoi_moves(n):
        disk = next(i for i in range(n) if pegs[i] == src)
        aggs_seq.append(aggregates_for_state(pegs, n, n_max_pad))
        actions.append(ACTION_TO_IDX[(src, dst)])
        pegs[disk] = dst
    return np.array(aggs_seq, dtype=np.int64), np.array(actions, dtype=np.int64)


class HanoiPolicy(nn.Module):
    """Same architecture as the aggregate MLP but no discrete bottleneck."""
    def __init__(self, n_max_pad=16, d_emb=8, d_hidden=64):
        super().__init__()
        self.n_max_pad = n_max_pad
        self.peg_emb = nn.Embedding(3, d_emb)
        self.count_emb = nn.Embedding(n_max_pad + 2, d_emb)
        self.top_emb = nn.Embedding(n_max_pad + 1, d_emb)
        d_in = 10 * d_emb
        self.body = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_hidden), nn.ReLU(),
        )
        self.actor = nn.Linear(d_hidden, N_ACTIONS)
        self.critic = nn.Linear(d_hidden, 1)

    def features(self, obs):
        a0 = self.peg_emb(obs[:, 0]); a1 = self.peg_emb(obs[:, 1])
        c0 = self.count_emb(obs[:, 2]); c1 = self.count_emb(obs[:, 3])
        c2 = self.count_emb(obs[:, 4])
        t0 = self.top_emb(obs[:, 5]); t1 = self.top_emb(obs[:, 6])
        t2 = self.top_emb(obs[:, 7])
        nd = self.count_emb(obs[:, 8]); pl = self.peg_emb(obs[:, 9])
        return self.body(torch.cat([a0,a1,c0,c1,c2,t0,t1,t2,nd,pl], dim=-1))

    def forward(self, obs):
        h = self.features(obs)
        return self.actor(h), self.critic(h).squeeze(-1)


# ── Phase 1: imitation ──────────────────────────────────────────

def imitation_pretrain(policy, train_ns, n_max_pad, steps, batch, lr, device, verbose=True):
    """Standard supervised cross-entropy on expert moves."""
    rng = np.random.default_rng(0)
    pairs = []
    for n in train_ns:
        a, act = expert_trace(n, n_max_pad)
        for i in range(len(act)):
            pairs.append((a[i], act[i]))
    states = np.stack([p[0] for p in pairs])
    actions = np.array([p[1] for p in pairs])
    N = len(pairs)
    if verbose:
        print(f"[imitation] {N} (state, action) pairs from n in {train_ns}")
    opt = torch.optim.AdamW(policy.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.05)
    for step in range(steps):
        idx = rng.integers(0, N, size=batch)
        s = torch.tensor(states[idx], device=device)
        y = torch.tensor(actions[idx], device=device)
        logits, _ = policy(s)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step(); sched.step()
        if verbose and (step + 1) % 1000 == 0:
            with torch.no_grad():
                logits, _ = policy(torch.tensor(states, device=device))
                acc = (logits.argmax(-1) == torch.tensor(actions, device=device)).float().mean().item()
            print(f"  imit step {step+1:>5}  loss={loss.item():.5f}  train_acc={acc:.3%}")


# ── Phase 2: RL fine-tune ───────────────────────────────────────

class HanoiEnv:
    def __init__(self, n_max_pad=16, n_min=2, n_max=10):
        self.n_max_pad = n_max_pad
        self.n_min = n_min; self.n_max = n_max
        self.rng = np.random.default_rng(0)
        self.reset()

    def reset(self, n_disks=None):
        if n_disks is None:
            n_disks = int(self.rng.integers(self.n_min, self.n_max + 1))
        self.n_disks = n_disks
        self.pegs = [-1] * self.n_max_pad
        for i in range(n_disks): self.pegs[i] = 0
        self.steps = 0
        self.max_steps = (2 ** n_disks - 1) * 4 + 16
        self._potential_prev = self._potential()
        return self._obs()

    def _obs(self):
        return aggregates_for_state(self.pegs, self.n_disks, self.n_max_pad)

    def _potential(self):
        n_correct = 0
        for d in range(self.n_disks - 1, -1, -1):
            if self.pegs[d] == 2: n_correct += 1
            else: break
        return n_correct

    def legal_action_mask(self):
        big = self.n_max_pad
        top = [next((i for i in range(self.n_max_pad) if self.pegs[i] == p), big)
               for p in range(3)]
        mask = np.zeros(N_ACTIONS, dtype=np.bool_)
        for i, (src, dst) in enumerate(ACTION_PAIRS):
            if top[src] >= big: continue
            if top[dst] < big and top[dst] < top[src]: continue
            mask[i] = True
        return mask

    def step(self, action_idx):
        src, dst = ACTION_PAIRS[action_idx]
        big = self.n_max_pad
        disk = next((i for i in range(self.n_max_pad) if self.pegs[i] == src), big)
        if disk >= big:
            return self._obs(), -1.0, True, {"illegal": True}
        dst_top = next((i for i in range(self.n_max_pad) if self.pegs[i] == dst), big)
        if dst_top < big and dst_top < disk:
            return self._obs(), -1.0, True, {"illegal": True}
        self.pegs[disk] = dst
        self.steps += 1
        done = all(self.pegs[i] == 2 for i in range(self.n_disks))
        new_pot = self._potential()
        shaping = float(new_pot - self._potential_prev)
        self._potential_prev = new_pot
        reward = -0.02 + shaping + (float(self.n_disks) * 5.0 if done else 0.0)
        if self.steps >= self.max_steps: done = True
        return self._obs(), reward, done, {}


def rollout(env, policy, device, deterministic=False):
    obs = env.reset()
    obs_list, act_list, rew_list, val_list, logp_list, mask_list = [], [], [], [], [], []
    done = False
    while not done:
        mask = env.legal_action_mask()
        ot = torch.tensor(obs, device=device).unsqueeze(0)
        mt = torch.tensor(mask, device=device).unsqueeze(0)
        logits, value = policy(ot)
        logits = logits.masked_fill(~mt, -1e9)
        probs = F.softmax(logits, dim=-1)
        if deterministic:
            action = int(probs.argmax(-1).item())
        else:
            action = int(torch.multinomial(probs, 1).item())
        logp = F.log_softmax(logits, dim=-1)[0, action]
        obs_list.append(obs); act_list.append(action); mask_list.append(mask)
        val_list.append(float(value.item())); logp_list.append(logp)
        obs, reward, done, info = env.step(action)
        rew_list.append(reward)
    return {"obs": np.array(obs_list, dtype=np.int64),
            "actions": np.array(act_list, dtype=np.int64),
            "rewards": np.array(rew_list, dtype=np.float32),
            "values": np.array(val_list, dtype=np.float32),
            "logps": logp_list,
            "masks": np.array(mask_list, dtype=np.bool_),
            "n_disks": env.n_disks, "n_steps": env.steps,
            "solved": all(env.pegs[i] == 2 for i in range(env.n_disks))}


def rl_finetune(policy, n_max_pad, n_min, n_max_train, episodes, lr, gamma,
                value_coef, entropy_coef, eval_every, device, verbose=True):
    env = HanoiEnv(n_max_pad=n_max_pad, n_min=n_min, n_max=n_max_train)
    opt = torch.optim.AdamW(policy.parameters(), lr=lr)
    t0 = time.time()
    eval_ns = list(range(n_min, n_max_train + 4))
    for ep in range(episodes):
        traj = rollout(env, policy, device, deterministic=False)
        rewards = traj["rewards"]; T = len(rewards)
        returns = np.zeros_like(rewards); running = 0.0
        for t in range(T - 1, -1, -1):
            running = rewards[t] + gamma * running; returns[t] = running
        returns_t = torch.tensor(returns, device=device)
        values_t = torch.tensor(traj["values"], device=device)
        adv = returns_t - values_t
        if adv.numel() > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        logps = torch.stack(traj["logps"])
        actor_loss = -(logps * adv.detach()).mean()
        critic_loss = F.mse_loss(values_t, returns_t.detach())
        ot = torch.tensor(traj["obs"], device=device)
        mt = torch.tensor(traj["masks"], device=device)
        logits, _ = policy(ot)
        logits = logits.masked_fill(~mt, -1e9)
        probs = F.softmax(logits, dim=-1); logp = F.log_softmax(logits, dim=-1)
        entropy = -(probs * logp).sum(-1).mean()
        loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        if verbose and (ep + 1) % eval_every == 0:
            results = []
            with torch.no_grad():
                for n in eval_ns:
                    e = HanoiEnv(n_max_pad=n_max_pad, n_min=n, n_max=n)
                    t = rollout(e, policy, device, deterministic=True)
                    results.append({"n": n, "solved": t["solved"], "steps": t["n_steps"],
                                    "optimal": 2**n - 1})
            n_solved = sum(1 for r in results if r["solved"])
            n_optimal = sum(1 for r in results if r["solved"] and r["steps"] == r["optimal"])
            print(f"  rl ep {ep+1:>5}  ent={entropy.item():.3f}  solved={n_solved}/{len(eval_ns)}  optimal={n_optimal}  ({time.time()-t0:.0f}s)")
            for r in results:
                tag = "★" if r["solved"] and r["steps"] == r["optimal"] else ("✓" if r["solved"] else "✗")
                held = " (held)" if r["n"] > n_max_train else ""
                ratio = f"{r['steps']/r['optimal']:.2f}x" if r["solved"] else "FAIL"
                print(f"    n={r['n']:>2}  {tag}  steps={r['steps']:>5} / {r['optimal']:>5}  {ratio}{held}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-min", type=int, default=2)
    ap.add_argument("--n-max-train", type=int, default=8)
    ap.add_argument("--n-max-pad", type=int, default=16)
    ap.add_argument("--imit-steps", type=int, default=8000)
    ap.add_argument("--imit-batch", type=int, default=512)
    ap.add_argument("--imit-lr", type=float, default=3e-3)
    ap.add_argument("--rl-episodes", type=int, default=2000)
    ap.add_argument("--rl-lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--value-coef", type=float, default=0.5)
    ap.add_argument("--entropy-coef", type=float, default=0.01)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()
    print(f"Device: {args.device}\n")

    policy = HanoiPolicy(n_max_pad=args.n_max_pad).to(args.device)
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"HanoiPolicy params: {n_params}\n")

    train_ns = list(range(args.n_min, args.n_max_train + 1))

    print("══ Phase 1: Imitation pretrain ══")
    imitation_pretrain(policy, train_ns, args.n_max_pad,
                       args.imit_steps, args.imit_batch, args.imit_lr, args.device)

    print("\n══ Phase 2: RL fine-tune ══")
    rl_finetune(policy, args.n_max_pad, args.n_min, args.n_max_train,
                args.rl_episodes, args.rl_lr, args.gamma,
                args.value_coef, args.entropy_coef, args.eval_every, args.device)


if __name__ == "__main__":
    main()
