"""discover_hanoi_rl — Hanoi via reinforcement learning.

The agent observes the aggregate-feature state and picks one of 6 moves.
Reward = -1 per step (so optimal policy minimizes moves) + bonus on solve.
Illegal moves are masked at the policy head — the agent can only choose
among legal moves.

Algorithm: REINFORCE with returns-as-advantage, parallel envs across
different n values. Each episode the env samples a random n in
[n_min, n_max_train], runs until solved or max_steps.

Held-out test: roll out the trained policy on n_max_train+1, +2, etc.
A perfectly generalizing policy solves any n in 2^n − 1 moves.
"""
import argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ACTION_PAIRS = [(0,1), (0,2), (1,0), (1,2), (2,0), (2,1)]
N_ACTIONS = len(ACTION_PAIRS)


class HanoiEnv:
    """Tower of Hanoi as an RL environment.

    Observation: 10 aggregate features (n-invariant by construction).
    Actions: 6 (src, dst) pairs.
    Illegal moves return reward -1 + episode-end (no episode termination
    since we mask actions, but defensive default).
    """
    def __init__(self, n_max_pad: int = 16, n_min: int = 2, n_max: int = 10):
        self.n_max_pad = n_max_pad
        self.n_min = n_min
        self.n_max = n_max
        self.rng = np.random.default_rng(0)
        self.reset()

    def reset(self, n_disks: int = None):
        if n_disks is None:
            n_disks = int(self.rng.integers(self.n_min, self.n_max + 1))
        self.n_disks = n_disks
        self.pegs = [-1] * self.n_max_pad
        for i in range(n_disks):
            self.pegs[i] = 0
        self.steps = 0
        # Optimal step count is 2^n - 1; cap at 8x so nonsense policies still terminate.
        self.max_steps = (2 ** n_disks - 1) * 8 + 16
        self._potential_prev = self._potential()
        return self._obs()

    def _potential(self):
        """Number of largest disks correctly stacked on peg 2 from the bottom.
        i.e. n_correct = largest m such that disks n_disks-1, n_disks-2, ...,
        n_disks-m are all on peg 2.
        Used for potential-based reward shaping (preserves optimality).
        """
        n_correct = 0
        for d in range(self.n_disks - 1, -1, -1):
            if self.pegs[d] == 2:
                n_correct += 1
            else:
                break
        return n_correct

    def _obs(self):
        big = self.n_max_pad
        peg0 = self.pegs[0]
        peg1 = self.pegs[1] if self.n_disks >= 2 else 0
        count0 = sum(1 for p in self.pegs if p == 0)
        count1 = sum(1 for p in self.pegs if p == 1)
        count2 = sum(1 for p in self.pegs if p == 2)
        top_p0 = next((i for i in range(self.n_max_pad) if self.pegs[i] == 0), big)
        top_p1 = next((i for i in range(self.n_max_pad) if self.pegs[i] == 1), big)
        top_p2 = next((i for i in range(self.n_max_pad) if self.pegs[i] == 2), big)
        peg_largest = self.pegs[max(0, self.n_disks - 1)]
        return np.array([peg0, peg1, count0, count1, count2,
                         top_p0, top_p1, top_p2, self.n_disks, peg_largest],
                        dtype=np.int64)

    def legal_action_mask(self):
        big = self.n_max_pad
        top = [next((i for i in range(self.n_max_pad) if self.pegs[i] == p), big)
               for p in range(3)]
        mask = np.zeros(N_ACTIONS, dtype=np.bool_)
        for i, (src, dst) in enumerate(ACTION_PAIRS):
            if top[src] >= big:                            # src is empty
                continue
            if top[dst] < big and top[dst] < top[src]:    # dst has smaller disk on top
                continue
            mask[i] = True
        return mask

    def step(self, action_idx: int):
        src, dst = ACTION_PAIRS[action_idx]
        # Find top disk on src
        big = self.n_max_pad
        disk = next((i for i in range(self.n_max_pad) if self.pegs[i] == src), big)
        if disk >= big:
            return self._obs(), -1.0, True, {"illegal": True}
        # Check destination is legal
        dst_top = next((i for i in range(self.n_max_pad) if self.pegs[i] == dst), big)
        if dst_top < big and dst_top < disk:
            return self._obs(), -1.0, True, {"illegal": True}
        self.pegs[disk] = dst
        self.steps += 1
        # Goal: all disks on peg 2
        done = all(self.pegs[i] == 2 for i in range(self.n_disks))
        # Potential-based shaping: reward when the bottom-correct stack grows.
        # +1 for each newly correctly-placed big disk, -1 if one comes off.
        new_potential = self._potential()
        shaping = float(new_potential - self._potential_prev)
        self._potential_prev = new_potential
        reward = -0.05 + shaping + (float(self.n_disks) * 5.0 if done else 0.0)
        if self.steps >= self.max_steps:
            done = True
        return self._obs(), reward, done, {}


class HanoiPolicy(nn.Module):
    def __init__(self, n_max_pad: int = 16, d_emb: int = 8, d_hidden: int = 64):
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
        # obs: (B, 10) ints
        a0 = self.peg_emb(obs[:, 0])
        a1 = self.peg_emb(obs[:, 1])
        c0 = self.count_emb(obs[:, 2])
        c1 = self.count_emb(obs[:, 3])
        c2 = self.count_emb(obs[:, 4])
        t0 = self.top_emb(obs[:, 5])
        t1 = self.top_emb(obs[:, 6])
        t2 = self.top_emb(obs[:, 7])
        nd = self.count_emb(obs[:, 8])
        pl = self.peg_emb(obs[:, 9])
        x = torch.cat([a0, a1, c0, c1, c2, t0, t1, t2, nd, pl], dim=-1)
        return self.body(x)

    def forward(self, obs):
        h = self.features(obs)
        return self.actor(h), self.critic(h).squeeze(-1)

    def policy(self, obs, mask):
        logits, value = self.forward(obs)
        # Mask illegal actions
        logits = logits.masked_fill(~mask, -1e9)
        return logits, value


def rollout(env, policy, device, deterministic=False):
    obs = env.reset()
    obs_list, action_list, reward_list, value_list, logp_list, mask_list = [], [], [], [], [], []
    done = False
    while not done:
        mask = env.legal_action_mask()
        obs_t = torch.tensor(obs, device=device).unsqueeze(0)
        mask_t = torch.tensor(mask, device=device).unsqueeze(0)
        logits, value = policy.policy(obs_t, mask_t)
        probs = F.softmax(logits, dim=-1)
        if deterministic:
            action = int(probs.argmax(-1).item())
        else:
            action = int(torch.multinomial(probs, 1).item())
        logp = F.log_softmax(logits, dim=-1)[0, action]
        obs_list.append(obs)
        action_list.append(action)
        mask_list.append(mask)
        value_list.append(float(value.item()))
        logp_list.append(logp)
        obs, reward, done, info = env.step(action)
        reward_list.append(reward)
    return {
        "obs": np.array(obs_list, dtype=np.int64),
        "actions": np.array(action_list, dtype=np.int64),
        "rewards": np.array(reward_list, dtype=np.float32),
        "values": np.array(value_list, dtype=np.float32),
        "logps": logp_list,
        "masks": np.array(mask_list, dtype=np.bool_),
        "n_disks": env.n_disks,
        "n_steps": env.steps,
        "solved": all(env.pegs[i] == 2 for i in range(env.n_disks)),
    }


def evaluate(policy, device, n_disks: int, n_max_pad: int):
    env = HanoiEnv(n_max_pad=n_max_pad, n_min=n_disks, n_max=n_disks)
    traj = rollout(env, policy, device, deterministic=True)
    optimal_steps = 2 ** n_disks - 1
    return {
        "n": n_disks,
        "solved": traj["solved"],
        "steps": traj["n_steps"],
        "optimal": optimal_steps,
        "ratio": traj["n_steps"] / optimal_steps if optimal_steps > 0 else float("inf"),
    }


def train_rl(n_max_pad=16, n_min=2, n_max_train=10,
             episodes=4000, lr=3e-3, gamma=0.99,
             value_coef=0.5, entropy_coef=0.02,
             eval_every=200, device="cpu", verbose=True,
             curriculum=True, advance_threshold=0.9):
    """RL with curriculum: start at n=n_min, advance the upper bound when
    recent success rate at the current ceiling exceeds advance_threshold.
    """
    env = HanoiEnv(n_max_pad=n_max_pad, n_min=n_min, n_max=n_min if curriculum else n_max_train)
    policy = HanoiPolicy(n_max_pad=n_max_pad).to(device)
    n_params = sum(p.numel() for p in policy.parameters())
    if verbose:
        print(f"HanoiPolicy params: {n_params}  curriculum={curriculum}")
    opt = torch.optim.AdamW(policy.parameters(), lr=lr)

    t0 = time.time()
    eval_ns = list(range(n_min, n_max_train + 6))   # train + held-out
    recent_solves = {n: [] for n in range(n_min, n_max_train + 1)}
    for ep in range(episodes):
        traj = rollout(env, policy, device, deterministic=False)
        # Track success rate at the current top-of-curriculum n
        if traj["n_disks"] in recent_solves:
            recent_solves[traj["n_disks"]].append(int(traj["solved"]))
            recent_solves[traj["n_disks"]] = recent_solves[traj["n_disks"]][-100:]
        # Advance curriculum if top n has ≥advance_threshold success rate over last 50 eps
        if curriculum and env.n_max < n_max_train:
            top = env.n_max
            recent = recent_solves[top][-50:]
            if len(recent) >= 30 and (sum(recent) / len(recent)) >= advance_threshold:
                env.n_max += 1
                if verbose:
                    print(f"  → curriculum advance: n_max = {env.n_max}")
        # Compute discounted returns
        rewards = traj["rewards"]
        T = len(rewards)
        returns = np.zeros_like(rewards)
        running = 0.0
        for t in range(T - 1, -1, -1):
            running = rewards[t] + gamma * running
            returns[t] = running
        returns_t = torch.tensor(returns, device=device)
        values_t = torch.tensor(traj["values"], device=device)
        advantages = returns_t - values_t
        # Normalise advantages for stability
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        logps = torch.stack(traj["logps"])
        # Actor loss: -E[logp * advantage]
        actor_loss = -(logps * advantages.detach()).mean()
        # Critic loss: MSE on returns
        critic_loss = F.mse_loss(values_t, returns_t.detach())
        # Entropy: encourage exploration (we re-evaluate logits to compute it)
        obs_t = torch.tensor(traj["obs"], device=device)
        mask_t = torch.tensor(traj["masks"], device=device)
        logits, _ = policy.policy(obs_t, mask_t)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(-1).mean()

        loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        opt.step()

        if verbose and (ep + 1) % eval_every == 0:
            results = [evaluate(policy, device, n, n_max_pad) for n in eval_ns]
            elapsed = time.time() - t0
            n_solved = sum(1 for r in results if r["solved"])
            print(f"  ep {ep+1:>5}  loss={loss.item():.3f}  ent={entropy.item():.3f}  "
                  f"solved={n_solved}/{len(eval_ns)}  ({elapsed:.0f}s)")
            for r in results:
                tag = "✓" if r["solved"] else "✗"
                ratio = f"{r['ratio']:.2f}x optimal" if r["solved"] else "FAIL"
                held = " (held-out)" if r["n"] > n_max_train else ""
                print(f"    n={r['n']:>2}  {tag}  steps={r['steps']:>6} / {r['optimal']:>6} "
                      f"({ratio}){held}")
    return policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-min", type=int, default=2)
    ap.add_argument("--n-max-train", type=int, default=8)
    ap.add_argument("--n-max-pad", type=int, default=16)
    ap.add_argument("--episodes", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()
    print(f"Device: {args.device}\n")
    train_rl(n_max_pad=args.n_max_pad, n_min=args.n_min,
             n_max_train=args.n_max_train, episodes=args.episodes,
             lr=args.lr, gamma=args.gamma, eval_every=args.eval_every,
             device=args.device)


if __name__ == "__main__":
    main()
