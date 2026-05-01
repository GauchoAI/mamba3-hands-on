"""analyze_role_errors — characterize remaining errors with K=10 roles."""
import sys; sys.path.insert(0, ".")
import numpy as np, torch, torch.nn.functional as F
from discover_hanoi_roles import (
    HanoiRoleMLP, generate_traces_for_ns, role_features, ACTION_PAIRS,
    N_SMALL, N_LARGE,
)


def main():
    rng = np.random.default_rng(0)
    n_max_pad = 18
    train_ns = list(range(2, 16))
    test_ns = [16, 17]
    train_pairs = generate_traces_for_ns(train_ns, n_max_pad)
    test_pairs  = generate_traces_for_ns(test_ns,  n_max_pad)
    train_states = np.array([p[0] for p in train_pairs], dtype=np.int64)
    train_actions = np.array([p[2] for p in train_pairs], dtype=np.int64)
    train_feats = role_features(train_states, n_max_pad)
    test_states = np.array([p[0] for p in test_pairs], dtype=np.int64)
    test_n_disks = np.array([p[1] for p in test_pairs], dtype=np.int64)
    test_actions = np.array([p[2] for p in test_pairs], dtype=np.int64)
    test_feats = role_features(test_states, n_max_pad)
    N = len(train_pairs)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"K={N_SMALL}/{N_LARGE}  features per cell: {N_SMALL + N_LARGE + 4}")

    model = HanoiRoleMLP(d_hidden=128).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=25000, eta_min=3e-3 * 0.05)
    test_a_t = torch.tensor(test_feats, device=device)
    test_y_t = torch.tensor(test_actions, device=device)
    best_test_acc = 0.0
    best_state = None

    for step in range(25000):
        idx = rng.integers(0, N, size=512)
        a = torch.tensor(train_feats[idx], device=device)
        y = torch.tensor(train_actions[idx], device=device)
        logits = model(a)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if (step + 1) % 500 == 0:
            with torch.no_grad():
                test_acc = (model(test_a_t).argmax(-1) == test_y_t).float().mean().item()
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    with torch.no_grad():
        t_logits = model(test_a_t)
        preds = t_logits.argmax(-1).cpu().numpy()
    err_mask = preds != test_actions
    err_idx = np.where(err_mask)[0]
    print(f"Errors: {len(err_idx)} / {len(test_actions)} ({100*len(err_idx)/len(test_actions):.4f}%)")

    # Are these training role vectors?
    from collections import defaultdict
    train_role_to_actions = defaultdict(list)
    for i in range(N):
        train_role_to_actions[tuple(train_feats[i].tolist())].append(int(train_actions[i]))

    n_seen = 0; n_correct_in_train = 0; n_unseen = 0
    print("\nFirst 12 errors:")
    for ei in err_idx[:12]:
        feats = tuple(test_feats[ei].tolist())
        true = int(test_actions[ei])
        pred = int(preds[ei])
        n = int(test_n_disks[ei])
        seen = feats in train_role_to_actions
        if seen:
            n_seen += 1
            tas = set(train_role_to_actions[feats])
            if true in tas: n_correct_in_train += 1
        else:
            n_unseen += 1
        # Pretty print features
        smallest = feats[:N_SMALL]
        largest = feats[N_SMALL:N_SMALL+N_LARGE]
        rest = feats[N_SMALL+N_LARGE:]
        print(f"  n={n}: state[..n_disks]={test_states[ei][:n].tolist()}")
        print(f"    smallest_pegs[0..{N_SMALL-1}]={smallest}")
        print(f"    largest_pegs[0..{N_LARGE-1}]={largest}")
        print(f"    parity={rest[0]}, cmp_01={rest[1]}, cmp_02={rest[2]}, cmp_12={rest[3]}")
        print(f"    true={ACTION_PAIRS[true]}  pred={ACTION_PAIRS[pred]}  feats_seen_in_train={seen}", end="")
        if seen:
            print(f"  (train_actions for this vector: {set(train_role_to_actions[feats])})")
        else:
            print()
    print(f"\nSummary: {n_seen} feature-vectors seen in training, "
          f"{n_correct_in_train} of those had the correct action seen too. "
          f"{n_unseen} unseen.")


if __name__ == "__main__":
    main()
