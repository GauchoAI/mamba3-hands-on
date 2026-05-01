"""find_the_one_error — identify the single systematic error."""
import sys; sys.path.insert(0, ".")
import numpy as np, torch, torch.nn.functional as F
from discover_hanoi_roles import (
    HanoiRoleMLP, generate_traces_for_ns, role_features,
    legal_action_mask, ACTION_PAIRS, N_ACTIONS, N_SMALL, N_LARGE,
)
from discover_hanoi_roles_ensemble import train_one


def main():
    n_max_pad = 18
    train_ns = list(range(2, 16))
    test_ns = [16, 17]
    train_pairs = generate_traces_for_ns(train_ns, n_max_pad)
    test_pairs = generate_traces_for_ns(test_ns, n_max_pad)
    train_states = np.array([p[0] for p in train_pairs], dtype=np.int64)
    train_actions = np.array([p[2] for p in train_pairs], dtype=np.int64)
    train_feats = role_features(train_states, n_max_pad)
    test_states = np.array([p[0] for p in test_pairs], dtype=np.int64)
    test_n_disks = np.array([p[1] for p in test_pairs], dtype=np.int64)
    test_actions = np.array([p[2] for p in test_pairs], dtype=np.int64)
    test_feats = role_features(test_states, n_max_pad)
    test_legal = legal_action_mask(test_states, n_max_pad)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Train 5 models, average, find the 1 error
    models = []
    for i in range(5):
        m, _ = train_one(42 + i, train_feats, train_actions,
                          test_feats, test_legal, test_actions,
                          steps=20000, device=device)
        models.append(m)

    test_a_t = torch.tensor(test_feats, device=device)
    test_legal_t = torch.tensor(test_legal, device=device)
    with torch.no_grad():
        avg_probs = None
        for m in models:
            logits = m(test_a_t).masked_fill(~test_legal_t, -1e9)
            probs = F.softmax(logits, dim=-1)
            avg_probs = probs if avg_probs is None else avg_probs + probs
        avg_probs /= len(models)
        ensemble_pred = avg_probs.argmax(-1).cpu().numpy()

    err_idx = np.where(ensemble_pred != test_actions)[0]
    print(f"Ensemble errors: {len(err_idx)}")

    # Search for this error in training
    from collections import defaultdict
    train_role_to_actions = defaultdict(list)
    for i in range(len(train_pairs)):
        train_role_to_actions[tuple(train_feats[i].tolist())].append(int(train_actions[i]))

    for ei in err_idx:
        n = int(test_n_disks[ei])
        full_state = test_states[ei][:n].tolist()
        feats = tuple(test_feats[ei].tolist())
        true = int(test_actions[ei])
        pred = int(ensemble_pred[ei])
        seen = feats in train_role_to_actions
        # Show all model probs at this error
        per_model_preds = []
        with torch.no_grad():
            x = test_a_t[ei:ei+1]
            mask = test_legal_t[ei:ei+1]
            for m in models:
                logits = m(x).masked_fill(~mask, -1e9)
                probs_m = F.softmax(logits, dim=-1)[0].cpu().numpy()
                per_model_preds.append(probs_m)
        print(f"\nError at n={n}:")
        print(f"  full state (peg per disk): {full_state}")
        print(f"  smallest_pegs[0..{N_SMALL-1}]: {feats[:N_SMALL]}")
        print(f"  largest_pegs[0..{N_LARGE-1}]:  {feats[N_SMALL:N_SMALL+N_LARGE]}")
        print(f"  parity={feats[-4]}  cmp_01={feats[-3]}  cmp_02={feats[-2]}  cmp_12={feats[-1]}")
        print(f"  legal_mask: {test_legal[ei]}")
        print(f"  true action: {ACTION_PAIRS[true]} (idx {true})")
        print(f"  ensemble pred: {ACTION_PAIRS[pred]} (idx {pred})")
        print(f"  features seen in training? {seen}")
        if seen:
            print(f"  training said: {set(train_role_to_actions[feats])}")
        print(f"  per-model probabilities (legal-masked):")
        for i, p in enumerate(per_model_preds):
            top = np.argsort(p)[::-1]
            print(f"    seed{42+i}: " + ", ".join([f"{ACTION_PAIRS[j]}={p[j]:.3f}" for j in top if p[j] > 0.001]))


if __name__ == "__main__":
    main()
