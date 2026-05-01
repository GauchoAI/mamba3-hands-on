"""discover_hanoi_roles_ensemble — train 3 role-encoding models, vote.

Run-to-run variance from random seeds is in [99.90%, 99.99%]. Ensembling
3 seeds and voting per-action should reduce errors by uncorrelated
disagreement. With legal-action masking, the ensemble agreement on the
true-true action should be high almost everywhere.
"""
import argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from discover_hanoi_roles import (
    HanoiRoleMLP, generate_traces_for_ns, role_features,
    legal_action_mask, ACTION_PAIRS, N_ACTIONS, N_SMALL, N_LARGE,
)


def train_one(seed, train_feats, train_actions, test_feats, test_legal,
              test_actions, steps, batch=512, lr=3e-3, d_hidden=128, device="cpu"):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = HanoiRoleMLP(d_hidden=d_hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.05)
    test_a_t = torch.tensor(test_feats, device=device)
    test_y_t = torch.tensor(test_actions, device=device)
    test_legal_t = torch.tensor(test_legal, device=device)
    best_test_acc = 0.0
    best_state = None
    N = len(train_feats)
    for step in range(steps):
        idx = rng.integers(0, N, size=batch)
        a = torch.tensor(train_feats[idx], device=device)
        y = torch.tensor(train_actions[idx], device=device)
        logits = model(a)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if (step + 1) % 500 == 0:
            with torch.no_grad():
                logits_eval = model(test_a_t).masked_fill(~test_legal_t, -1e9)
                test_acc = (logits_eval.argmax(-1) == test_y_t).float().mean().item()
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best_test_acc


def ensemble_eval(models, test_feats, test_legal, test_actions, device, n_disks_arr, test_ns):
    """Vote: sum softmax-probabilities across models, mask illegal, argmax."""
    test_a_t = torch.tensor(test_feats, device=device)
    test_y_t = torch.tensor(test_actions, device=device)
    test_legal_t = torch.tensor(test_legal, device=device)
    with torch.no_grad():
        avg_probs = None
        for m in models:
            m.eval()
            logits = m(test_a_t).masked_fill(~test_legal_t, -1e9)
            probs = F.softmax(logits, dim=-1)
            avg_probs = probs if avg_probs is None else avg_probs + probs
        avg_probs /= len(models)
        ensemble_pred = avg_probs.argmax(-1)
        acc = (ensemble_pred == test_y_t).float().mean().item()
    print(f"\nEnsemble of {len(models)} models  accuracy: {acc:.6%}  errors: {int((ensemble_pred != test_y_t).sum())}")
    print("  Per-n:")
    pred_np = ensemble_pred.cpu().numpy()
    for n in test_ns:
        mask = n_disks_arr == n
        if mask.sum() == 0: continue
        n_correct = int((pred_np[mask] == test_actions[mask]).sum())
        n_total = int(mask.sum())
        print(f"    n={n}: {n_correct}/{n_total} ({100*n_correct/n_total:.4f}%)")
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-ns", type=int, nargs="+",
                    default=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    ap.add_argument("--test-ns",  type=int, nargs="+", default=[16, 17])
    ap.add_argument("--n-max-pad", type=int, default=18)
    ap.add_argument("--n-models", type=int, default=3)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()
    print(f"Device: {args.device}\n")

    train_pairs = generate_traces_for_ns(args.train_ns, args.n_max_pad)
    test_pairs = generate_traces_for_ns(args.test_ns, args.n_max_pad)
    train_states = np.array([p[0] for p in train_pairs], dtype=np.int64)
    train_actions = np.array([p[2] for p in train_pairs], dtype=np.int64)
    train_feats = role_features(train_states, args.n_max_pad)
    test_states = np.array([p[0] for p in test_pairs], dtype=np.int64)
    test_n_disks = np.array([p[1] for p in test_pairs], dtype=np.int64)
    test_actions = np.array([p[2] for p in test_pairs], dtype=np.int64)
    test_feats = role_features(test_states, args.n_max_pad)
    test_legal = legal_action_mask(test_states, args.n_max_pad)
    print(f"Train pairs: {len(train_pairs)}, Test pairs: {len(test_pairs)}")

    models = []
    individual_accs = []
    t0 = time.time()
    for i in range(args.n_models):
        print(f"\n── Model {i+1}/{args.n_models}  seed={42+i} ──")
        model, acc = train_one(42 + i, train_feats, train_actions,
                                test_feats, test_legal, test_actions,
                                args.steps, device=args.device)
        models.append(model)
        individual_accs.append(acc)
        print(f"  best held-out: {acc:.6%}  elapsed={time.time()-t0:.0f}s")

    print(f"\nIndividual accuracies: {[f'{a:.4%}' for a in individual_accs]}")
    ensemble_eval(models, test_feats, test_legal, test_actions,
                  args.device, test_n_disks, args.test_ns)


if __name__ == "__main__":
    main()
