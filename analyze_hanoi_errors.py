"""analyze_hanoi_errors — identify the 0.04% errors and look for patterns."""
import sys; sys.path.insert(0, ".")
import numpy as np, torch, torch.nn.functional as F
from discover_hanoi_aggregates_plain import (
    HanoiPlainMLP, generate_traces_for_ns, compute_aggregates,
    ACTION_PAIRS, N_ACTIONS,
)


def analyze(train_ns, test_ns, n_max_pad, steps=15000, d_hidden=128, device="mps"):
    rng = np.random.default_rng(0)
    train_pairs = generate_traces_for_ns(train_ns, n_max_pad)
    test_pairs  = generate_traces_for_ns(test_ns,  n_max_pad)
    train_states = np.array([p[0] for p in train_pairs], dtype=np.int64)
    train_actions = np.array([p[2] for p in train_pairs], dtype=np.int64)
    train_aggs = compute_aggregates(train_states, n_max_pad)
    test_states = np.array([p[0] for p in test_pairs], dtype=np.int64)
    test_n_disks = np.array([p[1] for p in test_pairs], dtype=np.int64)
    test_actions = np.array([p[2] for p in test_pairs], dtype=np.int64)
    test_aggs = compute_aggregates(test_states, n_max_pad)
    N = len(train_pairs)

    model = HanoiPlainMLP(n_max_pad=n_max_pad, d_hidden=d_hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=3e-3 * 0.05)
    for step in range(steps):
        idx = rng.integers(0, N, size=512)
        a = torch.tensor(train_aggs[idx], device=device)
        y = torch.tensor(train_actions[idx], device=device)
        logits = model(a)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

    # Identify errors on held-out test
    with torch.no_grad():
        t_a = torch.tensor(test_aggs, device=device)
        t_logits = model(t_a)
        preds = t_logits.argmax(-1).cpu().numpy()

    err_mask = preds != test_actions
    n_err = int(err_mask.sum())
    n_total = len(test_actions)
    print(f"Total test pairs: {n_total}, errors: {n_err} ({100*n_err/n_total:.4f}%)")

    if n_err == 0:
        return

    # Now look at the error states
    err_states = test_states[err_mask]
    err_aggs = test_aggs[err_mask]
    err_true = test_actions[err_mask]
    err_pred = preds[err_mask]

    # Are these ambiguous in training? Check if same aggregate appears in
    # training but with the OTHER action.
    from collections import defaultdict
    train_agg_to_actions = defaultdict(list)
    for i in range(len(train_pairs)):
        train_agg_to_actions[tuple(train_aggs[i].tolist())].append(int(train_actions[i]))

    # For each error, check:
    # - Does its aggregate appear in training?
    # - If yes, what action does training say?
    # - If not, the model has to generalize
    n_seen = 0
    n_correct_in_train = 0
    n_unseen = 0
    for i in range(min(n_err, 30)):
        agg = tuple(err_aggs[i].tolist())
        true = int(err_true[i])
        pred = int(err_pred[i])
        if agg in train_agg_to_actions:
            n_seen += 1
            train_acts = set(train_agg_to_actions[agg])
            if true in train_acts:
                n_correct_in_train += 1
        else:
            n_unseen += 1

        if i < 10:
            print(f"\nError {i+1}:")
            print(f"  state (peg per disk): {err_states[i].tolist()}")
            print(f"  agg: {agg}")
            print(f"  true action: {ACTION_PAIRS[true]} ({true})")
            print(f"  pred action: {ACTION_PAIRS[pred]} ({pred})")
            print(f"  agg in train?: {agg in train_agg_to_actions} {'(actions: ' + str(set(train_agg_to_actions[agg])) + ')' if agg in train_agg_to_actions else ''}")

    print(f"\nOf first {min(n_err, 30)} errors:")
    print(f"  agg seen in training: {n_seen}")
    print(f"  agg seen with the correct action: {n_correct_in_train}")
    print(f"  agg unseen in training: {n_unseen}")


if __name__ == "__main__":
    analyze([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], [17], 18,
            steps=15000, d_hidden=128, device="mps")
