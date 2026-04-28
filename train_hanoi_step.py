"""train_hanoi_step — train the Hanoi step function on a wide
curriculum with multi-channel state feedback.

The model is a pure step function: f(state, byte_history) -> next_byte.
State comes in as K parallel channels (one int per disk's peg) via
the parameter-free MultiChannelStateFeedback module. Termination is
oracle-supplied via LoopCounter.

The architectural primitive scales to any K (any n disks); training
this run focuses on making the function GENERALIZE — wider curriculum
+ slightly bigger model so memorization can't shortcut.
"""
import argparse, json, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from progressive_model import ProgressiveModel, BOS, SEP, EOS, PAD
from hanoi_tool import hanoi_moves, precompute_channels, PEG_NAMES


def encode_move(k, src, dst):
    return list(f"{k} {PEG_NAMES[src]} {PEG_NAMES[dst]}\n".encode("ascii"))


def build_example(n: int, K: int):
    inp_bytes = list(f"HANOI {n}".encode("utf-8"))
    moves = hanoi_moves(n)
    answer_bytes = []
    for k, src, dst in moves:
        answer_bytes += encode_move(k, src, dst)
    seq = [BOS] + inp_bytes + [SEP] + answer_bytes + [EOS]
    L = len(seq)
    sep_pos = len(inp_bytes) + 1
    trace_len = len(answer_bytes)

    # LoopCounter: counter at sep_pos+k = trace_len - k
    counter = [-1] * L
    for k in range(trace_len + 1):
        p = sep_pos + k
        if 0 <= p < L:
            counter[p] = trace_len - k

    # Per-position state channels: K-vector each
    raw = precompute_channels(n, K)  # list of K-lists
    channels = [[3] * K for _ in range(L)]  # default = sentinel
    for k, vec in enumerate(raw):
        p = sep_pos + k
        if 0 <= p < L:
            channels[p] = vec

    return {
        "tokens": torch.tensor(seq, dtype=torch.long),
        "sep_pos": sep_pos,
        "counter_values": torch.tensor(counter, dtype=torch.long),
        "state_channels": torch.tensor(channels, dtype=torch.long),  # (L, K)
        "n": n,
    }


def collate(examples, K):
    L = max(e["tokens"].shape[0] for e in examples)
    B = len(examples)
    out_tokens = torch.full((B, L), PAD, dtype=torch.long)
    out_counter = torch.full((B, L), -1, dtype=torch.long)
    out_channels = torch.full((B, L, K), 3, dtype=torch.long)  # sentinel default
    seps = []
    for i, e in enumerate(examples):
        l = e["tokens"].shape[0]
        out_tokens[i, :l] = e["tokens"]
        out_counter[i, :l] = e["counter_values"]
        out_channels[i, :l, :] = e["state_channels"]
        seps.append(e["sep_pos"])
    return {
        "tokens": out_tokens,
        "sep": torch.tensor(seps, dtype=torch.long),
        "counter_values": out_counter,
        "state_channels": out_channels,
    }


def loss_fn(out, batch):
    tokens = batch["tokens"]
    B, L = tokens.shape
    pos = torch.arange(L, device=tokens.device).unsqueeze(0)
    sep = batch["sep"].unsqueeze(1)
    ans_mask = ((pos >= sep) & (pos < L - 1)).float()
    pad_mask = (tokens != PAD).float()
    pred_mask = ans_mask[:, :L-1] * pad_mask[:, 1:]
    logits = out["token_logits"] if isinstance(out, dict) else out
    flat_logits = logits[:, :L-1].reshape(-1, logits.shape[-1])
    flat_targets = tokens[:, 1:].reshape(-1)
    flat_mask = pred_mask.reshape(-1)
    raw = F.cross_entropy(flat_logits, flat_targets, reduction="none")
    return (raw * flat_mask).sum() / (flat_mask.sum() + 1e-8)


def teacher_forced_eval(model, n, K, device):
    e = build_example(n, K)
    batch = collate([e], K)
    batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
    model.eval()
    with torch.no_grad():
        out = model(batch["tokens"],
                    counter_values=batch["counter_values"],
                    state_channels=batch["state_channels"])
    logits = out["token_logits"] if isinstance(out, dict) else out
    L = batch["tokens"].shape[1]
    sep = batch["sep"][0].item()
    correct = 0
    total = 0
    for p in range(sep, L - 1):
        target = batch["tokens"][0, p + 1].item()
        if target == PAD:
            continue
        am = logits[0, p].argmax().item()
        correct += (am == target)
        total += 1
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--K", type=int, default=10, help="max disks supported (channel count)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-cycles", type=int, default=30)
    ap.add_argument("--steps-per-cycle", type=int, default=30)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--curriculum", default="2,3,4,5,6,7,8")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--save-to", default="checkpoints/specialists/hanoi_step.pt")
    args = ap.parse_args()

    print(f"Device: {args.device}, K={args.K}, curriculum: {args.curriculum}", flush=True)
    ns = [int(s) for s in args.curriculum.split(",")]

    model = ProgressiveModel(
        d_model=args.d_model, d_state=16, expand=2, headdim=16,
        use_loop_counter=True, lc_iteration_token=None,
        use_state_feedback=True, sf_value_range=4,  # 0=A,1=B,2=C,3=none
    ).to(args.device)
    for _ in range(args.layers):
        model.add_kernel_layer()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    import random; random.seed(0)
    history = []
    for cycle in range(args.max_cycles):
        t0 = time.time()
        model.train()
        cum = 0.0
        for step in range(args.steps_per_cycle):
            examples = [build_example(random.choice(ns), args.K)
                        for _ in range(args.batch_size)]
            batch = collate(examples, args.K)
            batch = {k: (v.to(args.device) if hasattr(v, "to") else v)
                     for k, v in batch.items()}
            out = model(batch["tokens"],
                        counter_values=batch["counter_values"],
                        state_channels=batch["state_channels"])
            loss = loss_fn(out, batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            cum += loss.item()
        cum /= args.steps_per_cycle
        accs = {n: teacher_forced_eval(model, n, args.K, args.device) for n in ns}
        avg = sum(accs.values()) / len(accs)
        print(f"cycle {cycle+1:>3}  loss={cum:.3f}  acc={avg:.0%}  by_n={accs}  ({time.time()-t0:.1f}s)",
              flush=True)
        history.append({"cycle": cycle+1, "loss": cum, "acc": avg, "by_n": accs})
        Path(args.save_to).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model": model.state_dict(),
            "config": {"d_model": args.d_model, "n_kernel_layers": args.layers, "K": args.K},
            "accuracy": avg,
            "history": history,
        }, args.save_to)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
