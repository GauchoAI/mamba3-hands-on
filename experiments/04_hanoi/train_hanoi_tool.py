"""train_hanoi_tool — train Mamba-3 to emit Hanoi traces with
state-feedback supplied by an external Python tool.

Architecture (no learned register bank):
  - Token bytes are embedded as usual.
  - Per-position state feedback (a single int packed from the
    full peg state) is computed by HanoiTool in plain Python.
  - The feedback int is mapped through a value embedding and
    ADDED to the input token embedding. This is the model's
    only state input.
  - LoopCounter externalises termination (counter at SEP =
    total trace bytes, decrementing per output position).
  - LM head emits the next byte.

There are NO register read/write heads: state is fully maintained
by the Python tool, exposed to the model as embeddings. The number
of "slots" the tool tracks is a Python concern; the model's
parameters are independent of it.
"""
import argparse, json, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from progressive_model import ProgressiveModel, ByteTokenizer, BOS, SEP, EOS, PAD
from hanoi_tool import HanoiTool, hanoi_moves, precompute_feedback, PEG_NAMES


def encode_move(k: int, src: int, dst: int):
    return list(f"{k} {PEG_NAMES[src]} {PEG_NAMES[dst]}\n".encode("ascii"))


def build_example(n: int, value_range: int):
    """Construct one full training example: tokens, sep_pos, counter
    trajectory, per-position state-feedback values."""
    inp_bytes = list(f"HANOI {n}".encode("utf-8"))
    moves = hanoi_moves(n)
    answer_bytes = []
    for k, src, dst in moves:
        answer_bytes += encode_move(k, src, dst)
    seq = [BOS] + inp_bytes + [SEP] + answer_bytes + [EOS]
    L = len(seq)
    sep_pos = len(inp_bytes) + 1
    trace_len = len(answer_bytes)

    # LoopCounter trajectory: counter at sep_pos+k = trace_len - k
    counter_values = [-1] * L
    for k in range(trace_len + 1):
        p = sep_pos + k
        if 0 <= p < L:
            counter_values[p] = trace_len - k

    # State feedback: tool emits (n_moves * 6) values via precompute.
    # In our convention: at sep_pos + k the model PREDICTS the byte
    # at sep_pos + k + 1, and the feedback at sep_pos + k reflects
    # state AFTER byte at sep_pos + k - 1 is emitted (or initial at k=0).
    fb = precompute_feedback(n)  # length = trace_len
    # Clamp to value_range to avoid embedding-table overflow.
    fb = [min(max(0, v), value_range - 1) for v in fb]
    state_feedback = [0] * L
    for k in range(trace_len):
        p = sep_pos + k
        if 0 <= p < L:
            state_feedback[p] = fb[k]

    return {
        "tokens": torch.tensor(seq, dtype=torch.long),
        "sep_pos": sep_pos,
        "counter_values": torch.tensor(counter_values, dtype=torch.long),
        "state_feedback": torch.tensor(state_feedback, dtype=torch.long),
        "n": n,
    }


def collate(examples):
    L = max(e["tokens"].shape[0] for e in examples)
    B = len(examples)
    out_tokens = torch.full((B, L), PAD, dtype=torch.long)
    out_counter = torch.full((B, L), -1, dtype=torch.long)
    out_fb = torch.zeros(B, L, dtype=torch.long)
    seps = []
    for i, e in enumerate(examples):
        l = e["tokens"].shape[0]
        out_tokens[i, :l] = e["tokens"]
        out_counter[i, :l] = e["counter_values"]
        out_fb[i, :l] = e["state_feedback"]
        seps.append(e["sep_pos"])
    return {
        "tokens": out_tokens,
        "sep": torch.tensor(seps, dtype=torch.long),
        "counter_values": out_counter,
        "state_feedback": out_fb,
    }


def loss_fn(out, batch):
    """Token CE only (no register heads)."""
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


def teacher_forced_eval(model, n, value_range, device):
    e = build_example(n, value_range)
    batch = collate([e])
    batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
    model.eval()
    with torch.no_grad():
        out = model(batch["tokens"],
                    counter_values=batch["counter_values"],
                    register_read_values=batch["state_feedback"])
    logits = out["token_logits"] if isinstance(out, dict) else out
    L = batch["tokens"].shape[1]
    sep = batch["sep"][0].item()
    ans_pos = list(range(sep, L - 1))
    correct = 0
    total = 0
    for p in ans_pos:
        target = batch["tokens"][0, p + 1].item()
        if target == PAD:
            continue
        am = logits[0, p].argmax().item()
        correct += (am == target)
        total += 1
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-model", type=int, default=32)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--value-range", type=int, default=256,  # supports n<=5
                    help="Embedding-table size for state-feedback values")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-cycles", type=int, default=25)
    ap.add_argument("--steps-per-cycle", type=int, default=30)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--curriculum", default="2,3,4")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--save-to", default="checkpoints/specialists/hanoi_tool.pt")
    args = ap.parse_args()

    print(f"Device: {args.device}, value_range: {args.value_range}, "
          f"curriculum: {args.curriculum}", flush=True)
    ns = [int(s) for s in args.curriculum.split(",")]

    # Use the existing RegisterBank as the value-embedding host; we
    # supply register_read_values from the tool but DON'T supervise
    # any register heads. The heads exist (small overhead) but their
    # gradient comes only from indirect paths.
    model = ProgressiveModel(
        d_model=args.d_model, d_state=16, expand=2, headdim=16,
        use_register_bank=True,
        reg_n_registers=1,                # minimal — heads are unused
        reg_value_range=args.value_range,
        use_loop_counter=True,
        lc_iteration_token=None,
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
            examples = [build_example(random.choice(ns), args.value_range)
                        for _ in range(args.batch_size)]
            batch = collate(examples)
            batch = {k: (v.to(args.device) if hasattr(v, "to") else v)
                     for k, v in batch.items()}
            out = model(batch["tokens"],
                        counter_values=batch["counter_values"],
                        register_read_values=batch["state_feedback"])
            loss = loss_fn(out, batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            cum += loss.item()
        cum /= args.steps_per_cycle
        accs = {n: teacher_forced_eval(model, n, args.value_range, args.device) for n in ns}
        avg_acc = sum(accs.values()) / len(accs)
        print(f"cycle {cycle+1:>3}  loss={cum:.3f}  acc={avg_acc:.0%}  by_n={accs}  ({time.time()-t0:.1f}s)",
              flush=True)
        history.append({"cycle": cycle + 1, "loss": cum, "acc": avg_acc, "by_n": accs})

        Path(args.save_to).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model": model.state_dict(),
            "config": {"d_model": args.d_model, "n_kernel_layers": args.layers,
                       "value_range": args.value_range},
            "accuracy": avg_acc,
            "history": history,
        }, args.save_to)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
