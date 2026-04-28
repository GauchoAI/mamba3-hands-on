"""train_hanoi_exec — train the small Mamba-3 + RegisterBank on the
Tower of Hanoi register-execution task.

The model sees an input "HANOI n", then has to autoregressively emit
the trace of moves while reading and writing a register bank that
tracks which disk is on which peg.

Per timestep, four heads are supervised:
  - token (260-vocab byte): the move-trace byte
  - read_addr (n_registers + 1): which register to read for next step
  - write_addr (n_registers + 1): which register to write this step
  - write_val (value_range): the value to write

Curriculum: n=2 -> 3 -> 5 -> 8. Same algorithm at every n; if the
model learns it at small n it should run correctly at larger n.

Usage:
  python train_hanoi_exec.py --max-cycles 50 --device mps
"""
import argparse, json, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from progressive_model import ProgressiveModel, ByteTokenizer, BOS, SEP, EOS, PAD
from hanoi_exec_oracle import gen_exec_trace, hanoi_moves


def build_example(n: int, n_registers: int, max_input_n: int = 99):
    """Build one full training example: input bytes + per-step trace.

    Returns a dict with token labels, per-step register IO ground truth,
    AND a LoopCounter trajectory (counter at SEP = total trace bytes,
    decrementing per position, 0 at EOS slot). The LoopCounter
    externalises termination from SSM-side n parsing.
    """
    NO_REG = n_registers
    inp_bytes = list(f"HANOI {n}".encode("utf-8"))
    trace, _ = gen_exec_trace(n, n_registers=n_registers)

    seq = [BOS] + inp_bytes + [SEP] + [r["token"] for r in trace] + [EOS]
    L = len(seq)
    sep_pos = len(inp_bytes) + 1
    trace_len = len(trace)

    # Build per-position labels. Indices align with positions in `seq`.
    # The input span (positions 0..sep_pos) has no ops; the answer
    # span starts at sep_pos+1. But the model PREDICTS the next
    # token from position p (i.e. at position p the model emits
    # what becomes seq[p+1]). So we align trace[k] with prediction
    # made AT position sep_pos + k.
    read_addr = [NO_REG] * L
    write_addr = [NO_REG] * L
    write_val = [0] * L
    read_input = [0] * L  # what value the model is "shown" at position p

    # Walk through the trace. Trace[k] describes the action the model
    # should produce when its prediction lands at position sep_pos + k
    # (i.e., emitting trace[k]['token'] as the next token after seq[sep_pos+k]).
    # The standard convention: at position p, model predicts seq[p+1]
    # and emits the heads for that prediction. So the head labels at
    # position p correspond to trace[(p - sep_pos)] when 0 <= p-sep_pos < len(trace).
    # The read_input at position p is what was read at the PREVIOUS step:
    # i.e., the register value at trace[k-1]['read_addr'] from the
    # snapshot trace[k-1]['registers_after'].
    for k, rec in enumerate(trace):
        p = sep_pos + k
        if p >= L:
            break
        read_addr[p] = rec["read_addr"]
        write_addr[p] = rec["write_addr"]
        write_val[p] = rec["write_val"]
        # read_input at the NEXT step is the result of THIS step's read
        if k + 1 < len(trace):
            ra = rec["read_addr"]
            if ra < NO_REG:
                # The register state right BEFORE the write of this step
                # is what gets read. Since registers_after reflects post-write,
                # we need the pre-step state. For our simple oracle, the
                # read happens at the start of the move and the write at
                # the end, so registers_after of the PREVIOUS record (or
                # the initial state) is what's read.
                if k == 0:
                    read_input[p + 1] = 0  # initial state, all zeros
                else:
                    read_input[p + 1] = trace[k - 1]["registers_after"][ra]
            # else: no-read, read_input stays 0

    # LoopCounter trajectory: at sep_pos+k the counter is (trace_len - k);
    # input span and post-EOS positions get sentinel (-1).
    counter_values = [-1] * L
    for k in range(trace_len + 1):  # 0 ... trace_len inclusive
        p = sep_pos + k
        if 0 <= p < L:
            counter_values[p] = trace_len - k

    return {
        "tokens": torch.tensor(seq, dtype=torch.long),
        "sep_pos": sep_pos,
        "read_addr": torch.tensor(read_addr, dtype=torch.long),
        "write_addr": torch.tensor(write_addr, dtype=torch.long),
        "write_val": torch.tensor(write_val, dtype=torch.long),
        "read_input": torch.tensor(read_input, dtype=torch.long),
        "counter_values": torch.tensor(counter_values, dtype=torch.long),
        "n": n,
    }


def collate(examples, pad_token=PAD):
    """Pad list of examples to the same length."""
    L = max(e["tokens"].shape[0] for e in examples)
    B = len(examples)
    out_tokens = torch.full((B, L), pad_token, dtype=torch.long)
    out_read_addr = torch.zeros(B, L, dtype=torch.long)
    out_write_addr = torch.zeros(B, L, dtype=torch.long)
    out_write_val = torch.zeros(B, L, dtype=torch.long)
    out_read_input = torch.zeros(B, L, dtype=torch.long)
    out_counter = torch.full((B, L), -1, dtype=torch.long)  # sentinel default
    seps = []
    for i, e in enumerate(examples):
        l = e["tokens"].shape[0]
        out_tokens[i, :l] = e["tokens"]
        out_read_addr[i, :l] = e["read_addr"]
        out_write_addr[i, :l] = e["write_addr"]
        out_write_val[i, :l] = e["write_val"]
        out_read_input[i, :l] = e["read_input"]
        out_counter[i, :l] = e["counter_values"]
        seps.append(e["sep_pos"])
    return {
        "tokens": out_tokens,
        "sep": torch.tensor(seps, dtype=torch.long),
        "read_addr": out_read_addr,
        "write_addr": out_write_addr,
        "write_val": out_write_val,
        "read_input": out_read_input,
        "counter_values": out_counter,
    }


def loss_fn(out, batch, n_registers, alpha=1.0, beta=1.0, gamma=1.0):
    """Compute multi-head CE losses, masked to the answer span."""
    tokens = batch["tokens"]
    B, L = tokens.shape
    pos = torch.arange(L, device=tokens.device).unsqueeze(0)
    sep = batch["sep"].unsqueeze(1)
    # Answer span: positions sep+0 ... L-2 (last position predicts EOS).
    ans_mask = ((pos >= sep) & (pos < L - 1)).float()
    pad_mask = (tokens != PAD).float()
    pred_mask = ans_mask[:, :L-1] * pad_mask[:, 1:]  # supervise position p iff token at p+1 is not PAD

    # Token CE: predict tokens[:,1:] from out['token_logits'][:,:-1]
    token_loss = F.cross_entropy(
        out["token_logits"][:, :L-1].reshape(-1, out["token_logits"].shape[-1]),
        tokens[:, 1:].reshape(-1),
        reduction="none",
    ).reshape(B, L-1)
    token_loss = (token_loss * pred_mask).sum() / (pred_mask.sum() + 1e-8)

    # Read/write head CE: target labels are read_addr, write_addr, write_val
    # at position p (the action emitted from position p when predicting p+1).
    # We supervise at positions in the answer span (where ans_mask is 1).
    head_mask = ans_mask  # (B, L)

    def head_ce(logits, labels):
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_labels = labels.reshape(-1)
        loss = F.cross_entropy(flat_logits, flat_labels, reduction="none").reshape(B, L)
        return (loss * head_mask).sum() / (head_mask.sum() + 1e-8)

    read_loss = head_ce(out["read_logits"], batch["read_addr"])
    write_loss = head_ce(out["write_logits"], batch["write_addr"])
    val_loss = head_ce(out["val_logits"], batch["write_val"])

    total = token_loss + alpha * read_loss + beta * write_loss + gamma * val_loss
    return total, {
        "token": token_loss.item(), "read": read_loss.item(),
        "write": write_loss.item(), "val": val_loss.item(),
    }


def teacher_forced_eval(model, n, n_registers, device):
    """Single-example teacher-forced check: does the model predict the
    correct token at every answer position when fed ground truth?"""
    e = build_example(n, n_registers)
    batch = collate([e])
    batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
    model.eval()
    with torch.no_grad():
        out = model(batch["tokens"],
                    counter_values=batch["counter_values"],
                    register_read_values=batch["read_input"])
    L = batch["tokens"].shape[1]
    sep = batch["sep"][0].item()
    ans_pos = list(range(sep, L - 1))
    correct_token = correct_read = correct_write = correct_val = 0
    total = 0
    for p in ans_pos:
        target_tok = batch["tokens"][0, p + 1].item()
        if target_tok == PAD:
            continue
        am_tok = out["token_logits"][0, p].argmax().item()
        am_read = out["read_logits"][0, p].argmax().item()
        am_write = out["write_logits"][0, p].argmax().item()
        am_val = out["val_logits"][0, p].argmax().item()
        correct_token += (am_tok == target_tok)
        correct_read += (am_read == batch["read_addr"][0, p].item())
        correct_write += (am_write == batch["write_addr"][0, p].item())
        correct_val += (am_val == batch["write_val"][0, p].item())
        total += 1
    return {
        "token_acc": correct_token / max(total, 1),
        "read_acc": correct_read / max(total, 1),
        "write_acc": correct_write / max(total, 1),
        "val_acc": correct_val / max(total, 1),
        "total": total,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--n-registers", type=int, default=16)
    ap.add_argument("--value-range", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-cycles", type=int, default=50)
    ap.add_argument("--steps-per-cycle", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--curriculum", default="2,3,5,8",
                    help="Comma list of n to draw from (uniform sampling)")
    ap.add_argument("--save-to", default="checkpoints/specialists/hanoi_exec.pt")
    args = ap.parse_args()

    print(f"Device: {args.device}")
    print(f"Curriculum n: {args.curriculum}")
    ns = [int(s) for s in args.curriculum.split(",")]

    # Build model with BOTH primitives:
    # - RegisterBank: tracks per-disk peg state, decides reads/writes
    # - LoopCounter:  oracle-supplied "moves remaining" counter,
    #                 drives termination (EOS at counter=0). No iter_token
    #                 (we set lc_iteration_token=None) so the LoopCounter
    #                 only contributes the EOS-gating signal.
    model = ProgressiveModel(
        d_model=args.d_model, d_state=16, expand=2, headdim=16,
        use_register_bank=True,
        reg_n_registers=args.n_registers,
        reg_value_range=args.value_range,
        use_loop_counter=True,
        lc_iteration_token=None,  # disable iter_bias path
    ).to(args.device)
    for _ in range(args.layers):
        model.add_kernel_layer()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    import random
    random.seed(0)
    history = []
    for cycle in range(args.max_cycles):
        t0 = time.time()
        model.train()
        cycle_loss = 0.0
        last_components = {}
        for step in range(args.steps_per_cycle):
            # Build a batch: sample n from curriculum, build example
            examples = [build_example(random.choice(ns), args.n_registers)
                        for _ in range(args.batch_size)]
            batch = collate(examples)
            batch = {k: (v.to(args.device) if hasattr(v, "to") else v) for k, v in batch.items()}

            out = model(batch["tokens"],
                        counter_values=batch["counter_values"],
                        register_read_values=batch["read_input"])
            loss, components = loss_fn(out, batch, args.n_registers)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            cycle_loss += loss.item()
            last_components = components
        cycle_loss /= args.steps_per_cycle

        # Eval: teacher-forced accuracy on each n
        eval_metrics = {n: teacher_forced_eval(model, n, args.n_registers, args.device)
                        for n in ns + [max(ns) + 2]}  # also test 1 OOD

        elapsed = time.time() - t0
        avg_token_acc = sum(m["token_acc"] for m in eval_metrics.values()) / len(eval_metrics)
        avg_read_acc = sum(m["read_acc"] for m in eval_metrics.values()) / len(eval_metrics)
        avg_write_acc = sum(m["write_acc"] for m in eval_metrics.values()) / len(eval_metrics)
        avg_val_acc = sum(m["val_acc"] for m in eval_metrics.values()) / len(eval_metrics)
        msg = (f"cycle {cycle+1:>3}  loss={cycle_loss:.3f}  "
               f"tok={avg_token_acc:.0%}  read={avg_read_acc:.0%}  "
               f"write={avg_write_acc:.0%}  val={avg_val_acc:.0%}  "
               f"({elapsed:.1f}s)")
        print(msg, flush=True)
        history.append({
            "cycle": cycle + 1, "loss": cycle_loss,
            "token_acc": avg_token_acc, "read_acc": avg_read_acc,
            "write_acc": avg_write_acc, "val_acc": avg_val_acc,
            "components": last_components,
            "by_n": eval_metrics,
        })

        # Save checkpoint
        Path(args.save_to).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model": model.state_dict(),
            "config": {
                "d_model": args.d_model, "n_kernel_layers": args.layers,
                "n_registers": args.n_registers, "value_range": args.value_range,
            },
            "accuracy": avg_token_acc,
            "history": history,
        }, args.save_to)

    print("Training complete.")
    print(json.dumps({"final_history": history[-1]}, indent=2, default=str))


if __name__ == "__main__":
    main()
