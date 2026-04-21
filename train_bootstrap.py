"""
Bootstrap training harness.

Trains one level at a time, validating before advancing.
Checkpoints are saved per level so the next level starts from
the previous level's trained weights.

Usage:
    python train_bootstrap.py --level 0 --data data/level0/patterns.jsonl
    python train_bootstrap.py --level 1 --data data/level1/ --resume checkpoints/level0.pt
"""
import os
os.environ["PYTHONUNBUFFERED"] = "1"
import sys
sys.path.insert(0, os.path.dirname(__file__))

import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from mamba3_minimal import Mamba3Block, Mamba3Config


# ── Tokenization ─────────────────────────────────────────────────────
# Simple char-level tokenizer for the bootstrap data.
# Tokens: digits 0-9, letters A-Z, special tokens (SAME, DIFF, etc.)

SPECIAL_TOKENS = {
    "<PAD>": 0, "<SEP>": 1, "<BOS>": 2, "<EOS>": 3,
    "?": 4, "SAME": 5, "DIFF": 6, "MIRROR": 7, "NO": 8,
    "COUNT": 9, "PERIOD": 10, " ": 11,
}
# Numbers 0-999 get tokens 64+
NUM_OFFSET = 64
# Total vocab: NUM_OFFSET(64) + 64(negative shift) + 1024(numbers 0-999) = 1152
VOCAB_SIZE = 1152


def tokenize(text):
    """Tokenize a string into token IDs."""
    tokens = []
    words = text.split(" ")
    for w in words:
        if w in SPECIAL_TOKENS:
            tokens.append(SPECIAL_TOKENS[w])
        elif w.lstrip("-").isdigit():
            n = int(w)
            if -64 < n < 1024:
                tokens.append(NUM_OFFSET + n + 64)  # shift so -64 maps to 0+NUM_OFFSET
            else:
                # Fallback: tokenize digit by digit
                for ch in w:
                    tokens.append(ord(ch) - ord("0") + 12 if ch.isdigit() else 11)
        else:
            # Unknown word — tokenize char by char
            for ch in w:
                if ch.isalpha():
                    tokens.append(ord(ch.upper()) - ord("A") + 32)
                elif ch.isdigit():
                    tokens.append(ord(ch) - ord("0") + 12)
                else:
                    tokens.append(11)  # space
        tokens.append(11)  # space between words
    if tokens and tokens[-1] == 11:
        tokens.pop()  # remove trailing space
    return tokens


def detokenize(token_ids):
    """Convert token IDs back to string (best effort)."""
    inv_special = {v: k for k, v in SPECIAL_TOKENS.items()}
    parts = []
    for t in token_ids:
        if t in inv_special:
            parts.append(inv_special[t])
        elif t >= NUM_OFFSET:
            parts.append(str(t - NUM_OFFSET - 64))
        elif 12 <= t < 22:
            parts.append(str(t - 12))
        elif 32 <= t < 58:
            parts.append(chr(t - 32 + ord("A")))
        else:
            parts.append("?")
    return " ".join(parts)


# ── Dataset ──────────────────────────────────────────────────────────

class BootstrapDataset:
    def __init__(self, path, device="cpu"):
        self.examples = []
        path = Path(path)
        files = [path] if path.is_file() else sorted(path.glob("*.jsonl"))

        for f in files:
            with open(f) as fh:
                for line in fh:
                    ex = json.loads(line.strip())
                    inp_tokens = [SPECIAL_TOKENS["<BOS>"]] + tokenize(ex["input"]) + [SPECIAL_TOKENS["<SEP>"]]
                    out_tokens = tokenize(ex["output"]) + [SPECIAL_TOKENS["<EOS>"]]
                    self.examples.append((inp_tokens, out_tokens, ex.get("type", "")))

        self.device = device
        print(f"BootstrapDataset: {len(self.examples)} examples from {path}", flush=True)

    def get_batch(self, batch_size):
        """Random batch with padding. Returns (tokens, sep_positions) for loss masking."""
        indices = torch.randint(0, len(self.examples), (batch_size,))
        max_len = 0
        batch_inp, batch_out = [], []

        for idx in indices:
            inp, out, _ = self.examples[idx.item()]
            batch_inp.append(inp)
            batch_out.append(out)
            max_len = max(max_len, len(inp) + len(out))

        # Build token sequences: [BOS input SEP output EOS PAD...]
        tokens = torch.full((batch_size, max_len), SPECIAL_TOKENS["<PAD>"],
                           dtype=torch.long, device=self.device)
        sep_positions = []

        for i, (inp, out) in enumerate(zip(batch_inp, batch_out)):
            seq = inp + out
            tokens[i, :len(seq)] = torch.tensor(seq)
            sep_positions.append(len(inp) - 1)  # SEP is last token of inp

        return tokens, sep_positions


# ── Model ────────────────────────────────────────────────────────────

class BootstrapModel(nn.Module):
    def __init__(self, cfg, vocab_size=VOCAB_SIZE):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, cfg.d_model)
        self.block = Mamba3Block(cfg)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, vocab_size)

    def forward(self, tokens):
        x = self.embed(tokens)
        x = self.block(x)
        x = self.norm(x)
        return self.head(x)


# ── Training ─────────────────────────────────────────────────────────

def compute_loss(logits, tokens, sep_positions):
    """Next-token loss only on positions AFTER SEP (the output portion)."""
    B, L, V = logits.shape
    loss = torch.tensor(0.0, device=logits.device)
    count = 0

    for b in range(B):
        sep = sep_positions[b]
        for t in range(sep, L - 1):
            target = tokens[b, t + 1]
            if target == SPECIAL_TOKENS["<PAD>"]:
                break
            loss = loss + F.cross_entropy(logits[b, t], target)
            count += 1

    return loss / max(count, 1)


def evaluate_on_examples(model, examples, device, n_eval=500):
    """Evaluate exact match on a list of (inp, out, type) tuples. BATCHED."""
    model.eval()
    correct = 0
    total = 0
    n = min(n_eval, len(examples))
    batch_size = 64

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_examples = examples[start:end]

            # Pad to same length
            seqs = []
            sep_positions = []
            out_lengths = []
            for inp, out, _ in batch_examples:
                seqs.append(inp + out)
                sep_positions.append(len(inp) - 1)
                out_lengths.append(len(out))
            max_len = max(len(s) for s in seqs)
            tokens = torch.full((len(seqs), max_len), SPECIAL_TOKENS["<PAD>"],
                               dtype=torch.long, device=device)
            for i, seq in enumerate(seqs):
                tokens[i, :len(seq)] = torch.tensor(seq)

            logits = model(tokens)  # (B, L, V) — one forward pass for whole batch

            for i in range(len(batch_examples)):
                sep = sep_positions[i]
                _, out, _ = batch_examples[i]
                all_correct = True
                for t in range(sep, sep + len(out) - 1):
                    if t + 1 >= tokens.shape[1]:
                        break
                    pred = logits[i, t].argmax().item()
                    target = tokens[i, t + 1].item()
                    if target == SPECIAL_TOKENS["<PAD>"]:
                        break
                    if pred != target:
                        all_correct = False
                        break
                if all_correct:
                    correct += 1
                total += 1

    model.train()
    return correct / max(total, 1)


def make_fresh_eval_set(count=500, seed=None):
    """Generate fresh examples never seen during training."""
    import generators.level0_patterns as gen0
    if seed is not None:
        import random as _r
        _r.seed(seed)
    raw = gen0.generate_dataset(count)
    examples = []
    for ex in raw:
        inp_tokens = [SPECIAL_TOKENS["<BOS>"]] + tokenize(ex["input"]) + [SPECIAL_TOKENS["<SEP>"]]
        out_tokens = tokenize(ex["output"]) + [SPECIAL_TOKENS["<EOS>"]]
        examples.append((inp_tokens, out_tokens, ex.get("type", "")))
    return examples


def evaluate(model, dataset, n_eval=500):
    """Evaluate on FRESH data (generalization), not training data."""
    fresh = make_fresh_eval_set(n_eval, seed=None)  # different every call
    return evaluate_on_examples(model, fresh, dataset.device, n_eval)


def evaluate_by_type(model, dataset, n_per_type=100):
    """Evaluate accuracy broken down by task type, on FRESH data. Uses batched eval."""
    fresh = make_fresh_eval_set(n_per_type * 8, seed=None)

    # Group by type
    by_type = {}
    for inp, out, task_type in fresh:
        if task_type not in by_type:
            by_type[task_type] = []
        if len(by_type[task_type]) < n_per_type:
            by_type[task_type].append((inp, out, task_type))

    results = {}
    for task_type, examples in sorted(by_type.items()):
        acc = evaluate_on_examples(model, examples, dataset.device, len(examples))
        results[task_type] = acc

    return results


def train(args):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    print(f"Level: {args.level}", flush=True)

    dataset = BootstrapDataset(args.data, device)

    cfg = Mamba3Config(
        d_model=args.d_model,
        d_state=args.d_state,
        expand=2,
        headdim=args.headdim,
    )
    model = BootstrapModel(cfg).to(device)

    if args.resume:
        print(f"Resuming from {args.resume}", flush=True)
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params", flush=True)

    base_lr = args.lr
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr)
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)

    # Track best fresh accuracy — carry forward from checkpoint if resuming
    best_fresh = 0.0
    if args.resume:
        best_fresh = ckpt.get("best_fresh", ckpt.get("fresh_acc", 0.0))
        print(f"Carrying forward best_fresh={best_fresh:.1%}", flush=True)
    current_lr = base_lr
    global_step = 0

    # Adaptive teacher
    from generators.teacher import AdaptiveTeacher
    teacher = AdaptiveTeacher()

    # Cycle parameters
    learn_steps = args.cycle_learn     # steps at high LR (learning)
    digest_steps = args.cycle_digest   # steps at low LR (digesting)
    cycle_len = learn_steps + digest_steps
    gap_throttle = 0.10                # if gap > this, slow down

    print(f"Cycles: {learn_steps} learn + {digest_steps} digest = {cycle_len} per cycle",
          flush=True)
    print(f"Gap throttle: >{gap_throttle:.0%} → reduce LR", flush=True)
    print(f"Target: {args.steps} total steps, ~{args.steps // cycle_len} cycles", flush=True)
    print(f"Adaptive teacher: ON", flush=True)
    print(flush=True)

    while global_step < args.steps:
        # ── Regenerate training data each cycle via teacher ──
        cycle_num = global_step // cycle_len + 1
        print(f"=== Cycle {cycle_num} — teacher generating adaptive data ===", flush=True)
        raw = teacher.generate(len(dataset.examples))
        dataset.examples = []
        for ex in raw:
            inp_tokens = [SPECIAL_TOKENS["<BOS>"]] + tokenize(ex["input"]) + [SPECIAL_TOKENS["<SEP>"]]
            out_tokens = tokenize(ex["output"]) + [SPECIAL_TOKENS["<EOS>"]]
            dataset.examples.append((inp_tokens, out_tokens, ex.get("type", "")))

        # ── Learning phase (high LR) ──
        for param_group in opt.param_groups:
            param_group["lr"] = current_lr
        phase = "LEARN"

        for step_in_cycle in range(cycle_len):
            global_step += 1
            if global_step > args.steps:
                break

            # Switch to digest phase
            if step_in_cycle == learn_steps:
                phase = "DIGEST"
                digest_lr = current_lr * 0.1
                for param_group in opt.param_groups:
                    param_group["lr"] = digest_lr

            tokens, sep_pos = dataset.get_batch(args.batch_size)
            logits = model(tokens)
            loss = compute_loss(logits, tokens, sep_pos)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if global_step % args.eval_every == 0:
                train_acc = evaluate_on_examples(model, dataset.examples, dataset.device, 500)
                fresh_acc = evaluate(model, dataset, n_eval=500)
                gap = train_acc - fresh_acc
                lr_now = opt.param_groups[0]["lr"]

                print(f"step {global_step:5d}  loss={loss.item():.3f}  "
                      f"train={train_acc:.1%}  fresh={fresh_acc:.1%}  "
                      f"gap={gap:+.1%}  lr={lr_now:.1e}  [{phase}]", flush=True)

                # Gap-based throttle: if memorizing too much, slow down
                if gap > gap_throttle and phase == "LEARN":
                    current_lr = max(current_lr * 0.7, base_lr * 0.01)
                    for param_group in opt.param_groups:
                        param_group["lr"] = current_lr
                    print(f"  ⚠ gap>{gap_throttle:.0%}, throttling LR → {current_lr:.1e}",
                          flush=True)
                elif gap < 0.05 and current_lr < base_lr:
                    # Gap is healthy, recover LR
                    current_lr = min(current_lr * 1.2, base_lr)
                    if phase == "LEARN":
                        for param_group in opt.param_groups:
                            param_group["lr"] = current_lr

                # Save immutable checkpoint (never overwritten)
                ckpt_data = {
                    "model": model.state_dict(),
                    "cfg": cfg,
                    "level": args.level,
                    "step": global_step,
                    "train_acc": train_acc,
                    "fresh_acc": fresh_acc,
                    "best_fresh": best_fresh,
                }
                torch.save(ckpt_data, ckpt_dir / f"level{args.level}_step{global_step}.pt")

                # Also save as "latest" pointer
                torch.save(ckpt_data, ckpt_dir / f"level{args.level}_latest.pt")

                if fresh_acc > best_fresh:
                    best_fresh = fresh_acc
                    ckpt_data["best_fresh"] = best_fresh
                    torch.save(ckpt_data, ckpt_dir / f"level{args.level}_best.pt")
                    print(f"  ★ new best fresh={fresh_acc:.1%}", flush=True)

                if global_step % (args.eval_every * 4) == 0:
                    print("  --- per-type (fresh data) ---", flush=True)
                    type_accs = evaluate_by_type(model, dataset)
                    for t, a in type_accs.items():
                        print(f"    {t}: {a:.0%}", flush=True)
                    # Feed back to teacher
                    teacher.observe(type_accs)
                    print("  --- teacher status ---", flush=True)
                    print(teacher.get_status(), flush=True)

    # Final evaluation
    print("\n--- Final evaluation (fresh data) ---", flush=True)
    type_accs = evaluate_by_type(model, dataset, n_per_type=200)
    all_pass = True
    for t, a in type_accs.items():
        status = "✓" if a >= args.threshold else "✗"
        print(f"  {status} {t}: {a:.0%}", flush=True)
        if a < args.threshold:
            all_pass = False

    train_acc = evaluate_on_examples(model, dataset.examples, dataset.device, 1000)
    fresh_acc = evaluate(model, dataset, n_eval=1000)
    print(f"\n  Train exact match: {train_acc:.1%}", flush=True)
    print(f"  Fresh exact match: {fresh_acc:.1%}  ← this is the real score", flush=True)
    print(f"  Best fresh ever:   {best_fresh:.1%}", flush=True)
    print(f"  Gap: {train_acc - fresh_acc:+.1%}", flush=True)
    print(f"  Threshold: {args.threshold:.0%}", flush=True)
    print(f"  {'PASS — ready for Level ' + str(args.level + 1) if all_pass else 'FAIL — needs more training'}",
          flush=True)

    # Save final
    torch.save({
        "model": model.state_dict(),
        "cfg": cfg,
        "level": args.level,
        "step": global_step,
        "train_acc": train_acc,
        "fresh_acc": fresh_acc,
    }, ckpt_dir / f"level{args.level}.pt")
    print(f"Saved checkpoint → {ckpt_dir / f'level{args.level}.pt'}", flush=True)
    print(f"Best checkpoint  → {ckpt_dir / f'level{args.level}_best.pt'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--resume", default=None, help="Checkpoint to resume from")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--headdim", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--cycle-learn", type=int, default=500,
                        help="Steps per cycle at high LR (learning phase)")
    parser.add_argument("--cycle-digest", type=int, default=200,
                        help="Steps per cycle at low LR (digest phase)")
    args = parser.parse_args()
    train(args)
