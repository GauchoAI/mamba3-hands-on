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
# Total vocab: 64 (specials+letters) + 1024 (numbers up to 999) = 1088
VOCAB_SIZE = 1088


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
    """Evaluate exact match on a list of (inp, out, type) tuples."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for i in range(min(n_eval, len(examples))):
            inp, out, task_type = examples[i]
            tokens = torch.tensor([inp + out], dtype=torch.long, device=device)
            logits = model(tokens)

            sep = len(inp) - 1
            all_correct = True
            for t in range(sep, sep + len(out) - 1):
                if t + 1 >= tokens.shape[1]:
                    break
                pred = logits[0, t].argmax().item()
                target = tokens[0, t + 1].item()
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
    """Evaluate accuracy broken down by task type, on FRESH data."""
    fresh = make_fresh_eval_set(n_per_type * 8, seed=None)  # enough for all types
    model.eval()
    type_stats = {}

    with torch.no_grad():
        for inp, out, task_type in fresh:
            if task_type not in type_stats:
                type_stats[task_type] = {"correct": 0, "total": 0}
            if type_stats[task_type]["total"] >= n_per_type:
                continue

            tokens = torch.tensor([inp + out], dtype=torch.long, device=dataset.device)
            logits = model(tokens)

            sep = len(inp) - 1
            all_correct = True
            for t in range(sep, sep + len(out) - 1):
                if t + 1 >= tokens.shape[1]:
                    break
                pred = logits[0, t].argmax().item()
                target = tokens[0, t + 1].item()
                if target == SPECIAL_TOKENS["<PAD>"]:
                    break
                if pred != target:
                    all_correct = False
                    break

            type_stats[task_type]["total"] += 1
            if all_correct:
                type_stats[task_type]["correct"] += 1

    model.train()
    return {t: s["correct"] / max(s["total"], 1)
            for t, s in sorted(type_stats.items())}


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

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for step in range(1, args.steps + 1):
        tokens, sep_pos = dataset.get_batch(args.batch_size)
        logits = model(tokens)
        loss = compute_loss(logits, tokens, sep_pos)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % args.eval_every == 0 or step == 1:
            # Eval on BOTH training data and fresh data
            train_acc = evaluate_on_examples(model, dataset.examples, dataset.device, 300)
            fresh_acc = evaluate(model, dataset, n_eval=300)
            gap = train_acc - fresh_acc
            print(f"step {step:5d}  loss={loss.item():.3f}  "
                  f"train={train_acc:.1%}  fresh={fresh_acc:.1%}  "
                  f"gap={gap:+.1%}", flush=True)

            if step % (args.eval_every * 5) == 0:
                print("  --- per-type (fresh data) ---", flush=True)
                type_accs = evaluate_by_type(model, dataset)
                for t, a in type_accs.items():
                    print(f"    {t}: {a:.0%}", flush=True)

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
    print(f"  Gap: {train_acc - fresh_acc:+.1%}  (smaller = less memorization)", flush=True)
    print(f"  Threshold: {args.threshold:.0%}", flush=True)
    print(f"  {'PASS — ready for Level ' + str(args.level + 1) if all_pass else 'FAIL — needs more training'}",
          flush=True)

    # Save checkpoint
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / f"level{args.level}.pt"
    torch.save({
        "model": model.state_dict(),
        "cfg": cfg,
        "level": args.level,
        "step": args.steps,
        "accuracy": overall,
    }, ckpt_path)
    print(f"Saved checkpoint → {ckpt_path}", flush=True)


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
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--threshold", type=float, default=0.95)
    args = parser.parse_args()
    train(args)
