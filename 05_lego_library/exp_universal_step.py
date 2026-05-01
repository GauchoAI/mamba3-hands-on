"""
Experiment: Universal Step Function.

One Mamba-3 block, one head, no task labels. The model must recognize
from the input pattern itself what computation to perform.

Format: every sequence has the same structure:
  [input tokens...] [SEP] [output tokens...]

The model is trained to predict the output tokens given the input + SEP.
No task indicator. The step function must learn to be general.

Tasks mixed together:
  1. PARITY:   0 1 1 0 1 | 1          (XOR of inputs)
  2. SORT:     5 2 8 1 | 1 2 5 8      (sorted version)
  3. REVERSE:  3 1 4 | 4 1 3          (reversed)
  4. MIN/MAX:  7 2 9 3 | 2 9          (min then max)
  5. LENGTH:   a b c d e | 5          (count of elements)

The SEP token and the output pattern ARE the only signal telling the
model what to do. The step function must figure it out from context.

Token encoding:
  0-15:  data values
  16:    SEP token
  17:    PAD token
"""
import os
os.environ["PYTHONUNBUFFERED"] = "1"
import sys
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba3_minimal import Mamba3Block, Mamba3Config

VOCAB = 16   # data values 0..15
SEP = 16
PAD = 17
TOTAL_VOCAB = 18


def make_parity_seq(L, device):
    bits = torch.randint(0, 2, (L,), device=device)
    parity = bits.sum() % 2
    return torch.cat([bits, torch.tensor([SEP, parity], device=device)])


def make_sort_seq(L, max_val, device):
    nums = torch.randint(0, max_val, (L,), device=device)
    sorted_nums, _ = torch.sort(nums)
    return torch.cat([nums, torch.tensor([SEP], device=device), sorted_nums])


def make_reverse_seq(L, max_val, device):
    nums = torch.randint(0, max_val, (L,), device=device)
    return torch.cat([nums, torch.tensor([SEP], device=device), nums.flip(0)])


def make_minmax_seq(L, max_val, device):
    nums = torch.randint(0, max_val, (L,), device=device)
    mn, mx = nums.min(), nums.max()
    return torch.cat([nums, torch.tensor([SEP, mn, mx], device=device)])


def make_length_seq(L, max_val, device):
    nums = torch.randint(0, max_val, (L,), device=device)
    return torch.cat([nums, torch.tensor([SEP, L], device=device)])


TASK_FNS = [make_parity_seq, make_sort_seq, make_reverse_seq,
            make_minmax_seq, make_length_seq]
TASK_NAMES = ["parity", "sort", "reverse", "minmax", "length"]


def make_batch(B, L_range=(4, 10), max_val=10, device="cpu"):
    """Generate a mixed batch. Each sequence is a random task with random length."""
    seqs = []
    tasks = []
    max_len = 0

    for _ in range(B):
        L = torch.randint(L_range[0], L_range[1] + 1, (1,)).item()
        task_idx = torch.randint(0, len(TASK_FNS), (1,)).item()
        fn = TASK_FNS[task_idx]

        if task_idx == 0:  # parity only uses 0/1
            seq = fn(L, device)
        else:
            seq = fn(L, max_val, device)

        seqs.append(seq)
        tasks.append(task_idx)
        max_len = max(max_len, len(seq))

    # Pad to same length
    padded = torch.full((B, max_len), PAD, dtype=torch.long, device=device)
    for i, seq in enumerate(seqs):
        padded[i, :len(seq)] = seq

    return padded, tasks


class UniversalStepModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed = nn.Embedding(TOTAL_VOCAB, cfg.d_model)
        self.block = Mamba3Block(cfg)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, TOTAL_VOCAB)

    def forward(self, tokens):
        x = self.embed(tokens)
        x = self.block(x)
        x = self.norm(x)
        return self.head(x)  # (B, L, vocab)


def compute_loss(logits, tokens):
    """Next-token prediction loss, but only on positions AFTER SEP."""
    B, L, V = logits.shape
    loss = 0
    count = 0
    for b in range(B):
        # Find SEP position
        sep_positions = (tokens[b] == SEP).nonzero(as_tuple=True)[0]
        if len(sep_positions) == 0:
            continue
        sep_pos = sep_positions[0].item()

        # Predict tokens after SEP (from SEP position onward, predict next)
        for t in range(sep_pos, L - 1):
            if tokens[b, t + 1] == PAD:
                break
            loss += F.cross_entropy(logits[b, t], tokens[b, t + 1])
            count += 1

    return loss / max(count, 1)


def evaluate_per_task(model, device, n_eval=200, L_range=(4, 10), max_val=10):
    """Evaluate accuracy per task."""
    model.eval()
    results = {}

    for task_idx, name in enumerate(TASK_NAMES):
        correct = 0
        total = 0

        with torch.no_grad():
            for _ in range(n_eval):
                L = torch.randint(L_range[0], L_range[1] + 1, (1,)).item()
                fn = TASK_FNS[task_idx]
                if task_idx == 0:
                    seq = fn(L, device)
                else:
                    seq = fn(L, max_val, device)

                tokens = seq.unsqueeze(0)  # (1, seq_len)
                logits = model(tokens)     # (1, seq_len, vocab)

                # Find SEP and check predictions after it
                sep_pos = (seq == SEP).nonzero(as_tuple=True)[0][0].item()
                for t in range(sep_pos, len(seq) - 1):
                    pred = logits[0, t].argmax().item()
                    target = seq[t + 1].item()
                    if target == PAD:
                        break
                    total += 1
                    if pred == target:
                        correct += 1

        results[name] = correct / max(total, 1)

    model.train()
    return results


def train(cfg, device, steps=8000, batch=64, lr=3e-3, eval_every=500):
    model = UniversalStepModel(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"UniversalStepModel: {n_params:,} params, d_state={cfg.d_state}",
          flush=True)

    for step in range(1, steps + 1):
        tokens, _ = make_batch(batch, device=device)
        logits = model(tokens)
        loss = compute_loss(logits, tokens)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % eval_every == 0 or step == 1:
            accs = evaluate_per_task(model, device)
            parts = "  ".join(f"{k}={v:.0%}" for k, v in accs.items())
            print(f"step {step:5d}  loss={loss.item():.3f}  {parts}", flush=True)

    return model, evaluate_per_task(model, device)


if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    print("\n" + "="*70, flush=True)
    print("UNIVERSAL STEP FUNCTION: one FSM, no task labels", flush=True)
    print("="*70, flush=True)

    results = {}
    for d_state in [16, 32, 64]:
        print(f"\n--- d_state={d_state} ---", flush=True)
        cfg = Mamba3Config(d_model=64, d_state=d_state, expand=2, headdim=16)
        _, accs = train(cfg, device, steps=8000)
        results[d_state] = accs

    print("\n" + "="*70, flush=True)
    print("RESULTS", flush=True)
    print("="*70, flush=True)
    header = f"{'d_state':<10}" + "".join(f"{n:>10}" for n in TASK_NAMES)
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for ds, accs in results.items():
        row = f"{ds:<10}" + "".join(f"{accs[n]:>9.0%}" for n in TASK_NAMES)
        print(row, flush=True)
    print(f"\n{'random':<10}" + "".join(f"{'':>7}{1/TOTAL_VOCAB:.0%}" for _ in TASK_NAMES),
          flush=True)
    print("\nDone.", flush=True)
