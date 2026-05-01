"""
Experiment A: Multi-task interference test.

Can one Mamba-3 model hold multiple algorithms simultaneously?
Three tasks, raw sequences, no language:

1. PARITY:  [bits...] → 0 or 1 (XOR accumulation)
2. SORTING: [numbers...] → [sorted numbers...] (comparison + reorder)
3. COUNTING: [tokens in {0,1,2}] → running sum mod 3 at last position

If the FPGA analogy holds, each task should use different "circuits" in
the state vector and they should coexist without interference.

We also train single-task baselines for comparison.
"""
import sys, os
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.nn.functional as F
from lab_platform.mamba3_minimal import Mamba3Block, Mamba3Config


# ── Data generators ──────────────────────────────────────────────────

def make_parity_batch(B, L, device):
    """Input: binary sequence. Target at last position: XOR of all bits."""
    bits = torch.randint(0, 2, (B, L), device=device)
    parity = bits.sum(dim=1) % 2  # (B,)
    # Encode: input tokens are 0,1. Target is 0 or 1.
    return bits, parity


def make_sorting_batch(B, L, vocab, device):
    """Input: sequence of numbers. Target: same numbers sorted ascending."""
    nums = torch.randint(0, vocab, (B, L), device=device)
    sorted_nums, _ = torch.sort(nums, dim=1)
    return nums, sorted_nums


def make_counting_batch(B, L, mod, device):
    """Input: tokens in {0..mod-1}. Target at last pos: cumsum mod M."""
    tokens = torch.randint(0, mod, (B, L), device=device)
    cumsum = tokens.sum(dim=1) % mod  # (B,)
    return tokens, cumsum


# ── Model ────────────────────────────────────────────────────────────

class MultiTaskModel(nn.Module):
    """
    Shared Mamba-3 backbone with separate task heads.
    A task token (0=parity, 1=sort, 2=count) prepended to the sequence
    tells the model which "circuit" to activate.
    """
    def __init__(self, cfg, vocab_size=16, n_tasks=3):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.n_tasks = n_tasks

        # +n_tasks for task indicator tokens
        self.embed = nn.Embedding(vocab_size + n_tasks, cfg.d_model)
        self.block = Mamba3Block(cfg)
        self.norm = nn.LayerNorm(cfg.d_model)

        # Task-specific heads
        self.parity_head = nn.Linear(cfg.d_model, 2)        # binary classification
        self.sort_head = nn.Linear(cfg.d_model, vocab_size)  # per-position classification
        self.count_head = nn.Linear(cfg.d_model, 3)          # mod-3 classification

    def forward(self, tokens, task_id):
        """
        tokens: (B, L) — raw task tokens (NOT including task indicator)
        task_id: int (0=parity, 1=sort, 2=count)
        """
        B, L = tokens.shape

        # Prepend task indicator token (offset by vocab_size)
        task_tok = torch.full((B, 1), self.vocab_size + task_id,
                              dtype=torch.long, device=tokens.device)
        x = torch.cat([task_tok, tokens], dim=1)  # (B, L+1)

        x = self.embed(x)
        x = self.block(x)
        x = self.norm(x)

        if task_id == 0:  # parity — classify from last position
            return self.parity_head(x[:, -1])  # (B, 2)
        elif task_id == 1:  # sort — classify each position (skip task token)
            return self.sort_head(x[:, 1:])    # (B, L, vocab)
        elif task_id == 2:  # count — classify from last position
            return self.count_head(x[:, -1])   # (B, 3)


# ── Training ─────────────────────────────────────────────────────────

def evaluate(model, task_id, device, L=16, vocab=8, n_eval=1024):
    model.eval()
    with torch.no_grad():
        if task_id == 0:
            tokens, targets = make_parity_batch(n_eval, L, device)
            logits = model(tokens, 0)
            acc = (logits.argmax(-1) == targets).float().mean().item()
        elif task_id == 1:
            tokens, targets = make_sorting_batch(n_eval, L, vocab, device)
            logits = model(tokens, 1)
            acc = (logits.argmax(-1) == targets).float().mean().item()
        elif task_id == 2:
            tokens, targets = make_counting_batch(n_eval, L, 3, device)
            logits = model(tokens, 2)
            acc = (logits.argmax(-1) == targets).float().mean().item()
    model.train()
    return acc


def train_multitask(cfg, device, steps=5000, L=16, vocab=8, batch=128,
                    lr=3e-3, eval_every=200, label="multi"):
    model = MultiTaskModel(cfg, vocab_size=vocab).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[{label}] {n_params:,} params, d_state={cfg.d_state}", flush=True)

    for step in range(1, steps + 1):
        total_loss = 0

        # Parity
        tok, tgt = make_parity_batch(batch, L, device)
        logits = model(tok, 0)
        loss_p = F.cross_entropy(logits, tgt)

        # Sorting
        tok, tgt = make_sorting_batch(batch, L, vocab, device)
        logits = model(tok, 1)
        loss_s = F.cross_entropy(logits.reshape(-1, vocab), tgt.reshape(-1))

        # Counting
        tok, tgt = make_counting_batch(batch, L, 3, device)
        logits = model(tok, 2)
        loss_c = F.cross_entropy(logits, tgt)

        loss = loss_p + loss_s + loss_c
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % eval_every == 0 or step == 1:
            acc_p = evaluate(model, 0, device, L, vocab)
            acc_s = evaluate(model, 1, device, L, vocab)
            acc_c = evaluate(model, 2, device, L, vocab)
            print(f"  [{label}] step {step:5d}  "
                  f"parity={acc_p:.1%}  sort={acc_s:.1%}  count={acc_c:.1%}  "
                  f"loss={loss.item():.3f}", flush=True)

    return model, {
        "parity": evaluate(model, 0, device, L, vocab),
        "sort": evaluate(model, 1, device, L, vocab),
        "count": evaluate(model, 2, device, L, vocab),
    }


def train_single_task(cfg, task_id, task_name, device, steps=5000, L=16,
                      vocab=8, batch=128, lr=3e-3, eval_every=200):
    """Single-task baseline for comparison."""
    model = MultiTaskModel(cfg, vocab_size=vocab).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    for step in range(1, steps + 1):
        if task_id == 0:
            tok, tgt = make_parity_batch(batch, L, device)
            logits = model(tok, 0)
            loss = F.cross_entropy(logits, tgt)
        elif task_id == 1:
            tok, tgt = make_sorting_batch(batch, L, vocab, device)
            logits = model(tok, 1)
            loss = F.cross_entropy(logits.reshape(-1, vocab), tgt.reshape(-1))
        elif task_id == 2:
            tok, tgt = make_counting_batch(batch, L, 3, device)
            logits = model(tok, 2)
            loss = F.cross_entropy(logits, tgt)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % eval_every == 0 or step == 1:
            acc = evaluate(model, task_id, device, L, vocab)
            print(f"  [single-{task_name}] step {step:5d}  acc={acc:.1%}", flush=True)

    return evaluate(model, task_id, device, L, vocab)


if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    L = 16
    vocab = 8
    steps = 5000

    # ── Experiment A: multi-task with varying d_state ──
    print("\n" + "="*70, flush=True)
    print("EXPERIMENT A: Multi-task interference test", flush=True)
    print("="*70, flush=True)

    results = {}
    for d_state in [8, 16, 32]:
        cfg = Mamba3Config(d_model=64, d_state=d_state, expand=2, headdim=16)
        _, accs = train_multitask(cfg, device, steps=steps, L=L, vocab=vocab)
        results[f"multi_d{d_state}"] = accs

    # ── Single-task baselines ──
    print("\n" + "="*70, flush=True)
    print("BASELINES: Single-task training", flush=True)
    print("="*70, flush=True)

    cfg_base = Mamba3Config(d_model=64, d_state=16, expand=2, headdim=16)
    results["single"] = {}
    for tid, name in [(0, "parity"), (1, "sort"), (2, "count")]:
        print(f"\n--- Training single-task: {name} ---", flush=True)
        acc = train_single_task(cfg_base, tid, name, device, steps=steps,
                                L=L, vocab=vocab)
        results["single"][name] = acc

    # ── Summary ──
    print("\n" + "="*70, flush=True)
    print("RESULTS SUMMARY", flush=True)
    print("="*70, flush=True)
    print(f"\n{'Config':<25} {'Parity':>10} {'Sort':>10} {'Count':>10}", flush=True)
    print("-"*55, flush=True)
    for key, accs in results.items():
        if key == "single":
            print(f"{'single-task (d16)':<25} "
                  f"{accs['parity']:>9.1%} "
                  f"{accs['sort']:>9.1%} "
                  f"{accs['count']:>9.1%}", flush=True)
        else:
            print(f"{key:<25} "
                  f"{accs['parity']:>9.1%} "
                  f"{accs['sort']:>9.1%} "
                  f"{accs['count']:>9.1%}", flush=True)

    print(f"\nRandom baselines:  parity=50.0%  sort={1/vocab:.1%}  count=33.3%",
          flush=True)
    print("\nDone.", flush=True)
