"""
Selective Copy experiment — can Mamba-3 learn to gate its state write?

Task: sequence of tokens from {0..vocab-1}. A special MARKER token appears
at random positions. The target at the final position is the token that
appeared *immediately after* the last MARKER. All other positions get a
don't-care label (-100).

Example (vocab=4, marker=4):
  input:   [2, 4, 1, 3, 4, 0, 2]
  target:  [-100, -100, -100, -100, -100, -100, 0]
                                          last marker at pos 4 → copy pos 5 value (0)

This tests a fundamentally different skill than parity:
  - Parity needs *every* token to update state (cumulative XOR).
  - Selective copy needs the model to *ignore* most tokens and only
    write when it sees the marker → read back the stored value at the end.

We compare Mamba-3 vs Mamba-2-like.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_platform.mamba3_minimal import Mamba3Block, Mamba2LikeBlock, Mamba3Config


def make_selective_copy_batch(B, L, vocab=4, marker=4, device="cpu"):
    """Generate a batch of selective-copy sequences."""
    # Random tokens from {0..vocab-1}
    tokens = torch.randint(0, vocab, (B, L), device=device)

    # Place markers at random positions (not last two positions — need room for value + readout)
    # Each sequence gets 1-3 markers
    targets = torch.full((B,), -100, dtype=torch.long, device=device)

    for b in range(B):
        n_markers = torch.randint(1, 4, (1,)).item()
        # Marker positions: not at last 2 positions
        positions = torch.randperm(L - 2)[:n_markers].sort().values
        for pos in positions:
            tokens[b, pos] = marker
        # Target = token after last marker
        last_marker_pos = positions[-1].item()
        # The value after the last marker is what we want to copy
        targets[b] = tokens[b, last_marker_pos + 1].item()

    return tokens, targets


class SelectiveCopyModel(nn.Module):
    def __init__(self, vocab_size, cfg, block_cls):
        super().__init__()
        self.embed = nn.Embedding(vocab_size + 1, cfg.d_model)  # +1 for marker token
        self.block = block_cls(cfg) if block_cls == Mamba2LikeBlock else block_cls(cfg)
        self.head = nn.Linear(cfg.d_model, vocab_size, bias=False)

    def forward(self, tokens):
        x = self.embed(tokens)           # (B, L, d_model)
        x = self.block(x)                # (B, L, d_model)
        logits = self.head(x[:, -1])     # (B, vocab) — only last position
        return logits


def train_and_eval(block_cls, name, cfg, vocab=4, marker=4, L=16, steps=1500,
                   lr=3e-3, batch=128, device="cpu"):
    model = SelectiveCopyModel(vocab + 1, cfg, block_cls).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    n_params = sum(p.numel() for p in model.parameters())

    best_acc = 0
    for step in range(1, steps + 1):
        tokens, targets = make_selective_copy_batch(batch, L, vocab, marker, device)
        logits = model(tokens)
        loss = F.cross_entropy(logits, targets)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 100 == 0 or step == steps:
            with torch.no_grad():
                tokens_e, targets_e = make_selective_copy_batch(512, L, vocab, marker, device)
                logits_e = model(tokens_e)
                preds = logits_e.argmax(dim=-1)
                valid = targets_e != -100
                acc = (preds[valid] == targets_e[valid]).float().mean().item()
                best_acc = max(best_acc, acc)
                print(f"  [{name}] step {step:4d}  loss={loss.item():.4f}  acc={acc:.1%}")

    # Length generalization
    print(f"  [{name}] --- length generalization ---")
    for L_eval in [L, L * 2, L * 4]:
        with torch.no_grad():
            tokens_e, targets_e = make_selective_copy_batch(512, L_eval, vocab, marker, device)
            logits_e = model(tokens_e)
            preds = logits_e.argmax(dim=-1)
            valid = targets_e != -100
            acc = (preds[valid] == targets_e[valid]).float().mean().item()
            print(f"  [{name}] L={L_eval:3d}  acc={acc:.1%}")

    return best_acc, n_params


if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    cfg = Mamba3Config(d_model=32, d_state=16, expand=2, headdim=16)
    vocab, marker, L = 4, 4, 16

    print(f"\nTask: selective copy, vocab={vocab}, marker={marker}, L={L}")
    print(f"Random baseline: {1/vocab:.1%}\n")

    print("=== Mamba-3 ===")
    acc3, p3 = train_and_eval(Mamba3Block, "Mamba-3", cfg,
                               vocab=vocab, marker=marker, L=L, device=device)

    print("\n=== Mamba-2-like ===")
    acc2, p2 = train_and_eval(Mamba2LikeBlock, "Mamba-2-like", cfg,
                               vocab=vocab, marker=marker, L=L, device=device)

    print(f"\n{'='*50}")
    print(f"Mamba-3:      {acc3:.1%}  ({p3:,} params)")
    print(f"Mamba-2-like: {acc2:.1%}  ({p2:,} params)")
    print(f"Random:       {1/vocab:.1%}")
