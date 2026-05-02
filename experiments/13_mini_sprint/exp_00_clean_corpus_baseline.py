"""exp_00 — clean within-movie corpus, byte-CE only baseline.

Control for the M4 mini sprint. No DeepSeek V4 lever yet — purely a
question of "does training on within-movie consecutive pairs (no
cross-movie contamination) at small scale produce a different retention
trajectory than the messy global corpus?"

Hypothesis: corpus contamination is one confounder we never separated.
If retention crosses 0.30 here at d_model=96 in 2000 steps, the
vast.ai 07_jepa runs were partly dataset-limited, not just architecture-
limited.

Config:
  d_model=96, n_layers=2, batch=32, seq_len=128, steps=2000
  byte_ce only, no jepa/conv/sigreg/contrastive, single corpus
  Adam lr=3e-4, cosine after warmup=200

Hardware: M4 mini, MPS device. ~10-15 min per run.

Run:
  cd ~/mamba3-hands-on
  .venv/bin/python experiments/13_mini_sprint/exp_00_clean_corpus_baseline.py \\
      --corpus data/movie_pairs_clean.txt
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Reuse the 07_jepa model — same Mamba-3 byte LM
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "experiments" / "07_jepa"))
from cortex_counting import CortexLM, CortexLMConfig

CANARY_PROMPTS = [
    b"Hello, how are you?\n",
    b"Hola, como estas?\n",
    b"The cat sat on the\n",
    b"En un lugar de la Mancha\n",
    b"It's getting cold today.\n",
    b"What did you do yesterday?\n",
    b"Tell me a short story.\n",
    b"Cuentame una historia corta.\n",
]


class CleanByteIterator:
    """Random fixed-window byte sampler over the clean corpus.

    Skips windows that are entirely empty (between-movie blank gaps).
    """
    def __init__(self, path: str | Path, batch_size: int, seq_len: int = 128,
                 seed: int = 0):
        self.data = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        return self

    def __next__(self):
        starts = self.rng.integers(0, len(self.data) - self.seq_len - 1,
                                   size=self.batch_size * 2)
        pick = []
        for s in starts:
            window = self.data[s:s + self.seq_len]
            if (window == 0).all():
                continue
            pick.append(s)
            if len(pick) == self.batch_size:
                break
        if len(pick) < self.batch_size:
            # Fallback: corpus is dense enough that this should never happen
            pick = list(starts[:self.batch_size])
        tokens = np.stack(
            [self.data[s:s + self.seq_len].astype(np.int64) for s in pick]
        )
        return torch.from_numpy(tokens)


def lr_at(step: int, base_lr: float, warmup: int, total: int) -> float:
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return base_lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))


@torch.no_grad()
def eval_canary(model: CortexLM, device: torch.device,
                max_new: int = 60) -> dict:
    """Compute retention/drift/diversity on canary prompts."""
    model.eval()
    completions, h_p_list, h_r_list = [], [], []
    for prompt in CANARY_PROMPTS:
        ids = torch.tensor([list(prompt)], dtype=torch.long, device=device)
        # h_p: residual at end-of-prompt
        _, _, residual_p, _ = model(ids, return_jepa=True,
                                     prompt_lens=torch.tensor([len(prompt)],
                                                              device=device))
        h_p = residual_p[0, -1].float().cpu()
        # greedy generate
        out = model.generate_greedy(list(prompt), max_new=max_new)
        completions.append(bytes(out).decode("utf-8", errors="replace"))
        # h_r: residual at end-of-(prompt+response)
        full_ids = torch.tensor([list(prompt) + list(out)],
                                dtype=torch.long, device=device)
        _, _, residual_r, _ = model(full_ids, return_jepa=True,
                                     prompt_lens=torch.tensor([len(prompt)],
                                                              device=device))
        h_r = residual_r[0, -1].float().cpu()
        h_p_list.append(h_p)
        h_r_list.append(h_r)
    H_p = torch.stack(h_p_list)
    H_r = torch.stack(h_r_list)
    cos = F.cosine_similarity(H_p, H_r, dim=-1)
    drift = (H_r - H_p).norm(dim=-1) / H_p.norm(dim=-1).clamp_min(1e-6)
    # Diversity: bigram-Jaccard mean across pairs
    sigs = []
    for c in completions:
        b = c.encode("utf-8")
        sigs.append(set(zip(b[:-1], b[1:])))
    n = len(sigs)
    sims = []
    for i in range(n):
        for j in range(i + 1, n):
            inter = len(sigs[i] & sigs[j])
            uni = len(sigs[i] | sigs[j])
            sims.append(inter / max(uni, 1))
    diversity = 1.0 - (sum(sims) / max(len(sims), 1))
    model.train()
    return {
        "retention": float(cos.mean()),
        "drift": float(drift.mean()),
        "diversity": diversity,
        "completions": completions,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/movie_pairs_clean.txt")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--d-model", type=int, default=96)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-name", default="exp_00_clean_corpus_baseline")
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available()
                              else "cpu")
    else:
        device = torch.device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = CortexLMConfig(
        n_layers=args.n_layers, d_model=args.d_model, d_state=16,
        expand=2, headdim=16, vocab_size=256, max_seq_len=args.seq_len,
        use_counter=False,
    )
    model = CortexLM(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[init] params={n_params:,} device={device}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=0.1)

    iterator = CleanByteIterator(args.corpus, args.batch_size,
                                 seq_len=args.seq_len, seed=args.seed)
    run_dir = Path("runs") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    loss_log = run_dir / "loss.jsonl"
    eval_log = run_dir / "eval.json"

    t0 = time.time()
    for step in range(args.steps):
        lr = lr_at(step, args.lr, args.warmup, args.steps)
        for pg in opt.param_groups:
            pg["lr"] = lr
        tokens = next(iterator).to(device)
        logits = model(tokens)
        pred = logits[:, :-1].reshape(-1, 256)
        tgt = tokens[:, 1:].reshape(-1)
        loss = F.cross_entropy(pred, tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 50 == 0:
            sps = (step + 1) / max(time.time() - t0, 1e-6)
            line = json.dumps({"step": step, "loss": float(loss),
                               "lr": lr, "sps": sps})
            print(f"step={step:5d} loss={float(loss):.4f} lr={lr:.2e} "
                  f"sps={sps:.2f}", flush=True)
            loss_log.open("a").write(line + "\n")

    # Final eval
    print("\n[eval] computing canary retention...", flush=True)
    metrics = eval_canary(model, device)
    metrics["final_loss"] = float(loss)
    metrics["steps"] = args.steps
    metrics["d_model"] = args.d_model
    metrics["params"] = n_params
    eval_log.write_text(json.dumps(metrics, indent=2))
    print(f"[eval] retention={metrics['retention']:.4f} "
          f"drift={metrics['drift']:.4f} "
          f"diversity={metrics['diversity']:.4f}", flush=True)
    print(f"[eval] sample completion 0: {metrics['completions'][0][:120]!r}",
          flush=True)
    print(f"[done] {run_dir}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
