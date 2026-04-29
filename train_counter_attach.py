"""Counter-attach experiment: bind a CounterPrimitive onto a frozen
bilingual LM and fine-tune ONLY the counter's adapters.

Tests the cortex thesis on a language-trained LM: can a 770-param
primitive learn to drive emit-a-vs-newline decisions through a
frozen 473k-param Mamba-3 LM that was trained on bilingual text?

The bilingual LM was trained with a 5% unary mixin (data/bilingual.txt
includes `*N:aN` lines), so the LM's hidden state already encodes
'I am reading the unary form'. The counter primitive just has to
discover those signals from a fresh init and use them to count.

Frozen: every parameter of the trained CortexLM (the bilingual LM).
Trainable: counter.inc_proj, counter.reset_proj, counter.read_proj
(plus their biases). Total trainable: ~770 params.

Run:
    python train_counter_attach.py \\
        --lm-ckpt checkpoints/lm/step_FINAL.pt \\
        --steps 1000 \\
        --out checkpoints/lm_counter/

The eval at the end runs the same probe prompts as cortex_counting.eval
(N in {3, 30, 50, 100, 200, 500}) and a couple of bilingual prompts
("count to fifty:", "cuenta hasta cincuenta:") to test that the
language layer still works while the counter primitive is engaged.
"""
from __future__ import annotations
import argparse
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from cortex_counting import (
    CortexLM, CortexLMConfig, CounterPrimitive,
    CountingDataset, eval_counting,
)


# ---------------------------------------------------------------------------
# Mixed-batch dataset: bilingual lines + cortex counting examples
# ---------------------------------------------------------------------------

class MixedBatchDataset:
    """Random window over byte corpus, with `unary_p` fraction of batch
    elements replaced by fresh `*N:aN\\n` examples."""

    def __init__(self, corpus_path: str, seq_len: int, n_min: int, n_max: int,
                 unary_p: float, device: str = "cpu", seed: int = 0):
        with open(corpus_path, "rb") as f:
            self.data = torch.tensor(list(f.read()), dtype=torch.long)
        self.seq_len = seq_len
        self.unary_p = unary_p
        self.device = device
        self.counting = CountingDataset(n_min, n_max, seq_len, device=device, seed=seed)
        # Use a separate generator to keep determinism even when CountingDataset
        # consumes its own seeded RNG.
        self.gen = torch.Generator()
        self.gen.manual_seed(seed)
        print(f"corpus: {len(self.data):,} bytes from {corpus_path}", flush=True)

    def get_batch(self, batch_size: int):
        L = self.seq_len
        # First fill the whole batch from the bilingual corpus
        max_start = len(self.data) - L - 1
        starts = torch.randint(0, max_start, (batch_size,), generator=self.gen)
        x = torch.stack([self.data[s : s + L] for s in starts])
        y = torch.stack([self.data[s + 1 : s + L + 1] for s in starts])
        mask = torch.ones((batch_size, L), dtype=torch.bool)

        # Replace ~unary_p fraction with cortex unary examples
        n_unary = max(1, int(round(batch_size * self.unary_p))) if self.unary_p > 0 else 0
        if n_unary > 0:
            cx, cy, cmask = self.counting.get_batch(n_unary)
            x[:n_unary]    = cx[:, :L].cpu()
            y[:n_unary]    = cy[:, :L].cpu()
            mask[:n_unary] = cmask[:, :L].cpu()

        return x.to(self.device), y.to(self.device), mask.to(self.device)


# ---------------------------------------------------------------------------
# Build attached model: load frozen LM + fresh counter
# ---------------------------------------------------------------------------

def build_attached_lm(lm_ckpt_path: str, device: str,
                      injection_scale: float = 10.0) -> CortexLM:
    """Load a trained bilingual CortexLM and attach a fresh CounterPrimitive."""
    sd = torch.load(lm_ckpt_path, map_location=device, weights_only=False)
    lm_cfg = sd["cfg"]
    print(f"loaded LM cfg: {lm_cfg}")

    # Fresh counter sized to this LM's d_model
    counter = CounterPrimitive(
        d_model=lm_cfg.d_model,
        layer=0,
        n_counters=2,
        readout="unbounded",
        injection_scale=injection_scale,
    )
    model = CortexLM(lm_cfg, primitives=[counter]).to(device)

    # Load LM weights but skip the new counter (which doesn't exist in the
    # ckpt). Strict=False allows missing keys (counter.* are missing).
    incompat = model.load_state_dict(sd["model"], strict=False)
    new_keys = [k for k in incompat.missing_keys if not k.startswith("primitives.0.")]
    if new_keys:
        raise RuntimeError(f"unexpected missing keys: {new_keys}")
    print(f"loaded LM weights; counter primitive freshly initialised "
          f"({sum(p.numel() for p in counter.parameters()):,} new params)")

    # Freeze the LM, leave only counter trainable
    for name, p in model.named_parameters():
        if "primitives.0." in name:
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params: total={n_total:,}  trainable={n_train:,}  frozen={n_total - n_train:,}")
    return model


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_counter(
    lm_ckpt: str,
    corpus: str,
    out_dir: str,
    steps: int,
    batch_size: int,
    lr: float,
    lambda_aux: float,
    unary_p: float,
    log_every: int,
    ckpt_every: int,
    seed: int,
    injection_scale: float = 10.0,
):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")
    torch.manual_seed(seed)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    model = build_attached_lm(lm_ckpt, device, injection_scale=injection_scale)
    seq_len = model.cfg.max_seq_len

    dataset = MixedBatchDataset(
        corpus, seq_len, n_min=1, n_max=30,
        unary_p=unary_p, device=device, seed=seed,
    )

    # Optimize only the counter params
    train_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(train_params, lr=lr, weight_decay=0.05, betas=(0.9, 0.95))

    log_path = out_path / "training.log"
    log_path.write_text(f"# counter-attach: lm={lm_ckpt} corpus={corpus} "
                        f"steps={steps} unary_p={unary_p} lambda_aux={lambda_aux}\n")
    def append_log(s): open(log_path, "a").write(s + "\n")

    print(f"training {sum(p.numel() for p in train_params):,} params, "
          f"steps={steps}, batch={batch_size}", flush=True)
    t0 = time.time()
    model.train()

    for step in range(1, steps + 1):
        x, y, mask = dataset.get_batch(batch_size)
        logits, prim_outputs = model(x, return_aux=True)
        ce = F.cross_entropy(
            logits.reshape(-1, model.cfg.vocab_size),
            y.reshape(-1), reduction="none",
        ).view_as(y)
        main_loss = (ce * mask.float()).sum() / mask.float().sum().clamp_min(1.0)
        aux_loss = model.aux_loss(x, prim_outputs, mask)
        loss = main_loss + lambda_aux * aux_loss

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(train_params, 1.0)
        opt.step()

        if step % log_every == 0 or step == 1:
            elapsed = time.time() - t0
            line = (f"step {step:5d}/{steps}  ce={main_loss.item():.3f}  "
                    f"aux={aux_loss.item():.4f}  "
                    f"elapsed={elapsed:.1f}s")
            print(line, flush=True)
            append_log(line)

        if step % ckpt_every == 0 or step == steps:
            ckpt = {"model": model.state_dict(), "cfg": model.cfg, "step": step}
            torch.save(ckpt, out_path / f"step_{step:06d}.pt")

    final_path = out_path / "step_FINAL.pt"
    torch.save({"model": model.state_dict(), "cfg": model.cfg, "step": steps}, final_path)
    print(f"\nFinal: {final_path}", flush=True)
    return model


# ---------------------------------------------------------------------------
# Eval — same shape as cortex_counting.eval, plus bilingual prompts
# ---------------------------------------------------------------------------

def run_eval(model: CortexLM, device: str):
    """Evaluate counting at OOD lengths AND a couple of bilingual prompts."""
    print("\n" + "=" * 64)
    print("Counting eval (N from training distribution to far-OOD):")
    # Hard gates at inference (the byte-perfect mode)
    if model.counter is not None:
        model.counter.hard_gates_inference = True
    eval_counting(model, [3, 30, 50, 100, 200, 500], device, print_samples=True)

    # Bilingual probes — generate from the LM with the counter active
    print("\nBilingual-probe samples (counter attached, hard gates):")
    model.eval()
    bilingual_prompts = [
        "*****:",
        "**********:",
        "***************:",
        "The cat ",
        "El gato ",
        "Hola, ",
    ]
    for p in bilingual_prompts:
        prompt_bytes = list(p.encode("utf-8"))
        with torch.no_grad():
            gen = model.generate_greedy(prompt_bytes, max_new=80, max_ctx=8192)
        text = bytes(gen).decode("utf-8", errors="replace")
        print(f"  {p!r:>20} -> {text[:60]!r}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lm-ckpt", required=True,
                    help="path to the trained bilingual LM checkpoint")
    ap.add_argument("--corpus", default="data/bilingual.txt")
    ap.add_argument("--out", default="checkpoints/lm_counter")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--lambda-aux", type=float, default=0.5)
    ap.add_argument("--unary-p", type=float, default=0.5,
                    help="fraction of each batch that is fresh cortex unary "
                         "examples (default 0.5 — heavy mixin so the counter "
                         "actually learns to fire on the right bytes)")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--injection-scale", type=float, default=10.0,
                    help="multiplier on residual injection (Phase 5/6 of cortex "
                         "showed bigger scale = louder counter signal in residual)")
    ap.add_argument("--eval-only", action="store_true",
                    help="skip training, just eval (use --resume in --out)")
    args = ap.parse_args()

    if args.eval_only:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        ckpt_path = Path(args.out) / "step_FINAL.pt"
        sd = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = sd["cfg"]
        counter = CounterPrimitive(d_model=cfg.d_model, layer=0, n_counters=2,
                                    readout="unbounded", injection_scale=args.injection_scale)
        model = CortexLM(cfg, primitives=[counter]).to(device)
        model.load_state_dict(sd["model"])
        run_eval(model, device)
        return

    model = train_counter(
        lm_ckpt=args.lm_ckpt, corpus=args.corpus, out_dir=args.out,
        steps=args.steps, batch_size=args.batch_size, lr=args.lr,
        lambda_aux=args.lambda_aux, unary_p=args.unary_p,
        log_every=args.log_every, ckpt_every=args.ckpt_every, seed=args.seed,
        injection_scale=args.injection_scale,
    )
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    run_eval(model, device)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
