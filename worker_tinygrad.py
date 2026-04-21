"""
Worker variant: tinygrad backend.

Same curriculum, same teacher, same metrics — but uses tinygrad instead
of PyTorch for the model. Competes in the genetic tournament.

tinygrad compiles the entire computation graph into fused GPU kernels,
potentially eliminating the Python loop overhead that limits PyTorch.

Usage:
    python worker_tinygrad.py --run-dir runs/exp_tg_001 --config runs/exp_tg_001/config.json
"""
import os
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["GPU"] = "1"  # tinygrad: use GPU
import sys
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import json
import time
import signal
import math
import random
from pathlib import Path
from collections import defaultdict

import numpy as np

from generators.teacher import AdaptiveTeacher
from progressive_model import ByteTokenizer, PAD, VOCAB_SIZE

try:
    from tinygrad import Tensor, dtypes, Device
    from tinygrad.nn import Linear, Embedding
    from tinygrad.nn.optim import AdamW
    from tinygrad.nn.state import safe_save, safe_load, get_state_dict, load_state_dict
    HAS_TINYGRAD = True
except ImportError:
    HAS_TINYGRAD = False
    print("tinygrad not available", flush=True)


# ── Minimal SSM in tinygrad ─────────────────────────────────────────

class TinyMamba:
    """Minimal Mamba-like SSM in tinygrad.
    Simplified: no RoPE, no trapezoidal — just the core scan.
    Tests whether tinygrad's compilation gives a speed advantage."""

    def __init__(self, d_model=64, d_state=16, expand=2):
        self.d_model = d_model
        self.d_inner = d_model * expand
        self.d_state = d_state

        # Input projection: z, x, B, C, dt, A
        d_proj = self.d_inner * 2 + d_state * 2 + self.d_inner // 16 * 2
        self.in_proj = Linear(d_model, d_proj)
        self.out_proj = Linear(self.d_inner, d_model)

        # D skip
        self.D = Tensor.ones(self.d_inner // 16)

    def __call__(self, u):
        B_, L, _ = u.shape
        d_inner = self.d_inner
        d_state = self.d_state
        nH = d_inner // 16
        hD = 16

        proj = self.in_proj(u)

        # Split
        z = proj[:, :, :d_inner].reshape(B_, L, nH, hD)
        x = proj[:, :, d_inner:d_inner*2].reshape(B_, L, nH, hD)
        Bp = proj[:, :, d_inner*2:d_inner*2+d_state]
        Cp = proj[:, :, d_inner*2+d_state:d_inner*2+d_state*2]
        dt_raw = proj[:, :, d_inner*2+d_state*2:d_inner*2+d_state*2+nH]
        A_raw = proj[:, :, d_inner*2+d_state*2+nH:]

        # Process
        DT = dt_raw.softplus()
        A = (-A_raw.softplus()).clip(max_=-1e-4)
        decay = (A * DT).exp()

        # Broadcast B, C
        Bp = Bp.unsqueeze(2).expand(B_, L, nH, d_state)
        Cp = Cp.unsqueeze(2).expand(B_, L, nH, d_state)

        # Sequential scan (tinygrad will compile this)
        h = Tensor.zeros(B_, nH, hD, d_state)
        outputs = []

        for t in range(L):
            x_t = x[:, t]          # (B, nH, hD)
            B_t = Bp[:, t]         # (B, nH, dS)
            C_t = Cp[:, t]         # (B, nH, dS)
            dec_t = decay[:, t]    # (B, nH)
            dt_t = DT[:, t]        # (B, nH)

            # inp = outer(x_t, B_t) * dt
            inp = (x_t.unsqueeze(-1) * B_t.unsqueeze(-2)) * dt_t.unsqueeze(-1).unsqueeze(-1)

            # State update
            h = dec_t.unsqueeze(-1).unsqueeze(-1) * h + inp

            # Output
            y_t = (h * C_t.unsqueeze(-2)).sum(axis=-1)  # (B, nH, hD)
            y_t = y_t + self.D.unsqueeze(0).unsqueeze(-1) * x_t
            y_t = y_t * (z[:, t] * z[:, t].sigmoid())  # silu gate

            outputs.append(y_t)

        y = Tensor.stack(outputs, dim=1)  # (B, L, nH, hD)
        y = y.reshape(B_, L, d_inner)
        return self.out_proj(y)


class TinyModel:
    """Progressive-compatible model in tinygrad."""
    def __init__(self, d_model=64, d_state=16):
        self.embed = Embedding(VOCAB_SIZE, d_model)
        self.block = TinyMamba(d_model, d_state)
        self.head = Linear(d_model, VOCAB_SIZE)
        self.d_model = d_model

    def __call__(self, tokens):
        x = self.embed(tokens)
        x = self.block(x)
        return self.head(x)

    def parameters(self):
        """Collect all parameters for optimizer."""
        params = []
        for obj in [self.embed, self.block.in_proj, self.block.out_proj, self.head]:
            if hasattr(obj, 'weight'):
                params.append(obj.weight)
            if hasattr(obj, 'bias') and obj.bias is not None:
                params.append(obj.bias)
        params.append(self.block.D)
        return params


# ── Training ────────────────────────────────────────────────────────

def write_metrics(run_dir, metrics):
    tmp = run_dir / "metrics.tmp"
    final = run_dir / "metrics.json"
    with open(tmp, "w") as f:
        json.dump(metrics, f, indent=2)
    tmp.rename(final)


def train(args):
    if not HAS_TINYGRAD:
        print("tinygrad not available, exiting", flush=True)
        return

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.load(open(args.config))

    print(f"[tinygrad worker] Device: {Device.DEFAULT}", flush=True)

    model = TinyModel(d_model=cfg["d_model"], d_state=cfg.get("d_state", 16))
    params = model.parameters()
    n_params = sum(p.numel() for p in params)
    print(f"[tinygrad] Model: {n_params:,} params", flush=True)

    opt = AdamW(params, lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.1))

    teacher = AdaptiveTeacher(sequential_unlock=True)
    tok = ByteTokenizer()

    log = open(run_dir / "train.log", "w")
    log.write(f"tinygrad worker: {cfg}\nParams: {n_params:,}\n{'='*60}\n")

    (run_dir / "status").write_text("running")

    should_stop = False
    def handle_signal(sig, frame):
        nonlocal should_stop
        should_stop = True
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    cycle = 0
    best_fresh = 0.0
    steps_per_cycle = cfg.get("steps_per_cycle", 200)

    while not should_stop:
        cycle += 1
        t0 = time.time()

        # Generate data
        raw = teacher.generate(5000)
        examples = []
        for ex in raw:
            tokens, sep_pos = tok.encode_curriculum(ex)
            examples.append((tokens, sep_pos))

        # Train
        cycle_loss = 0.0
        for step in range(steps_per_cycle):
            # Build batch
            bs = cfg["batch_size"]
            indices = [random.randint(0, len(examples)-1) for _ in range(bs)]
            max_len = max(len(examples[i][0]) for i in indices)
            tokens_np = np.full((bs, max_len), PAD, dtype=np.int32)
            sep_positions = []
            for bi, idx in enumerate(indices):
                toks, sep = examples[idx]
                tokens_np[bi, :len(toks)] = toks
                sep_positions.append(sep)

            tokens_t = Tensor(tokens_np)
            logits = model(tokens_t)

            # Loss (cross-entropy on output portion)
            B, L, V = logits.shape
            # Build mask
            mask_np = np.zeros((B, L), dtype=np.float32)
            for b in range(B):
                mask_np[b, sep_positions[b]:L-1] = 1.0
            pad_np = (tokens_np != PAD).astype(np.float32)
            target_valid = pad_np[:, 1:]
            pred_mask = mask_np[:, :L-1] * target_valid
            mask_sum = pred_mask.sum()

            if mask_sum > 0:
                logits_flat = logits[:, :L-1].reshape(-1, V)
                targets_flat = Tensor(tokens_np[:, 1:].reshape(-1).astype(np.int64))
                mask_flat = Tensor(pred_mask.reshape(-1))

                # Cross-entropy
                log_probs = logits_flat.log_softmax(axis=-1)
                nll = -log_probs.gather(idx=targets_flat.unsqueeze(-1), dim=-1).squeeze(-1)
                loss = (nll * mask_flat).sum() / mask_sum

                opt.zero_grad()
                loss.backward()
                opt.step()
                cycle_loss += loss.numpy().item()

        cycle_loss /= max(steps_per_cycle, 1)
        elapsed = time.time() - t0

        # Evaluate (using numpy, not tinygrad)
        from generators.level0_patterns import generate_dataset
        eval_raw = generate_dataset(200)
        by_type = defaultdict(list)
        for ex in eval_raw:
            by_type[ex["type"]].append(ex)

        type_accs = {}
        for task_type, task_examples in by_type.items():
            correct = 0
            total = 0
            for ex in task_examples[:30]:
                tokens, sep_pos = tok.encode_curriculum(ex)
                out_bytes = list(ex["output"].encode("utf-8"))
                t = Tensor(np.array([tokens], dtype=np.int32))
                logits = model(t)
                logits_np = logits.numpy()
                ok = True
                for j, expected in enumerate(out_bytes):
                    p = sep_pos + j
                    if p >= logits_np.shape[1] or logits_np[0, p].argmax() != expected:
                        ok = False
                        break
                if ok:
                    correct += 1
                total += 1
            type_accs[task_type] = correct / max(total, 1)

        fresh = sum(type_accs.values()) / max(len(type_accs), 1)
        best_fresh = max(best_fresh, fresh)

        teacher.set_step(cycle * steps_per_cycle)
        teacher.observe(type_accs)

        parity = type_accs.get("parity", 0)
        log.write(f"  cycle {cycle:4d}  loss={cycle_loss:.4f}  fresh={fresh:.1%}  "
                  f"parity={parity:.0%}  {elapsed:.1f}s  [tinygrad]\n")
        log.flush()

        write_metrics(run_dir, {
            "cycle": cycle,
            "loss": cycle_loss,
            "fresh": fresh,
            "best_fresh": best_fresh,
            "type_accs": type_accs,
            "elapsed": elapsed,
            "config": {**cfg, "backend": "tinygrad"},
            "params": n_params,
            "teacher_status": teacher.get_status(),
            "mastery_log": teacher.mastery_log,
        })

    (run_dir / "status").write_text("paused")
    log.write(f"Worker stopped at cycle {cycle}\n")
    log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(args)
