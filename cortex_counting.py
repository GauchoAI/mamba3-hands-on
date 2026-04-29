"""
Cortex experiment 1: a byte-perfect counter primitive wired into the residual
stream of a small Mamba-3 LM. Tests internal reasoning vs. external tool use.

The hypothesis from the cortex thread: a small LM cannot count to arbitrary N
from byte-level next-token prediction alone (Apple "Illusion of Thinking"
counting failure). But it should be able to *invoke* a counter primitive
internally — without emitting tool-call tokens — if the primitive is exposed
as a forward-pass module in the residual stream.

Format (unary in / unary out — isolates counting from digit composition):
    N=5  ->  "*****:aaaaa\n"
    N=12 ->  "************:aaaaaaaaaaaa\n"

The model reads N stars, sees ':', emits N a's, terminates with newline.
Pure counting: copy-the-length without the digit-composition confound.

Setup:
  - Train: N drawn uniformly from [1, 30]
  - Eval:  N in {3, 15, 30, 50, 100, 200, 500} — last three are far OOD
  - Baseline: vanilla Mamba-3 LM
  - Cortex:   same LM + a CounterPrimitive injected into the residual stream

If only the cortex model generalizes past N=30, that's the existence proof:
language reasoning can be implemented as differentiable composition over
crystallized algorithmic primitives, no tool-call loop required.

Run:
  python cortex_counting.py train     # trains both models, saves checkpoints
  python cortex_counting.py eval      # length-generalization eval
  python cortex_counting.py demo      # shows a few generated samples
"""
from __future__ import annotations
import math
import os
import sys
import time
import random
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba3_minimal import Mamba3Block, Mamba3Config


# ----------------------------------------------------------------------------
# Plugin / adapter architecture
# ----------------------------------------------------------------------------
#
# A Primitive is a forward-pass module that attaches to one of the LM's SSM
# layers. It reads the LM's hidden state there, runs its own (often
# parameter-free) compute, and emits a contribution to the residual stream.
# Optionally, it exposes intermediate signals for auxiliary supervision and
# implements its own aux_loss() so training stays plugin-agnostic.
#
# Adding a new primitive (HanoiGRU, GCD, etc.) is now: subclass `Primitive`,
# implement forward()/aux_loss(), pass it to CortexLM(cfg, primitives=[...]).
# CortexLM never imports primitive-specific machinery.
#
# Limitation today: each primitive's adapter (gate Linears + read_proj) is
# co-trained end-to-end with the LM. That is, the *interface* is plugin-style,
# but the *training story* is still co-training. Train-free composition (the
# real plug-and-play property) is the next architectural gate — likely a
# frozen-LM + per-primitive-adapter fine-tune, or a pre-trained "plugin port"
# that the LM is shaped to drive through a fixed protocol.

class Primitive(nn.Module):
    """Base class for residual-stream primitives.

    Subclasses implement:
      - forward(x, tokens) -> dict with at least {'injection': (B, L, d_model)}
        Optional keys: 'head_bias' (B,L,vocab) added to LM logits;
        primitive-specific intermediates for aux supervision.
      - aux_loss(tokens, fwd_out, mask) -> scalar tensor (default: zero)

    Each primitive attaches at one SSM layer (the `layer` field, 0-indexed).
    """

    def __init__(self, d_model: int, layer: int = 0, name: str = ""):
        super().__init__()
        self.d_model = d_model
        self.layer = layer
        self.name = name or self.__class__.__name__

    def forward(self, x: torch.Tensor, tokens: torch.Tensor) -> dict:
        raise NotImplementedError

    def aux_loss(self, tokens: torch.Tensor, fwd_out: dict,
                 mask: torch.Tensor) -> torch.Tensor:
        return torch.zeros((), device=tokens.device, dtype=torch.float32)


class CounterPrimitive(Primitive):
    """A stateful integer counter that runs INSIDE the forward pass.

    Per position, the LM emits two gates (sigmoid):
        inc_gate:    add this much (≈0 or ≈1) to each counter
        reset_gate:  multiply existing counter by (1 - this) before adding

    Recurrence (per counter k, position t):
        c[t] = (1 - reset[t]) * c[t-1] + inc[t]

    The arithmetic itself has no learned parameters — it is a parameter-free
    integer accumulator (the byte-perfect crystallized algorithm). Only the
    *gates* (read from LM hidden state) and the *readout projection* are
    learned. Counter is read back into d_model via sinusoidal embedding,
    which has no upper bound — same trick that lets transformers handle
    unbounded positions.

    Crucially: this is NOT a tool call. There are no tokens emitted, no
    parser, no Python loop outside the forward pass. The counter is a
    differentiable module the LM steers by its hidden state, and the
    counter's value flows back into the residual stream as a vector.
    """

    def __init__(self, d_model: int, layer: int = 0, n_counters: int = 2,
                 n_freqs: int = 16, max_period: float = 4096.0,
                 readout: str = "sinusoidal", vocab_size: int = 0,
                 injection_scale: float = 1.0):
        super().__init__(d_model, layer, name="counter")
        self.n_counters = n_counters
        self.n_freqs = n_freqs
        self.readout = readout
        self.injection_scale = injection_scale
        # When True and self.training is False, gate values become hard
        # 0/1 thresholds on the logit instead of sigmoid. Removes counter
        # slippage from SSM context drift at OOD positions. Toggleable
        # post-hoc on a trained checkpoint — no retraining required.
        self.hard_gates_inference = False

        # Increment gate per (position, counter)
        self.inc_proj = nn.Linear(d_model, n_counters)
        # Reset gate per (position, counter)  — multiplicative decay
        self.reset_proj = nn.Linear(d_model, n_counters)

        if readout == "sinusoidal":
            # Sinusoidal frequencies (log-spaced periods 1..max_period)
            freqs = torch.exp(torch.linspace(0.0, math.log(max_period), n_freqs))
            self.register_buffer("freqs", freqs, persistent=False)
            read_in = n_counters * 2 * n_freqs
        elif readout == "unbounded":
            # Phase 3 (v1): raw + tanh — broke OOD because raw features
            # extrapolated linearly past training range and flooded residual.
            # Phase 3 (v2): tanh-only with temperature k=8. Features:
            #   tanh(c/k) (K)  +  tanh(diff/k) (P)
            # where P = K*(K-1)/2. All features bounded in (-1, 1) regardless
            # of N; k=8 gives smooth resolution across training range 0..30
            # (tanh(30/8)≈0.9997) and clean saturation at OOD.
            P = n_counters * (n_counters - 1) // 2
            read_in = n_counters + P
        else:
            raise ValueError(f"unknown readout: {readout}")

        self.read_proj = nn.Linear(read_in, d_model)
        # Small random init: contribution is small at start, but gradients
        # can flow back to the gates (zeroing the weight kills gate gradients).
        nn.init.normal_(self.read_proj.weight, std=0.02)
        nn.init.zeros_(self.read_proj.bias)

        # Phase 4: optional direct head-logit bias path. Counter readout
        # features → vocab logits, added to LM head output. Bypasses the
        # SSM-and-tied-embedding-head bottleneck so the counter can
        # decisively control which byte is emitted (specifically: high
        # tanh(diff/k) → boost 'a' logit, suppress '\n').
        if vocab_size > 0:
            self.head_bias_proj = nn.Linear(read_in, vocab_size)
            nn.init.normal_(self.head_bias_proj.weight, std=0.02)
            nn.init.zeros_(self.head_bias_proj.bias)
        else:
            self.head_bias_proj = None

        # Increment gate: bias toward NOT incrementing every position.
        # We want the LM to learn WHEN to increment.
        nn.init.constant_(self.inc_proj.bias, -2.0)
        # Reset gate: bias toward NEVER resetting. Counters persist by default.
        nn.init.constant_(self.reset_proj.bias, -5.0)

    def _sinusoidal(self, c: torch.Tensor) -> torch.Tensor:
        """c: (B, L, K) -> (B, L, K * 2 * n_freqs)"""
        sc = c.unsqueeze(-1) / self.freqs
        emb = torch.cat([torch.sin(sc), torch.cos(sc)], dim=-1)
        B, L, K, _ = emb.shape
        return emb.view(B, L, K * 2 * self.n_freqs)

    def _unbounded(self, c: torch.Tensor) -> torch.Tensor:
        """c: (B, L, K) -> (B, L, K + P) where P = K*(K-1)/2.

        Saturated readout: tanh(c/k) and tanh(diff/k) only — every feature
        bounded in (-1, 1) regardless of N. Temperature k=8 gives smooth
        resolution across the training range (tanh(30/8) ≈ 0.9997) and
        clean saturation OOD. The tanh(diff/k) channel is the decisive
        "is c[A] still ahead of c[B]" signal that lets the LM emit '\\n'
        at the right moment regardless of absolute count.
        """
        K = self.n_counters
        k = 8.0
        feats = [torch.tanh(c / k)]
        if K >= 2:
            diffs = []
            for i in range(K):
                for j in range(i + 1, K):
                    diffs.append(c[..., i:i + 1] - c[..., j:j + 1])
            diff = torch.cat(diffs, dim=-1)
            feats.append(torch.tanh(diff / k))
        return torch.cat(feats, dim=-1)

    def forward(self, x: torch.Tensor, tokens: torch.Tensor = None) -> dict:
        """
        x: (B, L, d_model) — LM hidden state at attachment point
        tokens: (B, L) — input bytes (unused here, but part of Primitive API)

        Returns dict:
            injection: (B, L, d_model) — added to residual stream
            inc_logits: (B, L, K) — pre-sigmoid increment logits, for aux loss
            head_bias: (B, L, vocab) or None — optional direct head bias path
        """
        B, L, _ = x.shape
        K = self.n_counters

        inc_logits = self.inc_proj(x)              # (B, L, K)
        reset_logits = self.reset_proj(x)
        if self.hard_gates_inference and not self.training:
            inc = (inc_logits > 0).float()
            rst = (reset_logits > 0).float()
        else:
            inc = torch.sigmoid(inc_logits)
            rst = torch.sigmoid(reset_logits)
        keep = 1.0 - rst

        # Sequential scan: c[t] = keep[t] * c[t-1] + inc[t]
        c_prev = torch.zeros(B, K, device=x.device, dtype=x.dtype)
        outs = []
        for t in range(L):
            c_prev = keep[:, t] * c_prev + inc[:, t]
            outs.append(c_prev)
        counters = torch.stack(outs, dim=1)        # (B, L, K)

        if self.readout == "sinusoidal":
            emb = self._sinusoidal(counters)
        else:
            emb = self._unbounded(counters)
        injection = self.injection_scale * self.read_proj(emb)
        head_bias = self.head_bias_proj(emb) if self.head_bias_proj is not None else None
        return {
            "injection": injection,
            "inc_logits": inc_logits,
            "head_bias": head_bias,
            "counters": counters,
        }

    def aux_loss(self, tokens: torch.Tensor, fwd_out: dict,
                 mask: torch.Tensor) -> torch.Tensor:
        """BCE on inc_logits vs byte-conditional targets.

        Counter A target = 1 where token == '*' (counts input length).
        Counter B target = 1 where token == 'a' (counts output length).
        """
        is_star = (tokens == STAR_BYTE).float()
        is_a    = (tokens == A_BYTE).float()
        target = torch.stack([is_star, is_a], dim=-1)             # (B, L, K)
        bce = F.binary_cross_entropy_with_logits(
            fwd_out["inc_logits"], target, reduction="none"
        ).mean(dim=-1)                                            # (B, L)
        return (bce * mask.float()).sum() / mask.float().sum().clamp_min(1.0)


# ----------------------------------------------------------------------------
# LMs: baseline (vanilla) and cortex (with counter primitive)
# ----------------------------------------------------------------------------

@dataclass
class CortexLMConfig:
    n_layers: int = 2
    d_model: int = 96
    d_state: int = 16
    expand: int = 2
    headdim: int = 16
    vocab_size: int = 256       # byte-level
    max_seq_len: int = 80       # N=30 unary needs ~62 chars
    # counter
    use_counter: bool = True
    n_counters: int = 2
    counter_layer: int = 0      # inject after this SSM layer (0-indexed)
    counter_readout: str = "sinusoidal"   # "sinusoidal" | "unbounded" (Phase 3)
    counter_head_bias: bool = False       # Phase 4: direct counter→logit path
    counter_injection_scale: float = 1.0  # Phase 5: multiplier on residual injection


class CortexLM(nn.Module):
    """Mamba-3 LM with a registry of residual-stream Primitives.

    The LM holds a `primitives` ModuleList. Each primitive declares which
    SSM layer it attaches to. CortexLM's forward is primitive-agnostic:
    it iterates over registered primitives at each layer's residual point,
    sums their `injection` contributions into the residual, and collects
    optional `head_bias` paths into the final logits.

    Two construction paths:
      1. `CortexLM(cfg)` — backward-compat, builds default counter from
         CortexLMConfig flags (use_counter, counter_*).
      2. `CortexLM(cfg, primitives=[...])` — explicit, future-friendly.
         Pass any list of Primitive subclasses; the LM forward handles them
         uniformly. Use cfg.use_counter=False to skip the auto-built counter.
    """
    def __init__(self, cfg: CortexLMConfig, primitives=None):
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.embed_norm = nn.LayerNorm(cfg.d_model)

        ssm_cfg = Mamba3Config(
            d_model=cfg.d_model,
            d_state=cfg.d_state,
            expand=cfg.expand,
            headdim=cfg.headdim,
        )
        self.layers = nn.ModuleList()
        for _ in range(cfg.n_layers):
            self.layers.append(nn.ModuleDict({
                "norm": nn.LayerNorm(cfg.d_model),
                "block": Mamba3Block(ssm_cfg),
            }))

        if primitives is None and cfg.use_counter:
            primitives = [CounterPrimitive(
                cfg.d_model,
                layer=cfg.counter_layer,
                n_counters=cfg.n_counters,
                readout=cfg.counter_readout,
                vocab_size=cfg.vocab_size if cfg.counter_head_bias else 0,
                injection_scale=cfg.counter_injection_scale,
            )]
        self.primitives = nn.ModuleList(primitives or [])

        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.embed.weight  # tied

        for name, p in self.named_parameters():
            skip = ("read_proj" in name or "inc_proj" in name
                    or "reset_proj" in name or "head_bias_proj" in name)
            if p.dim() >= 2 and not skip:
                nn.init.normal_(p, std=0.02)
            elif "bias" in name and not skip:
                nn.init.zeros_(p)

        n_params = sum(p.numel() for p in self.parameters())
        prim_summary = ", ".join(
            f"{p.name}({sum(q.numel() for q in p.parameters()):,}p, L{p.layer})"
            for p in self.primitives
        ) or "none"
        tag = "cortex" if self.primitives else "baseline"
        print(f"[{tag}] {cfg.n_layers}L d_model={cfg.d_model} "
              f"params={n_params:,} primitives=[{prim_summary}]")

    @property
    def counter(self):
        """Backward-compat alias: first CounterPrimitive in the registry."""
        for p in self.primitives:
            if isinstance(p, CounterPrimitive):
                return p
        return None

    def forward(self, tokens: torch.Tensor, return_aux: bool = False):
        """
        tokens: (B, L) long
        return_aux=False -> logits (B, L, vocab)
        return_aux=True  -> (logits, prim_outputs: dict[name -> dict])
        """
        x = self.embed(tokens)
        x = self.embed_norm(x)
        prim_outputs = {}
        head_bias_total = None
        for i, layer in enumerate(self.layers):
            residual = x
            x = layer["norm"](x)
            x = layer["block"](x)
            x = residual + x
            for p in self.primitives:
                if p.layer == i:
                    out = p(x, tokens)
                    x = x + out["injection"]
                    prim_outputs[p.name] = out
                    hb = out.get("head_bias")
                    if hb is not None:
                        head_bias_total = hb if head_bias_total is None else head_bias_total + hb
        x = self.final_norm(x)
        logits = self.head(x)
        if head_bias_total is not None:
            logits = logits + head_bias_total
        if return_aux:
            return logits, prim_outputs
        return logits

    def aux_loss(self, tokens: torch.Tensor, prim_outputs: dict,
                 mask: torch.Tensor) -> torch.Tensor:
        """Sum of all primitives' aux losses. Zero if no primitives or none expose aux."""
        total = torch.zeros((), device=tokens.device, dtype=torch.float32)
        for p in self.primitives:
            if p.name in prim_outputs:
                total = total + p.aux_loss(tokens, prim_outputs[p.name], mask)
        return total

    @torch.no_grad()
    def generate_greedy(self, prompt_bytes: list[int], max_new: int,
                        max_ctx: int | None = None) -> list[int]:
        device = next(self.parameters()).device
        toks = torch.tensor([prompt_bytes], dtype=torch.long, device=device)
        max_ctx = max_ctx or 4096
        for _ in range(max_new):
            ctx = toks[:, -max_ctx:]
            logits = self(ctx)[:, -1]
            nxt = logits.argmax(dim=-1, keepdim=True)
            toks = torch.cat([toks, nxt], dim=1)
        return toks[0, len(prompt_bytes):].tolist()


# ----------------------------------------------------------------------------
# Counting dataset
# ----------------------------------------------------------------------------

def make_count_example(n: int) -> str:
    """Unary copy-the-length: N stars, colon, N a's, newline.

    This format isolates counting from digit composition: the model never
    has to read or write decimal numbers. Train on N in [1, 30] and the
    model has never seen any sequence longer than ~62 chars; at eval, we
    ask for sequences of 100+ a's. Pure length generalization.
    """
    return "*" * n + ":" + "a" * n + "\n"


class CountingDataset:
    """Generates 'count to N' examples on the fly. Pads to max_seq_len."""

    def __init__(self, n_min: int, n_max: int, max_seq_len: int,
                 device: str = "cpu", seed: int = 0):
        self.n_min = n_min
        self.n_max = n_max
        self.max_seq_len = max_seq_len
        self.device = device
        self.rng = random.Random(seed)

    def _example_bytes(self, n: int) -> list[int]:
        return list(make_count_example(n).encode("utf-8"))

    def get_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Sample N's, pack each into max_seq_len with padding
        L = self.max_seq_len
        # We use byte 0 as pad/ignore; loss masks it out.
        x = torch.zeros(batch_size, L, dtype=torch.long)
        y = torch.zeros(batch_size, L, dtype=torch.long)
        mask = torch.zeros(batch_size, L, dtype=torch.bool)
        for b in range(batch_size):
            n = self.rng.randint(self.n_min, self.n_max)
            tokens = self._example_bytes(n)
            # Truncate if too long
            tokens = tokens[: L + 1]
            if len(tokens) < 2:
                tokens = [0, 0]
            seq_in  = tokens[:-1]
            seq_out = tokens[1:]
            sl = min(len(seq_in), L)
            x[b, :sl] = torch.tensor(seq_in[:sl], dtype=torch.long)
            y[b, :sl] = torch.tensor(seq_out[:sl], dtype=torch.long)
            mask[b, :sl] = True
        return (x.to(self.device), y.to(self.device), mask.to(self.device))


# ----------------------------------------------------------------------------
# Phase 2: counter target derivation for auxiliary supervision
# ----------------------------------------------------------------------------

STAR_BYTE = ord("*")   # 42
A_BYTE    = ord("a")   # 97


def counter_targets(tokens: torch.Tensor) -> torch.Tensor:
    """Per-position increment-gate targets, derived from the input bytes.

    Counter A (idx 0): increments on '*' bytes — counts the input length N.
    Counter B (idx 1): increments on 'a' bytes — counts the output a's.

    Both held by default (reset gate stays low). At end of input phase,
    A==N. During output, B grows from 1 to N. Match B==A signals the
    LM to emit '\\n'.

    tokens: (B, L) int64
    returns: (B, L, 2) float — target for sigmoid(inc_logits)
    """
    is_star = (tokens == STAR_BYTE).float()
    is_a    = (tokens == A_BYTE).float()
    return torch.stack([is_star, is_a], dim=-1)


# ----------------------------------------------------------------------------
# Eval
# ----------------------------------------------------------------------------

def parse_count_output(generated: str) -> int | None:
    """From a model output 'aaaa...a\n', return number of a's, or None on parse fail."""
    end = generated.find("\n")
    body = generated[:end] if end >= 0 else generated
    if not all(c == "a" for c in body):
        return None
    return len(body)


@torch.no_grad()
def eval_counting(model: CortexLM, ns: list[int], device: str,
                  print_samples: bool = False) -> dict[int, dict]:
    """Prompt with N stars + ':' and check the model emits exactly N a's, then \\n."""
    model.eval()
    results = {}
    for n in ns:
        prompt = "*" * n + ":"
        prompt_bytes = list(prompt.encode("utf-8"))
        max_new = n + 4   # N a's + newline + a little slack
        gen = model.generate_greedy(prompt_bytes, max_new=max_new, max_ctx=8192)
        try:
            text = bytes(gen).decode("utf-8", errors="replace")
        except Exception:
            text = ""
        parsed = parse_count_output(text)
        ok = (parsed == n)
        results[n] = {"ok": ok, "parsed": parsed, "text_head": text[:60]}
        if print_samples:
            status = "OK" if ok else "FAIL"
            head = text[:40] + ("…" if len(text) > 40 else "")
            print(f"  N={n:4d} [{status}] parsed={parsed} :: {head!r}")
    return results


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------

@dataclass
class TrainConfig:
    train_n_min: int = 1
    train_n_max: int = 30
    eval_ns: list = field(default_factory=lambda: [3, 15, 30, 50, 100, 200, 500])
    total_steps: int = 4000
    batch_size: int = 32
    lr: float = 1e-3
    warmup_steps: int = 200
    weight_decay: float = 0.05
    eval_interval: int = 500
    seed: int = 0
    # Phase 2: auxiliary supervision of counter increment gates.
    # When > 0 and the model has a counter, BCE_with_logits between
    # the counter's inc_logits and counter_targets(tokens) is added
    # to the main loss with this weight.
    lambda_aux: float = 0.0


def train_model(model: CortexLM, tcfg: TrainConfig, device: str,
                tag: str, ckpt_path: str):
    model.to(device).train()
    n_params = sum(p.numel() for p in model.parameters())
    dataset = CountingDataset(
        n_min=tcfg.train_n_min,
        n_max=tcfg.train_n_max,
        max_seq_len=model.cfg.max_seq_len,
        device=device,
        seed=tcfg.seed,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr,
                            weight_decay=tcfg.weight_decay, betas=(0.9, 0.95))

    def lr_at(step):
        if step < tcfg.warmup_steps:
            return step / max(1, tcfg.warmup_steps)
        progress = (step - tcfg.warmup_steps) / max(1, tcfg.total_steps - tcfg.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    use_aux = tcfg.lambda_aux > 0.0 and len(model.primitives) > 0

    print(f"\n=== Training [{tag}] — {n_params:,} params"
          f"{f' | aux λ={tcfg.lambda_aux}' if use_aux else ''} ===")
    t0 = time.time()
    for step in range(1, tcfg.total_steps + 1):
        x, y, mask = dataset.get_batch(tcfg.batch_size)

        if use_aux:
            logits, prim_outputs = model(x, return_aux=True)
        else:
            logits = model(x)
            prim_outputs = None

        loss_per = F.cross_entropy(
            logits.reshape(-1, model.cfg.vocab_size),
            y.reshape(-1),
            reduction="none",
        ).view_as(y)
        main_loss = (loss_per * mask.float()).sum() / mask.float().sum().clamp_min(1.0)

        if use_aux:
            aux_loss = model.aux_loss(x, prim_outputs, mask)
            loss = main_loss + tcfg.lambda_aux * aux_loss
        else:
            aux_loss = None
            loss = main_loss

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % tcfg.eval_interval == 0 or step == 1:
            elapsed = time.time() - t0
            lr_now = sched.get_last_lr()[0]
            extra = f"  aux={aux_loss.item():.4f}" if aux_loss is not None else ""
            print(f"  step {step:5d}/{tcfg.total_steps}  "
                  f"loss={main_loss.item():.4f}{extra}  "
                  f"lr={lr_now:.2e}  elapsed={elapsed:.1f}s", flush=True)

    # Save
    torch.save({"model": model.state_dict(), "cfg": model.cfg}, ckpt_path)
    print(f"  saved -> {ckpt_path}")


# ----------------------------------------------------------------------------
# Entry points
# ----------------------------------------------------------------------------

CKPT_DIR = Path("checkpoints/cortex")


def cmd_train():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    tcfg = TrainConfig()
    torch.manual_seed(tcfg.seed)

    # Two models: identical except for `use_counter`
    base_cfg = CortexLMConfig(use_counter=False)
    cortex_cfg = CortexLMConfig(use_counter=True)

    base = CortexLM(base_cfg)
    cortex = CortexLM(cortex_cfg)

    train_model(base, tcfg, device, "baseline",
                str(CKPT_DIR / "baseline.pt"))
    train_model(cortex, tcfg, device, "cortex",
                str(CKPT_DIR / "cortex.pt"))


def cmd_train_phase2():
    """Phase 2: train cortex (with counter) under auxiliary gate supervision."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    tcfg = TrainConfig(lambda_aux=0.5)   # Phase 2 aux weight
    torch.manual_seed(tcfg.seed)

    cortex_cfg = CortexLMConfig(use_counter=True)
    cortex = CortexLM(cortex_cfg)
    train_model(cortex, tcfg, device, "cortex+aux",
                str(CKPT_DIR / "cortex_aux.pt"))


def cmd_train_phase5():
    """Phase 5: revert to v3.2 (residual-only) but amplify injection by 3×.

    Phase 4 diagnosis: adding the head_bias path *hurt* OOD scaling because
    it gave the model a side channel for emit-a-vs-newline, leaving the
    residual-injection path under-trained. The thesis is sound but the
    counter signal needs more residual authority. v3.2 = residual-only
    scaled to 183/500 OOD; multiplying read_proj output by 3 should let
    it overpower the SSM's positional-OOD noise further.
    """
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    tcfg = TrainConfig(lambda_aux=0.5)
    torch.manual_seed(tcfg.seed)

    cortex_cfg = CortexLMConfig(
        use_counter=True,
        counter_readout="unbounded",
        counter_head_bias=False,
        counter_injection_scale=3.0,
    )
    cortex = CortexLM(cortex_cfg)
    train_model(cortex, tcfg, device, "cortex+aux+unbounded+scale3",
                str(CKPT_DIR / "cortex_aux_v5.pt"))


def cmd_train_phase6():
    """Phase 6: same architecture, push injection scale to 10×.

    Phase 5 (scale=3) hit 371/500 at N=500 — the trend suggests linear
    capacity scaling with injection magnitude (1× → 183, 3× → 371).
    Pushing to 10× should put OOD capacity well above N=500, which means
    the cap becomes N itself: byte-perfect at every tested OOD length.
    """
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    tcfg = TrainConfig(lambda_aux=0.5)
    torch.manual_seed(tcfg.seed)

    cortex_cfg = CortexLMConfig(
        use_counter=True,
        counter_readout="unbounded",
        counter_head_bias=False,
        counter_injection_scale=10.0,
    )
    cortex = CortexLM(cortex_cfg)
    train_model(cortex, tcfg, device, "cortex+aux+unbounded+scale10",
                str(CKPT_DIR / "cortex_aux_v6.pt"))


def cmd_train_phase4():
    """Phase 4: tanh-only readout + direct head-logit bias path.

    Phase 3.2 diagnosis: cortex+aux+unbounded scaled OOD with N (good!) but
    voluntarily stopped early (bad — got 183 a's at N=500). The SSM is
    processing positions 1000+ which are far OOD from the max-80 training
    seqlen, and emits noisy '\\n' logits the residual-injected counter
    can't outvote through the tied embedding head. Phase 4 adds a direct
    Linear(read_in → vocab) from counter readout to LM logits — bypasses
    SSM/embedding-head bottleneck so the counter can dictate emit-a-vs-\\n
    regardless of what the SSM is doing.
    """
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    tcfg = TrainConfig(lambda_aux=0.5)
    torch.manual_seed(tcfg.seed)

    cortex_cfg = CortexLMConfig(
        use_counter=True,
        counter_readout="unbounded",
        counter_head_bias=True,
    )
    cortex = CortexLM(cortex_cfg)
    train_model(cortex, tcfg, device, "cortex+aux+unbounded+headbias",
                str(CKPT_DIR / "cortex_aux_v4.pt"))


def cmd_train_phase3():
    """Phase 3: same aux supervision as Phase 2, but unbounded counter readout.

    Phase 2 diagnosis: cortex+aux floored output at ~30 a's for every OOD N
    because the sinusoidal readout was trained only on counter values 0..30,
    and wraps unpredictably above that. Swap for a scale-invariant readout
    (raw + tanh + diff + tanh(diff)) and rerun.
    """
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    tcfg = TrainConfig(lambda_aux=0.5)
    torch.manual_seed(tcfg.seed)

    cortex_cfg = CortexLMConfig(use_counter=True, counter_readout="unbounded")
    cortex = CortexLM(cortex_cfg)
    train_model(cortex, tcfg, device, "cortex+aux+unbounded",
                str(CKPT_DIR / "cortex_aux_v3.pt"))


def _load(model_cls, cfg, path, device):
    """Load a checkpoint, migrating pre-refactor 'counter.*' keys to 'primitives.0.*'."""
    m = model_cls(cfg).to(device)
    sd = torch.load(path, map_location=device, weights_only=False)
    state_dict = sd["model"]
    # State-dict migration: pre-refactor checkpoints stored the counter at
    # `counter.*`. After the plugin refactor it lives at `primitives.0.*`.
    migrated = {}
    for k, v in state_dict.items():
        if k.startswith("counter."):
            migrated["primitives.0." + k[len("counter."):]] = v
        else:
            migrated[k] = v
    m.load_state_dict(migrated)
    m.eval()
    return m


def cmd_eval():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tcfg = TrainConfig()

    base_cfg = CortexLMConfig(use_counter=False)
    cortex_cfg = CortexLMConfig(use_counter=True)
    cortex_v3_cfg = CortexLMConfig(use_counter=True, counter_readout="unbounded")
    cortex_v4_cfg = CortexLMConfig(
        use_counter=True, counter_readout="unbounded", counter_head_bias=True,
    )
    cortex_v5_cfg = CortexLMConfig(
        use_counter=True, counter_readout="unbounded", counter_injection_scale=3.0,
    )
    cortex_v6_cfg = CortexLMConfig(
        use_counter=True, counter_readout="unbounded", counter_injection_scale=10.0,
    )

    base   = _load(CortexLM, base_cfg,   str(CKPT_DIR / "baseline.pt"),  device)
    cortex = _load(CortexLM, cortex_cfg, str(CKPT_DIR / "cortex.pt"),    device)
    aux_path = CKPT_DIR / "cortex_aux.pt"
    v3_path  = CKPT_DIR / "cortex_aux_v3.pt"
    v4_path  = CKPT_DIR / "cortex_aux_v4.pt"
    v5_path  = CKPT_DIR / "cortex_aux_v5.pt"
    v6_path  = CKPT_DIR / "cortex_aux_v6.pt"
    cortex_aux = _load(CortexLM, cortex_cfg, str(aux_path), device) if aux_path.exists() else None
    cortex_v3  = _load(CortexLM, cortex_v3_cfg, str(v3_path), device) if v3_path.exists() else None
    cortex_v4  = _load(CortexLM, cortex_v4_cfg, str(v4_path), device) if v4_path.exists() else None
    cortex_v5  = _load(CortexLM, cortex_v5_cfg, str(v5_path), device) if v5_path.exists() else None
    cortex_v6  = _load(CortexLM, cortex_v6_cfg, str(v6_path), device) if v6_path.exists() else None

    print("\n=== Length-generalization eval ===")
    print(f"Trained on N ∈ [{tcfg.train_n_min}, {tcfg.train_n_max}]\n")

    print("--- Baseline (no counter primitive) ---")
    base_results = eval_counting(base, tcfg.eval_ns, device, print_samples=True)

    print("\n--- Cortex (counter, no supervision) ---")
    cortex_results = eval_counting(cortex, tcfg.eval_ns, device, print_samples=True)

    aux_results = None
    if cortex_aux is not None:
        print("\n--- Cortex+aux (sinusoidal readout, Phase 2) ---")
        aux_results = eval_counting(cortex_aux, tcfg.eval_ns, device, print_samples=True)

    v3_results = None
    if cortex_v3 is not None:
        print("\n--- Cortex+aux+unbounded (Phase 3) ---")
        v3_results = eval_counting(cortex_v3, tcfg.eval_ns, device, print_samples=True)

    v4_results = None
    if cortex_v4 is not None:
        print("\n--- Cortex+aux+unbounded+headbias (Phase 4) ---")
        v4_results = eval_counting(cortex_v4, tcfg.eval_ns, device, print_samples=True)

    v5_results = None
    if cortex_v5 is not None:
        print("\n--- Cortex+aux+unbounded+scale3 (Phase 5) ---")
        v5_results = eval_counting(cortex_v5, tcfg.eval_ns, device, print_samples=True)

    v6_results = None
    if cortex_v6 is not None:
        print("\n--- Cortex+aux+unbounded+scale10 (Phase 6) ---")
        v6_results = eval_counting(cortex_v6, tcfg.eval_ns, device, print_samples=True)

    print("\n=== Summary ===")
    cols = [("baseline", base_results), ("cortex", cortex_results)]
    if aux_results is not None:
        cols.append(("cortex+aux", aux_results))
    if v3_results is not None:
        cols.append(("v3:unbounded", v3_results))
    if v4_results is not None:
        cols.append(("v4:headbias", v4_results))
    if v5_results is not None:
        cols.append(("v5:scale3", v5_results))
    if v6_results is not None:
        cols.append(("v6:scale10", v6_results))

    header = f"{'N':>5}  " + "  ".join(f"{name:>13}" for name, _ in cols)
    print(header)
    for n in tcfg.eval_ns:
        cells = []
        for _, res in cols:
            r = res[n]
            cells.append("OK" if r["ok"] else f"FAIL({r['parsed']})")
        marker = " <-- OOD" if n > tcfg.train_n_max else ""
        print(f"{n:>5}  " + "  ".join(f"{c:>13}" for c in cells) + marker)


def cmd_demo():
    """Side-by-side: baseline LM vs cortex LM (counter + scale=10 + hard gates).

    Both are byte-level Mamba-3 LMs (~151k params), trained on the same
    unary count-to-N corpus with N ∈ [1, 30]. The only difference is whether
    a 772-param CounterPrimitive is wired into the residual stream.

    Format: prompt is "*"*N + ":". Model is asked to emit exactly N a's
    followed by '\\n'. We show the first N+8 generated bytes for each N.
    """
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    base_cfg = CortexLMConfig(use_counter=False)
    cortex_v6_cfg = CortexLMConfig(
        use_counter=True, counter_readout="unbounded", counter_injection_scale=10.0,
    )
    base   = _load(CortexLM, base_cfg, str(CKPT_DIR / "baseline.pt"), device)
    cortex = _load(CortexLM, cortex_v6_cfg, str(CKPT_DIR / "cortex_aux_v6.pt"), device)
    cortex.counter.hard_gates_inference = True   # the existence-proof config

    print("=" * 72)
    print("Both models: byte-level Mamba-3 LM, ~151k params, trained on N∈[1,30]")
    print("Cortex adds: 772-param counter primitive in residual stream")
    print("=" * 72)

    def show(model, tag, n):
        prompt = "*" * n + ":"
        gen = model.generate_greedy(list(prompt.encode("utf-8")),
                                    max_new=n + 8, max_ctx=8192)
        body = bytes(gen).decode("utf-8", errors="replace")
        end = body.find("\n")
        emitted = body[:end] if end >= 0 else body
        n_a = sum(1 for c in emitted if c == "a")
        ok = (n_a == n) and all(c == "a" for c in emitted)
        status = "✓" if ok else "✗"
        # Truncate long outputs for display
        preview = emitted if len(emitted) <= 60 else f"{emitted[:30]}…(len {len(emitted)})"
        print(f"  {tag:>8} {status}  emitted {n_a:>3} a's  ::  {preview!r}")

    for n in [3, 10, 30, 50, 100, 200, 500]:
        marker = "  (training distribution)" if n <= 30 else "  (OOD)"
        print(f"\nPrompt: '{'*' * min(n, 12)}{'…' if n > 12 else ''}:' (N={n}){marker}")
        print(f"  Expected: {n} a's then newline")
        show(base,   "baseline", n)
        show(cortex, "cortex",   n)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    if cmd == "train":
        cmd_train()
    elif cmd == "train_phase2":
        cmd_train_phase2()
    elif cmd == "train_phase3":
        cmd_train_phase3()
    elif cmd == "train_phase4":
        cmd_train_phase4()
    elif cmd == "train_phase5":
        cmd_train_phase5()
    elif cmd == "train_phase6":
        cmd_train_phase6()
    elif cmd == "eval":
        cmd_eval()
    elif cmd == "demo":
        cmd_demo()
    else:
        print(f"unknown command: {cmd}")
        sys.exit(1)
