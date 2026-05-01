"""train_tool_renderer_copy — Pointer-Networks-style copy mechanism on top of Mamba-3.

The previous renderer (`train_tool_renderer.py`) emits templates with named
placeholders ($N, $OPTIMAL, ...) and lets the orchestrator do deterministic
substitution. That works, but it requires a hand-written SLOT_MAP per tool;
the LM never learns to copy.

This file builds the proper ML answer: a **copy mechanism** that lets the LM
itself decide, at every output position, whether to *generate* a token from
the vocabulary or *copy* a byte from a specific position in the prefix. The
mechanism is trained end-to-end on the *concrete* (non-templated) targets,
i.e. with real digits in the answer.

Architecture (CopyMamba3LM):
  - Standard Mamba-3 stack runs over the full sequence (prefix + answer).
  - At each output position, the model produces three things:
      1. p_vocab : a vocab distribution from the existing LM head.
      2. attn   : an attention distribution over PREFIX positions, computed
                  by attending from the current hidden state (query) to all
                  prefix hidden states (keys). This is the "where to copy from."
      3. gate   : a scalar in [0, 1] = sigmoid(g_head(h_t)). Probability of
                  generating from vocab vs copying.
  - Final next-token distribution at position t:
        p(b) = gate * p_vocab(b) + (1 - gate) * p_copy(b)
    where p_copy aggregates the attention distribution over prefix positions
    by adding mass to whichever byte each prefix position holds.
  - Loss: standard NLL on p(b) for the target byte at every answer position.

The model learns *when* to copy (the gate goes near 0 at digit positions)
and *where to point* (attn peaks on the right prefix byte). With a tiny
SSM (74k params) this should solve the digit-copy problem cleanly.

Inference:
  - At each step, sample / argmax from the mixture distribution.
  - Optionally inspect attn and gate for trace output: "at this byte, gate=0.04
    and pointed at prefix position 18 (the '4' of 'optimal=4095')."

Training format and special tokens are the same as train_tool_renderer.py:
  <payload>\\x01<concrete_sentence>\\x02
"""
from __future__ import annotations
import _path_shim  # noqa: F401  (adds repo root to sys.path)
import argparse, math, random, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

from lab_platform.mamba3_lm import Mamba3LM, LMConfig
from lab_platform.mamba3_minimal import Mamba3Block, Mamba3Config


PAD = 0
BOA = 1
EOS = 2
MAX_LEN = 256
VOCAB = 256


# ----------------------------------------------------------- synthetic data ---
# Concrete (non-templated) targets — back to the original phrasings with real
# digits. The point of this experiment is the LM doing its own copy, not
# having the orchestrator do it.


def hanoi_payload_and_sentence(rng: random.Random) -> tuple[str, str]:
    n = rng.randint(2, 23)
    optimal = (1 << n) - 1
    params = 45318
    timing = rng.randint(50, 5000)
    payload = f"hanoi_solver|n={n}|optimal={optimal}|params={params}|timing={timing}"
    sentence = f"The optimal solution to Tower of Hanoi with {n} disks requires {optimal:,} moves."
    return payload, sentence


def gcd_payload_and_sentence(rng: random.Random) -> tuple[str, str]:
    a = rng.randint(2, 9999)
    b = rng.randint(2, 9999)
    g = math.gcd(a, b)
    payload = f"gcd|a={a}|b={b}|gcd={g}"
    sentence = f"The greatest common divisor of {a} and {b} is {g}."
    return payload, sentence


def gcdhanoi_payload_and_sentence(rng: random.Random) -> tuple[str, str]:
    a = rng.randint(2, 18)
    b = rng.randint(2, 18)
    ma = (1 << a) - 1
    mb = (1 << b) - 1
    g = math.gcd(ma, mb)
    payload = f"gcdhanoi|a={a}|b={b}|moves_a={ma}|moves_b={mb}|gcd={g}"
    sentence = f"Hanoi({a}) needs {ma:,} moves; Hanoi({b}) needs {mb:,} moves; their gcd is {g}."
    return payload, sentence


GENERATORS = [hanoi_payload_and_sentence, gcd_payload_and_sentence, gcdhanoi_payload_and_sentence]


def gen_example(rng: random.Random) -> tuple[bytes, bytes]:
    payload, sentence = rng.choice(GENERATORS)(rng)
    return payload.encode("utf-8"), sentence.encode("utf-8")


def encode_pair(payload_b: bytes, sentence_b: bytes, max_len: int = MAX_LEN) -> tuple[torch.Tensor, int]:
    """Returns (token_tensor, prefix_len_including_BOA)."""
    prefix_tokens = [b for b in payload_b] + [BOA]
    answer_tokens = [b for b in sentence_b] + [EOS]
    tokens = prefix_tokens + answer_tokens
    tokens = tokens[:max_len]
    while len(tokens) < max_len:
        tokens.append(PAD)
    return torch.tensor(tokens, dtype=torch.long), len(prefix_tokens)


# ------------------------------------------------------- copy-aware model ---


class CopyMamba3LM(nn.Module):
    """Mamba-3 LM augmented with a Pointer-Networks-style copy mechanism."""

    def __init__(self, n_layers: int = 2, d_model: int = 64, attn_dim: int = 32):
        super().__init__()
        self.d_model = d_model
        self.attn_dim = attn_dim
        ssm_cfg = Mamba3Config(d_model=d_model)
        self.embed = nn.Embedding(VOCAB, d_model, padding_idx=PAD)
        self.embed_norm = nn.LayerNorm(d_model)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                "norm": nn.LayerNorm(d_model),
                "block": Mamba3Block(ssm_cfg),
            }))
        self.final_norm = nn.LayerNorm(d_model)
        # Vocab head (with weight tying to embedding)
        self.vocab_head = nn.Linear(d_model, VOCAB, bias=False)
        self.vocab_head.weight = self.embed.weight
        # Copy attention: query from output position, key from prefix position
        self.q_proj = nn.Linear(d_model, attn_dim)
        self.k_proj = nn.Linear(d_model, attn_dim)
        # Gate head: scalar in [0, 1], 1 = generate from vocab, 0 = copy.
        # Bias init to +2.0 so initial gate ≈ sigmoid(2) ≈ 0.88 (mostly vocab).
        # The previous run got stuck at val_loss ~0.31 with gate near 0 on
        # language bytes — degenerate "copy everything that's available in
        # the prefix" mode. Biasing toward vocab at init breaks that local
        # minimum: the model learns to use copy as a deliberate deviation
        # for digits, not as the default for any matching byte.
        self.gate_head = nn.Linear(d_model, 1)
        with torch.no_grad():
            self.gate_head.bias.fill_(2.0)

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Run the SSM stack over the full sequence, return hidden states."""
        x = self.embed(tokens)
        x = self.embed_norm(x)
        for layer in self.layers:
            residual = x
            x = layer["norm"](x)
            x = layer["block"](x)
            x = residual + x
        x = self.final_norm(x)
        return x  # (B, L, d)

    def forward(self, tokens: torch.Tensor, prefix_lens: torch.Tensor):
        """tokens: (B, L) padded sequence. prefix_lens: (B,) length of prefix
        (payload + BOA) per row.
        Returns log p(next_token) at every position (B, L, V), to be used with
        a shift-by-one loss on answer positions only."""
        B, L = tokens.shape
        h = self.encode(tokens)  # (B, L, d)

        # Vocab distribution from the LM head
        vocab_logits = self.vocab_head(h)  # (B, L, V)

        # Copy attention: at each output position t, compute attention over
        # *prefix* positions of the same row. Mask non-prefix positions.
        q = self.q_proj(h)  # (B, L, attn_dim)
        k = self.k_proj(h)  # (B, L, attn_dim)
        # Score: dot product between q at every position and k at every position
        scores = torch.einsum("btd,bsd->bts", q, k) / (self.attn_dim ** 0.5)  # (B, L_q=L, L_k=L)

        # Mask: prefix positions only. prefix_mask[b, s] = 1 if s < prefix_lens[b].
        idx = torch.arange(L, device=tokens.device).unsqueeze(0)  # (1, L)
        prefix_mask = (idx < prefix_lens.unsqueeze(1))  # (B, L)
        # Also exclude pad tokens within prefix (shouldn't be any, but be safe)
        prefix_mask = prefix_mask & (tokens != PAD)
        # Broadcast over query positions
        attn_mask = prefix_mask.unsqueeze(1).expand(B, L, L)  # (B, L_q, L_k)
        scores = scores.masked_fill(~attn_mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)  # (B, L_q, L_k)

        # Aggregate copy-distribution: for each query position, attn distributes
        # mass over prefix positions. Each prefix position holds a specific byte.
        # Sum mass for each vocab byte.
        # Build a (B, L_k, V) one-hot of prefix tokens, then einsum.
        prefix_onehot = F.one_hot(tokens, num_classes=VOCAB).float()  # (B, L_k, V)
        # Zero out non-prefix positions in the one-hot
        prefix_onehot = prefix_onehot * prefix_mask.unsqueeze(-1).float()
        copy_dist = torch.einsum("bts,bsv->btv", attn, prefix_onehot)  # (B, L_q, V)

        # Gate at each position
        gate = torch.sigmoid(self.gate_head(h))  # (B, L, 1)

        # Mix: convert vocab_logits to probabilities, blend
        vocab_probs = F.softmax(vocab_logits, dim=-1)
        mixed = gate * vocab_probs + (1 - gate) * copy_dist  # (B, L, V)
        mixed = mixed.clamp(min=1e-12)
        return torch.log(mixed), gate.squeeze(-1), attn  # log_probs (B, L, V), gate (B, L), attn (B, L, L)

    @torch.no_grad()
    def generate(self, prefix_bytes: list[int], max_new: int = 200,
                 temperature: float = 0.1, top_k: int = 1, return_trace: bool = False):
        """AR generation. Re-encodes the whole sequence each step (small model;
        fine for demos). Returns generated bytes; optionally a trace of (gate,
        argmax_attn_position) per step.
        """
        device = next(self.parameters()).device
        prefix_len = len(prefix_bytes)
        tokens = torch.tensor([prefix_bytes], dtype=torch.long, device=device)
        prefix_lens = torch.tensor([prefix_len], dtype=torch.long, device=device)
        trace = []
        out = []
        for _ in range(max_new):
            log_probs, gate, attn = self.forward(tokens, prefix_lens)
            t = tokens.shape[1] - 1
            logits = log_probs[0, t] / max(temperature, 1e-6)
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = torch.where(logits < v[-1], torch.full_like(logits, float("-inf")), logits)
            probs = torch.softmax(logits, dim=-1)
            nxt = int(torch.argmax(probs).item()) if top_k == 1 else int(torch.multinomial(probs, 1).item())
            if return_trace:
                ap = int(torch.argmax(attn[0, t]).item())
                trace.append({"byte": nxt, "char": chr(nxt) if 32 <= nxt < 127 else "?",
                              "gate": float(gate[0, t]), "attn_argmax": ap,
                              "attn_byte": chr(int(tokens[0, ap])) if ap < prefix_len else "?"})
            out.append(nxt)
            tokens = torch.cat([tokens, torch.tensor([[nxt]], device=device)], dim=1)
            if nxt == EOS:
                break
        return (out, trace) if return_trace else out


# ------------------------------------------------------------- training ---


def train(steps: int, batch: int, lr: float, device: str, seed: int = 42,
          val_size: int = 256, save_to: str = "checkpoints/tool_renderer_copy.pt"):
    rng = random.Random(seed)
    torch.manual_seed(seed)

    model = CopyMamba3LM().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"CopyMamba3LM params: {n_params}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.05)

    val_pairs = [gen_example(rng) for _ in range(val_size)]
    val_tokens = torch.stack([encode_pair(p, s)[0] for p, s in val_pairs]).to(device)
    val_prefix_lens = torch.tensor([encode_pair(p, s)[1] for p, s in val_pairs]).to(device)

    t0 = time.time()
    best_loss = float("inf")
    best_state = None

    for step in range(steps):
        examples = [gen_example(rng) for _ in range(batch)]
        tokens_list = [encode_pair(p, s) for p, s in examples]
        tokens = torch.stack([t[0] for t in tokens_list]).to(device)
        prefix_lens = torch.tensor([t[1] for t in tokens_list]).to(device)

        log_probs, gate, attn = model(tokens, prefix_lens)

        # Shift: predict tokens[:, 1:] from positions tokens[:, :-1]
        # Loss is computed only on answer positions (>= prefix_len, < EOS+1, not PAD)
        L = tokens.shape[1]
        idx = torch.arange(L, device=device).unsqueeze(0)
        # Position t predicts tokens[t+1]; only train when position t is in answer-region:
        #   prefix_len - 1 <= t (so that the BOA itself drives the first answer byte)
        #   tokens[t+1] != PAD (target exists)
        ans_pos_mask = (idx >= (prefix_lens.unsqueeze(1) - 1))  # (B, L)
        ans_pos_mask = ans_pos_mask[:, :-1]  # (B, L-1)
        target = tokens[:, 1:]  # (B, L-1)
        ans_pos_mask = ans_pos_mask & (target != PAD)

        # Gather log_probs at the target token
        lp = log_probs[:, :-1]  # (B, L-1, V)
        # NLL = -log p(target). Use gather.
        nll = -lp.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # (B, L-1)
        loss = nll[ans_pos_mask].mean()

        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        if (step + 1) % 100 == 0:
            with torch.no_grad():
                vlog_probs, _, _ = model(val_tokens, val_prefix_lens)
                vL = val_tokens.shape[1]
                vidx = torch.arange(vL, device=device).unsqueeze(0)
                v_ans_mask = (vidx >= (val_prefix_lens.unsqueeze(1) - 1))[:, :-1]
                v_target = val_tokens[:, 1:]
                v_ans_mask = v_ans_mask & (v_target != PAD)
                v_lp = vlog_probs[:, :-1]
                v_nll = -v_lp.gather(-1, v_target.unsqueeze(-1)).squeeze(-1)
                v_loss = v_nll[v_ans_mask].mean().item()
            if v_loss < best_loss:
                best_loss = v_loss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                Path(save_to).parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "state_dict": best_state,
                    "config": {"n_layers": 2, "d_model": 64, "attn_dim": 32},
                    "max_len": MAX_LEN,
                    "best_val_loss": best_loss,
                    "tokens": {"PAD": PAD, "BOA": BOA, "EOS": EOS},
                }, save_to)
            print(f"  step {step+1}/{steps}  train_loss={loss.item():.4f}  "
                  f"val_loss={v_loss:.4f}  best_val={best_loss:.4f}  "
                  f"elapsed={time.time()-t0:.0f}s")

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"\nSaved → {save_to}  (best val loss = {best_loss:.4f})")

    # Sample with trace
    print("\nSamples (with copy trace at digit positions):")
    rng2 = random.Random(seed + 1000)
    for _ in range(3):
        payload, _ = rng2.choice(GENERATORS)(rng2)
        prefix = list(payload.encode("utf-8")) + [BOA]
        gen, trace = model.generate(prefix, max_new=120, temperature=0.1, top_k=1, return_trace=True)
        if EOS in gen:
            gen = gen[:gen.index(EOS)]
        text = bytes([b for b in gen if 32 <= b < 256]).decode("utf-8", errors="ignore")
        print(f"  payload: {payload}")
        print(f"  → {text}")
        # Show low-gate (copy) positions
        copies = [(i, t) for i, t in enumerate(trace[:len(gen)]) if t["gate"] < 0.5]
        if copies:
            print(f"  copy events (gate<0.5):")
            for i, t in copies[:8]:
                print(f"    pos {i}: char='{t['char']}' gate={t['gate']:.3f} pointed at prefix idx {t['attn_argmax']} ('{t['attn_byte']}')")
        print()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--save-to", default="checkpoints/tool_renderer_copy.pt")
    args = ap.parse_args()
    print(f"Device: {args.device}")
    train(args.steps, args.batch, args.lr, args.device, save_to=args.save_to)


if __name__ == "__main__":
    main()
