"""
Mamba-3 Language Model — stacked blocks for character-level bilingual generation.

Architecture:
  - Character embedding (vocab ~256 + special tokens)
  - N stacked Mamba3Blocks with residual connections + LayerNorm
  - Linear head → next-character prediction

This is our first multi-layer Mamba-3 model, moving from toy tasks to
actual language modeling on English + Spanish text.
"""
from __future__ import annotations
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mamba3_minimal import Mamba3Block, Mamba3Config


@dataclass
class LMConfig:
    # Model
    n_layers: int = 2
    d_model: int = 128
    d_state: int = 16
    expand: int = 2
    headdim: int = 16
    # SSM
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4
    A_floor: float = 1e-4
    # Training
    vocab_size: int = 256      # byte-level
    max_seq_len: int = 64
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.1
    warmup_steps: int = 100
    total_steps: int = 5000
    eval_interval: int = 100
    # Generation
    temperature: float = 0.8
    top_k: int = 40


class Mamba3LM(nn.Module):
    def __init__(self, cfg: LMConfig):
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.embed_norm = nn.LayerNorm(cfg.d_model)

        ssm_cfg = Mamba3Config(
            d_model=cfg.d_model,
            d_state=cfg.d_state,
            expand=cfg.expand,
            headdim=cfg.headdim,
            dt_min=cfg.dt_min,
            dt_max=cfg.dt_max,
            dt_init_floor=cfg.dt_init_floor,
            A_floor=cfg.A_floor,
        )

        self.layers = nn.ModuleList()
        for _ in range(cfg.n_layers):
            self.layers.append(nn.ModuleDict({
                "norm": nn.LayerNorm(cfg.d_model),
                "block": Mamba3Block(ssm_cfg),
            }))

        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.embed.weight

        self._init_weights()
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Mamba3LM: {cfg.n_layers} layers, {n_params:,} params")

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() >= 2:
                nn.init.normal_(p, std=0.02)
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: (B, L) long tensor of byte values
        returns: (B, L, vocab_size) logits
        """
        x = self.embed(tokens)
        x = self.embed_norm(x)

        for layer in self.layers:
            residual = x
            x = layer["norm"](x)
            if self.training and torch.is_grad_enabled():
                x = torch.utils.checkpoint.checkpoint(
                    layer["block"], x, use_reentrant=False
                )
            else:
                x = layer["block"](x)
            x = residual + x

        x = self.final_norm(x)
        logits = self.head(x)
        return logits

    @torch.no_grad()
    def generate(self, prompt_bytes: list[int], max_new: int = 200,
                 temperature: float = 0.8, top_k: int = 40) -> list[int]:
        """Autoregressive generation from a byte-level prompt."""
        device = next(self.parameters()).device
        tokens = torch.tensor([prompt_bytes], dtype=torch.long, device=device)

        for _ in range(max_new):
            # Use last max_seq_len tokens for context
            ctx = tokens[:, -self.cfg.max_seq_len:]
            logits = self(ctx)[:, -1]  # (1, vocab)
            logits = logits / temperature

            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            tokens = torch.cat([tokens, next_token], dim=1)

        return tokens[0].tolist()[len(prompt_bytes):]


class TextDataset:
    """Byte-level text dataset — streams random chunks from a text file."""
    def __init__(self, path: str, seq_len: int, device: str = "cpu"):
        with open(path, "rb") as f:
            self.data = f.read()
        self.data = torch.tensor(list(self.data), dtype=torch.long)
        self.seq_len = seq_len
        self.device = device
        print(f"TextDataset: {len(self.data):,} bytes from {path}")

    def get_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Random batch of (input, target) pairs, each of length seq_len."""
        max_start = len(self.data) - self.seq_len - 1
        starts = torch.randint(0, max_start, (batch_size,))
        x = torch.stack([self.data[s : s + self.seq_len] for s in starts])
        y = torch.stack([self.data[s + 1 : s + self.seq_len + 1] for s in starts])
        return x.to(self.device), y.to(self.device)


def train(cfg: LMConfig, data_path: str, device: str = "mps"):
    dataset = TextDataset(data_path, cfg.max_seq_len, device)
    model = Mamba3LM(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)

    # Linear warmup then cosine decay
    def lr_schedule(step):
        if step < cfg.warmup_steps:
            return step / cfg.warmup_steps
        progress = (step - cfg.warmup_steps) / max(1, cfg.total_steps - cfg.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_schedule)

    best_loss = float("inf")
    for step in range(1, cfg.total_steps + 1):
        x, y = dataset.get_batch(cfg.batch_size)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1))

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()

        if step % cfg.eval_interval == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                ex, ey = dataset.get_batch(cfg.batch_size * 4)
                elogits = model(ex)
                eval_loss = F.cross_entropy(elogits.view(-1, cfg.vocab_size), ey.view(-1))
            model.train()

            bpc = eval_loss.item() / math.log(2)
            best_loss = min(best_loss, eval_loss.item())
            lr_now = scheduler.get_last_lr()[0]
            print(f"step {step:5d}  loss={eval_loss.item():.3f}  bpc={bpc:.3f}  lr={lr_now:.2e}", flush=True)

            # Generate a sample
            prompts = ["The ", "El ", "Hello ", "Hola "]
            prompt = prompts[(step // cfg.eval_interval) % len(prompts)]
            prompt_bytes = list(prompt.encode("utf-8"))
            gen = model.generate(prompt_bytes, max_new=100,
                                 temperature=cfg.temperature, top_k=cfg.top_k)
            try:
                text = bytes(gen).decode("utf-8", errors="replace")
            except Exception:
                text = str(gen[:50])
            print(f"  prompt={prompt!r} → {text[:120]!r}", flush=True)
            print()

    # Save checkpoint
    ckpt_path = data_path.rsplit("/", 1)[0] + "/mamba3_lm.pt"
    torch.save({"model": model.state_dict(), "cfg": cfg}, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")
    return model


if __name__ == "__main__":
    import sys, os
    os.environ["PYTHONUNBUFFERED"] = "1"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/bilingual.txt"
    print(f"Device: {device}", flush=True)
    cfg = LMConfig()
    train(cfg, data_path, device)
