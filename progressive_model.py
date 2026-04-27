"""
Progressive-growing Mamba-3 with kernel/cortex split.

Starts tiny (1 layer), grows on demand. Byte-level vocab (260 tokens).
Kernel layers learn curriculum (reasoning), cortex layers learn language.
Shared embedding bridges both.

Usage:
    model = ProgressiveModel(d_model=64)
    model.add_kernel_layer()           # start with 1 kernel layer
    model.set_mode("kernel")           # freeze cortex, train kernel + embed
    # ... train on curriculum ...
    model.add_cortex_layer()           # grow when needed
    model.set_mode("cortex")           # freeze kernel, train cortex + embed
    # ... train on language ...
"""
import torch
import torch.nn as nn
from mamba3_minimal import Mamba3Block, Mamba3Config


# ── Byte-level tokenizer ────────────────────────────────────────────

BYTE_VOCAB = 256
BOS = 256
EOS = 257
SEP = 258
PAD = 259
VOCAB_SIZE = 260


class ByteTokenizer:
    """Encode strings as raw bytes. Universal — works for numbers,
    language, code, markdown. No learned components."""

    @staticmethod
    def encode(text: str) -> list[int]:
        return list(text.encode("utf-8"))

    @staticmethod
    def decode(ids: list[int]) -> str:
        return bytes(i for i in ids if i < BYTE_VOCAB).decode("utf-8", errors="replace")

    @staticmethod
    def encode_curriculum(example: dict) -> tuple[list[int], int]:
        """Encode a curriculum example as [BOS input SEP output EOS].
        Returns (token_ids, sep_position)."""
        inp_bytes = list(example["input"].encode("utf-8"))
        out_bytes = list(example["output"].encode("utf-8"))
        tokens = [BOS] + inp_bytes + [SEP] + out_bytes + [EOS]
        sep_pos = len(inp_bytes) + 1  # position of SEP
        return tokens, sep_pos

    @staticmethod
    def encode_text(text: str, seq_len: int = 64) -> list[int]:
        """Encode raw text as bytes, truncated to seq_len."""
        return list(text.encode("utf-8"))[:seq_len]


# ── Progressive model ───────────────────────────────────────────────

class OutputHistoryAttention(nn.Module):
    """A single causal attention head added as a side-channel to the SSM
    stack. At each timestep t, attend to all previous positions 0..t and
    add the result (scaled by a learnable mix factor) to the hidden state.

    Why this exists: the SSM scan blends information into the recurrent
    state with a decay factor; the model can't directly query "what was
    my hidden state at position k?" When generating multi-digit answers
    autoregressively, that's the difference between losing track of a
    carry vs. holding it. This module is the smallest architectural
    change that gives Mamba a copy/lookup primitive without abandoning
    the SSM scan.

    Init: mix=0.1 keeps the layer near-identity at start so the model
    learns to USE the new pathway rather than competing with it from
    step 1.
    """
    def __init__(self, d_model: int, d_attn: int = 32):
        super().__init__()
        import math
        self.q_proj = nn.Linear(d_model, d_attn)
        self.k_proj = nn.Linear(d_model, d_attn)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.scale = 1.0 / math.sqrt(d_attn)
        # Init out_proj near zero so the new pathway starts as a no-op;
        # the model has to learn its way into using it. Pairs with the
        # learnable mix factor to give a smooth gradient pathway.
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.out_proj.bias)
        self.mix = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        B, L, D = x.shape
        Q = self.q_proj(x)               # (B, L, d_attn)
        K = self.k_proj(x)               # (B, L, d_attn)
        V = self.v_proj(x)               # (B, L, D)
        scores = torch.einsum("bld,bmd->blm", Q, K) * self.scale
        # Causal mask: position l can only attend to positions ≤ l
        mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        out = torch.einsum("blm,bmd->bld", weights, V)
        out = self.out_proj(out)
        return self.mix * out


class ProgressiveModel(nn.Module):
    """
    Mamba-3 model that grows layer by layer.

    Architecture:
        embed(260) → [kernel layers] → [cortex layers] → [history attention] → head(260)

    Modes:
        "kernel" — train kernel layers + embed + head, freeze cortex
        "cortex" — train cortex layers + embed + head, freeze kernel
        "all"    — train everything
    """
    def __init__(self, d_model=64, d_state=16, expand=2, headdim=16,
                 use_history_attn: bool = False, history_d_attn: int = 32):
        super().__init__()
        self.d_model = d_model
        self.ssm_cfg = Mamba3Config(
            d_model=d_model, d_state=d_state,
            expand=expand, headdim=headdim,
        )

        # Shared embedding + head (weight-tied)
        self.embed = nn.Embedding(VOCAB_SIZE, d_model)
        self.embed_norm = nn.LayerNorm(d_model)
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, VOCAB_SIZE, bias=False)
        self.head.weight = self.embed.weight  # weight tying

        # Layer groups
        self.kernel_layers = nn.ModuleList()
        self.cortex_layers = nn.ModuleList()

        # Optional output-history attention (one layer after the SSM stack)
        self.use_history_attn = use_history_attn
        self.history_attn = (OutputHistoryAttention(d_model, d_attn=history_d_attn)
                             if use_history_attn else None)

        self.mode = "all"

    def _make_layer(self):
        """Create a near-identity SSM layer."""
        block = Mamba3Block(self.ssm_cfg)
        # Scale output projection small so new layer is near-identity
        with torch.no_grad():
            block.out_proj.weight.mul_(0.01)
        norm = nn.LayerNorm(self.d_model)
        scale = nn.Parameter(torch.tensor(0.01))
        return nn.ModuleDict({"block": block, "norm": norm, "scale": nn.ParameterList([scale])})

    def _get_device(self):
        """Get the device of the model."""
        return self.embed.weight.device

    def add_kernel_layer(self):
        layer = self._make_layer().to(self._get_device())
        self.kernel_layers.append(layer)
        n = len(self.kernel_layers)
        n_params = sum(p.numel() for p in layer.parameters())
        print(f"  + kernel layer {n} ({n_params:,} params)", flush=True)
        return n - 1

    def add_cortex_layer(self):
        layer = self._make_layer().to(self._get_device())
        self.cortex_layers.append(layer)
        n = len(self.cortex_layers)
        n_params = sum(p.numel() for p in layer.parameters())
        print(f"  + cortex layer {n} ({n_params:,} params)", flush=True)
        return n - 1

    def set_mode(self, mode: str):
        """Set training mode: 'kernel', 'cortex', or 'all'."""
        assert mode in ("kernel", "cortex", "all")
        self.mode = mode

        # Embed + head always trainable
        self.embed.requires_grad_(True)
        self.embed_norm.requires_grad_(True)
        self.final_norm.requires_grad_(True)
        # head shares weights with embed, no separate toggle

        # Kernel layers
        for layer in self.kernel_layers:
            layer.requires_grad_(mode in ("kernel", "all"))

        # Cortex layers
        for layer in self.cortex_layers:
            layer.requires_grad_(mode in ("cortex", "all"))

    def get_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, tokens):
        x = self.embed_norm(self.embed(tokens))

        for layer in self.kernel_layers:
            scale = layer["scale"][0]
            x = x + scale * layer["block"](layer["norm"](x))

        for layer in self.cortex_layers:
            scale = layer["scale"][0]
            x = x + scale * layer["block"](layer["norm"](x))

        # Output-history attention (causal): adds a copy/lookup primitive.
        # No-op if not configured. Lives after the SSM stack so attention
        # operates on the SSM's accumulated representation.
        if self.history_attn is not None:
            x = x + self.history_attn(x)

        x = self.final_norm(x)
        return self.head(x)

    def forward_from_hidden(self, x):
        """Run the SSM stack starting from a continuous hidden state — i.e.
        skip token embedding and the LM head. Returns the post-final-norm
        hidden state shape `(B, L, d_model)`.
        """
        for layer in self.kernel_layers:
            scale = layer["scale"][0]
            x = x + scale * layer["block"](layer["norm"](x))
        for layer in self.cortex_layers:
            scale = layer["scale"][0]
            x = x + scale * layer["block"](layer["norm"](x))
        if self.history_attn is not None:
            x = x + self.history_attn(x)
        return self.final_norm(x)

    def forward_hidden(self, tokens):
        """Run the model end-to-end (token in, hidden state out) without
        the LM head. Returns the post-final-norm hidden state shape
        `(B, L, d_model)`. This is the entry point a higher-order router
        uses when this model is plugged in as a specialist via AttendBridge
        — it runs on its native input distribution and exposes its
        representation, not its predicted logits.
        """
        x = self.embed_norm(self.embed(tokens))
        return self.forward_from_hidden(x)

    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def summary(self) -> str:
        n_kernel = len(self.kernel_layers)
        n_cortex = len(self.cortex_layers)
        n_total = self.total_params()
        n_train = self.get_trainable_params()
        return (f"ProgressiveModel: {n_kernel} kernel + {n_cortex} cortex layers, "
                f"{n_total:,} total params, {n_train:,} trainable (mode={self.mode})")


# ── Quick test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    tok = ByteTokenizer()

    # Test tokenizer
    example = {"input": "3 2", "output": "DIFF"}
    ids, sep = tok.encode_curriculum(example)
    print(f"Curriculum: {example} → {ids} (sep={sep})")
    print(f"  decoded input: {tok.decode(ids[1:sep])}")
    print(f"  decoded output: {tok.decode(ids[sep+1:-1])}")

    # Test model
    model = ProgressiveModel(d_model=64, d_state=16)
    model.add_kernel_layer()
    print(model.summary())

    tokens = torch.randint(0, VOCAB_SIZE, (2, 16))
    logits = model(tokens)
    print(f"Forward: {tokens.shape} → {logits.shape}")
    logits.sum().backward()
    print("Backward OK")

    # Test near-identity: adding a layer shouldn't change output much
    with torch.no_grad():
        logits_before = model(tokens).clone()
    model.add_kernel_layer()
    with torch.no_grad():
        logits_after = model(tokens)
    diff = (logits_before - logits_after).abs().max().item()
    print(f"Near-identity: max diff after adding layer = {diff:.6f}")

    # Test mode switching
    model.add_cortex_layer()
    model.set_mode("kernel")
    print(f"Kernel mode: {model.summary()}")
    model.set_mode("cortex")
    print(f"Cortex mode: {model.summary()}")
    model.set_mode("all")
    print(f"All mode: {model.summary()}")

    print("\nAll tests passed.")
