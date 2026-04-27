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

class ExplicitRegisters(nn.Module):
    """A bank of K register vectors that live *separate* from the SSM
    hidden state, written and read via differentiable soft-attention.

    Distinct from OutputHistoryAttention: that one is a pathway *through*
    the SSM (attention over the same tokens). This one is *external
    working memory* — the registers are persistent state that the SSM
    queries and updates. The SSM's continuous-state blending dynamics
    don't apply to the registers; a value written at timestep t stays
    until *explicitly* overwritten.

    Mental model: the model has K = 8 small CPU-style registers. At each
    timestep it can:
      - read a value via soft-addressing of the bank
      - write a value to a soft-selected register
    The recurrence `x_{k+1} = 2*x_k + 1` becomes "load running value
    into r0; multiply by 2 and store; emit digits; repeat."

    Per attention experiments: pre-norm input, zero-init out path, very
    small mix factor at start. Write gate biased toward closed (-3.0)
    so the registers are no-op until training opens them.

    Sequential per-timestep loop (O(L)). For L≤200 typical, fine.
    """

    def __init__(self, d_model: int, n_registers: int = 8, d_register: int = 32):
        super().__init__()
        self.n_registers = n_registers
        self.d_register = d_register

        self.in_norm = nn.LayerNorm(d_model)

        # Read: soft-addressing query over n_registers
        self.read_query = nn.Linear(d_model, n_registers)
        self.read_proj = nn.Linear(d_register, d_model)

        # Write: which register, what value, how strongly
        self.write_query = nn.Linear(d_model, n_registers)
        self.write_value = nn.Linear(d_model, d_register)
        self.write_gate = nn.Linear(d_model, 1)

        # Mix factor for adding the read result to the SSM state.
        # Init very small (0.001) — the new pathway is near-no-op at
        # step 1. Per output-history-attention learnings, aggressive
        # init destabilizes training within a few cycles.
        self.mix = nn.Parameter(torch.tensor(0.001))

        # Init the read out_proj to zero — the read path is a literal
        # no-op at step 1; the model has to learn its way into using it.
        nn.init.zeros_(self.read_proj.weight)
        nn.init.zeros_(self.read_proj.bias)
        # Write gate biased toward closed at start (sigmoid(-3) ≈ 0.05)
        # so the registers stay near-zero until training opens them.
        nn.init.zeros_(self.write_gate.weight)
        nn.init.constant_(self.write_gate.bias, -3.0)

    def forward(self, x):
        # x: (B, L, d_model)
        B, L, D = x.shape
        h = self.in_norm(x)

        # Initialize the register bank to zero per batch element.
        registers = torch.zeros(B, self.n_registers, self.d_register,
                                device=x.device, dtype=x.dtype)

        outputs = []
        # Per-position write values, exposed for trajectory distillation.
        # We store the *post-update* state of register r0 (the most-attended
        # register's accumulated value) at every position. Used by the
        # auxiliary trajectory loss to match against an oracle target.
        write_values_traj = []
        for t in range(L):
            h_t = h[:, t, :]                              # (B, d_model)

            r_q = self.read_query(h_t)
            r_w = torch.softmax(r_q, dim=-1)
            r_v = torch.einsum("br,brd->bd", r_w, registers)
            r_out = self.read_proj(r_v)

            w_q = self.write_query(h_t)
            w_w = torch.softmax(w_q, dim=-1)
            w_v = self.write_value(h_t)
            w_g = torch.sigmoid(self.write_gate(h_t))
            update_strength = (w_w * w_g).unsqueeze(-1)
            new_val = w_v.unsqueeze(1)
            registers = (1 - update_strength) * registers + update_strength * new_val

            outputs.append(r_out)
            # Track the post-update value of register 0 (the slot we
            # supervise via the oracle trajectory). Shape: (B, d_register).
            write_values_traj.append(registers[:, 0, :])

        self.last_register_traj = torch.stack(write_values_traj, dim=1)  # (B, L, d_register)
        return self.mix * torch.stack(outputs, dim=1)


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
        # Pre-norm on input (matches the existing layer norm pattern).
        # Critical for stability: the SSM's accumulated state has large
        # variance; running attention scores on unnormalized values can
        # produce extreme softmax values that send gradients to NaN.
        self.in_norm = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_attn)
        self.k_proj = nn.Linear(d_model, d_attn)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.scale = 1.0 / math.sqrt(d_attn)
        # Init out_proj at ZERO so the new pathway is a literal no-op
        # at step 1. The mix factor + learned out_proj give the model
        # full control over how much to use this pathway, growing it
        # gradually if it helps.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        self.mix = nn.Parameter(torch.tensor(0.01))

    def forward(self, x):
        B, L, D = x.shape
        h = self.in_norm(x)
        Q = self.q_proj(h)
        K = self.k_proj(h)
        V = self.v_proj(h)
        scores = torch.einsum("bld,bmd->blm", Q, K) * self.scale
        mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        out = torch.einsum("blm,bmd->bld", weights, V)
        out = self.out_proj(out)
        return self.mix * out


class LoopCounter(nn.Module):
    """A discrete integer-register pathway. Oracle-supervised: external
    code provides per-position counter values (parsed-from-input n at
    SEP, decremented one-per-output-position, sentinel outside the
    answer span). The module just looks up an embedding and adds a
    gated projection to the model stream.

    Unlike `ExplicitRegisters` (continuous, model-internal, soft I/O)
    this is *handed* to the model — at inference an external parser
    reads n from the input bytes and feeds the same trajectory.

    The argument we're testing: if the SSM is given a counter primitive,
    can it learn to gate EOS prediction on `c==0`? The bidir experiment
    showed the SSM cannot extract n itself; this externalises the part
    it failed at and asks if the rest of the program is learnable.
    """
    SENTINEL_OFFSET = 1  # max_count + 1 = sentinel index

    def __init__(self, d_model: int, max_count: int = 1024,
                 iteration_token: int = 49):
        super().__init__()
        self.d_model = d_model
        self.max_count = max_count
        # Token id to bias UP when counter > 0 (i.e., the loop body's
        # output token). For HANOIBIN this is ord('1') = 49.
        self.iteration_token = iteration_token
        # max_count + 2 entries: [0..max_count] valid + 1 sentinel
        self.c_emb = nn.Embedding(max_count + 2, d_model)
        nn.init.normal_(self.c_emb.weight, std=0.02)
        # Sentinel embedding starts at 0 so out-of-span positions are
        # initially a no-op contribution.
        with torch.no_grad():
            self.c_emb.weight[max_count + 1].zero_()
        self.read_proj = nn.Linear(d_model, d_model)
        # Init read_proj small so the counter pathway is near-identity at
        # start; the model has to learn to use it.
        nn.init.normal_(self.read_proj.weight, std=0.01)
        nn.init.zeros_(self.read_proj.bias)
        # mix=1.0: stronger initial presence in the hidden state. The
        # previous 0.1 init never grew (additive injection insufficient
        # finding); start strong and let the model attenuate if needed.
        self.mix = nn.Parameter(torch.tensor(1.0))

        # Direct EOS-logit bias per counter value. Bypasses the LM head's
        # weight tying so the counter overrides instead of competing with
        # the SSM's prediction. Hot init: c=0 -> +15 (force EOS even
        # against a confident '1' prediction; logit gap typically ~10),
        # c in [1..max_count] -> -5 (suppress EOS), sentinel -> 0
        # (input span / past-EOS, no bias). The previous additive
        # injection failed because the SSM dominates the logit space; a
        # direct override gives the counter pathway its own channel.
        self.eos_bias = nn.Embedding(max_count + 2, 1)
        with torch.no_grad():
            self.eos_bias.weight.fill_(-15.0)
            self.eos_bias.weight[0] = 30.0
            self.eos_bias.weight[max_count + 1] = 0.0

        # Mirror bias for the iteration token: push UP when counter > 0,
        # DOWN when counter == 0 (so EOS wins decisively at the boundary).
        # Without this, deep in the answer span the SSM's hidden-state
        # alignment drifts and other tokens (notably SEP=258) start
        # winning over '1'. This explicitly encodes loop-body semantics:
        # "while counter>0: emit iteration_token; at counter==0: stop."
        self.iter_bias = nn.Embedding(max_count + 2, 1)
        with torch.no_grad():
            self.iter_bias.weight.fill_(15.0)
            self.iter_bias.weight[0] = -30.0
            self.iter_bias.weight[max_count + 1] = 0.0

    @property
    def sentinel(self) -> int:
        return self.max_count + 1

    def forward(self, counter_values: torch.Tensor) -> torch.Tensor:
        """counter_values: int64 (B, L). Returns (B, L, d_model)."""
        # Clamp to valid range; out-of-range -> sentinel.
        cv = counter_values.clamp(0, self.max_count + 1)
        return self.mix * self.read_proj(self.c_emb(cv))

    def get_eos_bias(self, counter_values: torch.Tensor) -> torch.Tensor:
        """counter_values: int64 (B, L). Returns (B, L) scalar bias on the
        EOS logit, applied AFTER the LM head."""
        cv = counter_values.clamp(0, self.max_count + 1)
        return self.eos_bias(cv).squeeze(-1)

    def get_iter_bias(self, counter_values: torch.Tensor) -> torch.Tensor:
        """counter_values: int64 (B, L). Returns (B, L) scalar bias on the
        iteration-token logit, applied AFTER the LM head."""
        cv = counter_values.clamp(0, self.max_count + 1)
        return self.iter_bias(cv).squeeze(-1)


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
                 use_history_attn: bool = False, history_d_attn: int = 32,
                 use_explicit_registers: bool = False,
                 n_registers: int = 8, d_register: int = 32,
                 use_loop_counter: bool = False, loop_counter_max: int = 1024):
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

        # Optional explicit register bank — separate persistent state
        # the model can read/write across timesteps. CPU-register-style
        # working memory for unbounded program execution.
        self.use_explicit_registers = use_explicit_registers
        self.registers = (ExplicitRegisters(d_model, n_registers=n_registers,
                                             d_register=d_register)
                          if use_explicit_registers else None)

        # Optional discrete loop counter — oracle-supervised integer
        # register the model reads at every output position.
        self.use_loop_counter = use_loop_counter
        self.loop_counter = (LoopCounter(d_model, max_count=loop_counter_max)
                             if use_loop_counter else None)

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

    def init_decode_state(self, batch_size: int):
        """Per-layer SSM state for step-mode autoregressive decoding.
        Returns a list of dicts (one per kernel + cortex layer).

        Note: history_attn / explicit_registers are NOT supported in step
        mode — they need their own caches. Asserts they're absent so we
        catch surprises early.
        """
        if self.history_attn is not None or self.registers is not None:
            raise NotImplementedError(
                "history_attn / explicit_registers not supported in step mode")
        device = self._get_device()
        dtype = self.embed.weight.dtype
        states = []
        for layer in self.kernel_layers:
            states.append(layer["block"].init_state(batch_size, device, dtype))
        for layer in self.cortex_layers:
            states.append(layer["block"].init_state(batch_size, device, dtype))
        return states

    def forward_step(self, token, counter_value=None, states=None):
        """Single-token forward pass for autoregressive decoding.

        token:         (B, 1) int64
        counter_value: (B, 1) int64 or None
        states:        list[dict] from init_decode_state()

        Returns (logits: (B, 1, V), new_states).

        For batch_size=1 and a model trained at d_model=64 L=3, this is
        ~50–100x faster than re-running the prefix at every step.
        """
        if states is None:
            states = self.init_decode_state(token.shape[0])
        new_states = []
        x = self.embed_norm(self.embed(token))
        for i, layer in enumerate(self.kernel_layers):
            scale = layer["scale"][0]
            normed = layer["norm"](x)
            ssm_out, new_state = layer["block"].step(normed, states[i])
            x = x + scale * ssm_out
            new_states.append(new_state)
        offset = len(self.kernel_layers)
        for j, layer in enumerate(self.cortex_layers):
            scale = layer["scale"][0]
            normed = layer["norm"](x)
            ssm_out, new_state = layer["block"].step(normed, states[offset + j])
            x = x + scale * ssm_out
            new_states.append(new_state)
        # Loop counter: same as full forward but for L=1
        eos_bias = None
        iter_bias = None
        iter_tok = None
        if self.loop_counter is not None:
            if counter_value is None:
                counter_value = torch.full(token.shape, self.loop_counter.sentinel,
                                           dtype=torch.long, device=token.device)
            x = x + self.loop_counter(counter_value)
            eos_bias = self.loop_counter.get_eos_bias(counter_value)
            iter_bias = self.loop_counter.get_iter_bias(counter_value)
            iter_tok = self.loop_counter.iteration_token
        x = self.final_norm(x)
        logits = self.head(x)
        if eos_bias is not None:
            addition = torch.zeros_like(logits)
            addition[..., EOS] = eos_bias
            if iter_tok is not None and iter_bias is not None:
                addition[..., iter_tok] = iter_bias
            logits = logits + addition
        return logits, new_states

    def forward(self, tokens, counter_values=None):
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

        # Explicit registers: external working memory bank, read/write
        # via soft-addressing, persistent across timesteps. No-op if not
        # configured.
        if self.registers is not None:
            x = x + self.registers(x)

        # Loop counter: oracle/parser-supervised. counter_values must be
        # provided when this pathway is active; if None and the module
        # exists we feed an all-sentinel tensor (no-op).
        eos_bias = None
        iter_bias = None
        iter_tok = None
        if self.loop_counter is not None:
            if counter_values is None:
                counter_values = torch.full(tokens.shape, self.loop_counter.sentinel,
                                            dtype=torch.long, device=tokens.device)
            x = x + self.loop_counter(counter_values)
            eos_bias = self.loop_counter.get_eos_bias(counter_values)  # (B, L)
            iter_bias = self.loop_counter.get_iter_bias(counter_values)
            iter_tok = self.loop_counter.iteration_token

        x = self.final_norm(x)
        logits = self.head(x)

        # Direct logit overrides from the LoopCounter. Bypasses weight
        # tying so the counter doesn't have to fight the SSM through the
        # LM head. Two channels:
        #   - EOS column gets a bias that is positive at counter==0,
        #     negative otherwise (force "stop" at boundary).
        #   - Iteration-token column gets the mirror (force "continue"
        #     when counter > 0). Without this, the SSM's hidden state
        #     drifts deep in the answer span and other tokens win.
        if eos_bias is not None:
            addition = torch.zeros_like(logits)
            addition[..., EOS] = eos_bias
            if iter_tok is not None and iter_bias is not None:
                addition[..., iter_tok] = iter_bias
            logits = logits + addition
        return logits

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
        if self.registers is not None:
            x = x + self.registers(x)
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
