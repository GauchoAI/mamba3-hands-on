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
from mamba_platform.mamba3_minimal import Mamba3Block, Mamba3Config


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
    """Parameter-free counter pathway: arbitrary unbounded c.

    The previous design used (max_count+2, ...) embedding tables, but
    after training the iter_bias / eos_bias values were essentially
    constant for c>0 and only c_emb[0] grew meaningfully. The actual
    architectural distinction was always 3-way:

      c < 0  : sentinel (input span / past EOS) — no contribution
      c == 0 : at the EOS-target slot — emit EOS
      c > 0  : in answer span, iterating — emit iter_token

    This refactor encodes those three states as torch.where dispatch
    on the SIGN of c, with two learned d_model embeddings (stop_emb /
    iter_emb) and four scalar biases. There is no `max_count` — c can
    be any int64. HANOIBIN at n=10000 works the same as at n=20.

    The depth ceiling we still have is the *training-curriculum* one
    (the SSM's hidden state at deep answer-span positions is only
    supervised up to the deepest training example). That's a real
    limit; the table-size limit was an artifact of the previous design.
    """
    # Magic sentinel value the oracles use for "outside answer span".
    # Any negative value works; -1 is human-readable.
    SENTINEL = -1

    def __init__(self, d_model: int, iteration_token: int = 49,
                 # `max_count` kept for backwards-compatible kwargs in
                 # ProgressiveModel call sites; ignored.
                 max_count: int = None):
        super().__init__()
        self.d_model = d_model
        self.iteration_token = iteration_token

        # Two d_model embeddings: "stop" (c==0) and "iterate" (c>0).
        # Sentinel positions add zero (no parameters needed).
        self.stop_emb = nn.Parameter(torch.randn(d_model) * 0.02)
        self.iter_emb = nn.Parameter(torch.randn(d_model) * 0.02)

        self.read_proj = nn.Linear(d_model, d_model)
        nn.init.normal_(self.read_proj.weight, std=0.01)
        nn.init.zeros_(self.read_proj.bias)
        self.mix = nn.Parameter(torch.tensor(1.0))

        # Hot-init biases. These are scalars; no per-c table.
        # eos_bias_zero  +70 (force EOS even when LM head has logit~50 on
        #                     a previous-digit alignment via weight tying)
        # eos_bias_pos   -30 (suppress EOS mid-loop)
        # iter_bias_zero -40 (suppress iter_token at boundary; for HANOIBIN
        #                     this suppresses '1' at counter=0)
        # iter_bias_pos  +70 (force iter_token in answer span; matches
        #                     eos_bias_zero in magnitude — dominates LM
        #                     head's previous-input alignment)
        # Both biases at sentinel are 0 (no-op outside answer span).
        # Symmetric magnitudes (70/40) matter: at counter=0 the eos
        # boost (+70) and iter suppress (-40) together overwhelm any
        # base LM-head alignment of ~50; same logic at counter>0 with
        # iter boost (+70) + eos suppress (-30).
        self.eos_bias_zero = nn.Parameter(torch.tensor(70.0))
        self.eos_bias_pos  = nn.Parameter(torch.tensor(-30.0))
        self.iter_bias_zero = nn.Parameter(torch.tensor(-40.0))
        self.iter_bias_pos  = nn.Parameter(torch.tensor(70.0))

    @property
    def sentinel(self) -> int:
        return self.SENTINEL

    def _state_masks(self, c: torch.Tensor):
        """Returns (is_zero, is_pos) bool masks; sentinel is the negation
        of (is_zero | is_pos)."""
        is_zero = (c == 0)
        is_pos = (c > 0)
        return is_zero, is_pos

    def forward(self, counter_values: torch.Tensor) -> torch.Tensor:
        """counter_values: int64 (B, L). Returns (B, L, d_model).
        Three-way: sentinel -> 0, zero -> stop_emb, positive -> iter_emb."""
        is_zero, is_pos = self._state_masks(counter_values)
        # (B, L, d_model)
        emb = torch.where(
            is_zero.unsqueeze(-1),
            self.stop_emb.expand_as(counter_values.unsqueeze(-1).expand(*counter_values.shape, self.d_model)),
            torch.where(
                is_pos.unsqueeze(-1),
                self.iter_emb.expand_as(counter_values.unsqueeze(-1).expand(*counter_values.shape, self.d_model)),
                torch.zeros(*counter_values.shape, self.d_model,
                            device=counter_values.device, dtype=self.stop_emb.dtype),
            ),
        )
        return self.mix * self.read_proj(emb)

    def get_eos_bias(self, counter_values: torch.Tensor) -> torch.Tensor:
        is_zero, is_pos = self._state_masks(counter_values)
        return torch.where(is_zero, self.eos_bias_zero,
               torch.where(is_pos, self.eos_bias_pos,
                           torch.zeros_like(counter_values, dtype=self.eos_bias_zero.dtype)))

    def get_iter_bias(self, counter_values: torch.Tensor) -> torch.Tensor:
        """Bias amount on the iteration-token logit. The iteration token
        itself is the scalar self.iteration_token (HANOIBIN, FIB-unary)
        or per-position via iter_token_per_pos (FIB-decimal)."""
        is_zero, is_pos = self._state_masks(counter_values)
        return torch.where(is_zero, self.iter_bias_zero,
               torch.where(is_pos, self.iter_bias_pos,
                           torch.zeros_like(counter_values, dtype=self.iter_bias_zero.dtype)))


class MultiChannelStateFeedback(nn.Module):
    """K parallel state channels, each holding a small int.

    For Hanoi: K = number of disks; each channel holds the peg of disk k
    (in {0=A, 1=B, 2=C, 3=none}). Adding more disks = more channels at
    runtime, no parameter change.

    Architecture:
        embed(channel k, value v) = value_emb(v) + sin_pos(k)
        feedback(B, L) = sum_{k=0..K-1} embed(k, channel_value[k])
        output = mix * read_proj(feedback)

    `value_emb` is small (value_range × d_model). Channel-position is
    sinusoidal — zero learned per-channel params. K can be anything at
    inference; train uses the curriculum's max disk count.

    Per-channel param cost: ZERO. Total module params: ~value_range *
    d_model + d_model^2 + d_model + 1 = small.
    """
    def __init__(self, d_model: int, value_range: int = 4):
        super().__init__()
        self.d_model = d_model
        self.value_range = value_range
        self.value_emb = nn.Embedding(value_range, d_model)
        nn.init.normal_(self.value_emb.weight, std=0.02)
        self.read_proj = nn.Linear(d_model, d_model)
        nn.init.normal_(self.read_proj.weight, std=0.01)
        nn.init.zeros_(self.read_proj.bias)
        self.mix = nn.Parameter(torch.tensor(0.1))

    def _channel_pos_codes(self, K: int, device, dtype) -> torch.Tensor:
        """Sinusoidal positional encoding for channel index. (K, d_model).
        No learned parameters."""
        positions = torch.arange(K, device=device, dtype=dtype).unsqueeze(1)
        i = torch.arange(0, self.d_model, 2, device=device, dtype=dtype)
        omega = 1.0 / (10000.0 ** (i / self.d_model))
        angles = positions * omega.unsqueeze(0)  # (K, d_model/2)
        codes = torch.zeros(K, self.d_model, device=device, dtype=dtype)
        codes[:, 0::2] = torch.sin(angles)
        codes[:, 1::2] = torch.cos(angles)
        return codes

    def forward(self, channels: torch.Tensor) -> torch.Tensor:
        """channels: int64 (B, L, K). Returns (B, L, d_model)."""
        B, L, K = channels.shape
        clamped = channels.clamp(0, self.value_range - 1)
        val_emb = self.value_emb(clamped)  # (B, L, K, d_model)
        ch_pos = self._channel_pos_codes(K, channels.device, val_emb.dtype)  # (K, d_model)
        combined = val_emb + ch_pos.view(1, 1, K, self.d_model)
        feedback = combined.sum(dim=2)  # (B, L, d_model)
        return self.mix * self.read_proj(feedback)


class RegisterBank(nn.Module):
    """A discrete integer-register pathway with hard read/write semantics.

    Sibling to LoopCounter — but where LoopCounter exposes a single
    ternary signal (sentinel/zero/positive), RegisterBank exposes
    a *bank* of discrete integer registers that the model can
    address, read, and write.

    Per timestep the model emits 3 control signals from the SSM
    hidden state:
        read_addr  ∈ [0, n_registers]   (n_registers means "no-read")
        write_addr ∈ [0, n_registers]   (n_registers means "no-write")
        write_val  ∈ [0, value_range)   (only meaningful if write_addr < n_registers)

    Plus the regular token output. So the model is producing a
    4-tuple per step.

    Read FEEDBACK: the value at register[read_addr] is fed back
    as an additional input embedding on the *next* step (a learned
    embedding indexed by the integer value). This is how the model
    sees register contents.

    Discretization: straight-through estimator. Forward uses
    argmax (truly discrete addresses + values), backward uses
    softmax of the head logits (smooth gradient). Same trick as
    VQ-VAE / Gumbel-softmax with hard=True.

    The bank STATE itself (the actual register values) is not stored
    in this module — it's maintained by the decoder loop at inference
    time, and oracle-supplied at training time. This keeps the
    architecture stateless and lets training run in parallel over
    a full sequence.
    """
    NO_READ_INDEX = lambda self: self.n_registers   # last index = no-op
    NO_WRITE_INDEX = lambda self: self.n_registers

    def __init__(self, d_model: int, n_registers: int = 16,
                 value_range: int = 16):
        super().__init__()
        self.d_model = d_model
        self.n_registers = n_registers
        self.value_range = value_range

        # Heads producing per-step control signals.
        # read_addr / write_addr have an extra "no-op" index at n_registers.
        self.read_addr_head = nn.Linear(d_model, n_registers + 1)
        self.write_addr_head = nn.Linear(d_model, n_registers + 1)
        self.write_val_head = nn.Linear(d_model, value_range)
        # Init the heads near zero so the model starts with no preference;
        # cross-entropy loss on the oracle targets shapes the distribution.
        for h in (self.read_addr_head, self.write_addr_head, self.write_val_head):
            nn.init.normal_(h.weight, std=0.01)
            nn.init.zeros_(h.bias)
        # No-op bias on the last (no-op) index of read/write addr is +2
        # at init: model defaults to "do nothing" until trained. Avoids
        # spurious early register operations from random init.
        with torch.no_grad():
            self.read_addr_head.bias[n_registers] = 2.0
            self.write_addr_head.bias[n_registers] = 2.0

        # Read-value embedding: V_REG -> d_model. Fed back as additive
        # input on the *next* step (the decoder concatenates token_emb +
        # last_read_emb before the SSM stack).
        self.value_emb = nn.Embedding(value_range, d_model)
        nn.init.normal_(self.value_emb.weight, std=0.02)
        # mix at 0.1 — start small, model has to learn to use the read.
        # Same lesson as LoopCounter: not too large at init or it
        # over-influences early dynamics, not too small or it never grows.
        self.value_mix = nn.Parameter(torch.tensor(0.1))

    def heads(self, x: torch.Tensor):
        """SSM hidden state -> (read_logits, write_logits, val_logits).
        x: (B, L, d_model). Returns three tensors with last-dim sizes
        n_registers+1, n_registers+1, value_range respectively.
        """
        return (
            self.read_addr_head(x),
            self.write_addr_head(x),
            self.write_val_head(x),
        )

    def read_feedback(self, read_values: torch.Tensor) -> torch.Tensor:
        """Embed integer read-results for feed-back into the next step.
        read_values: int64 (B,) or (B, L). Returns (..., d_model)."""
        return self.value_mix * self.value_emb(read_values.clamp(0, self.value_range - 1))

    @staticmethod
    def straight_through_argmax(logits: torch.Tensor) -> torch.Tensor:
        """Forward = argmax index (discrete). Backward = softmax gradient.
        Returns int64 (..., ) — the chosen index."""
        soft = torch.softmax(logits, dim=-1)
        idx = soft.argmax(dim=-1)
        # Straight-through: index is discrete, gradient flows via soft.
        # We don't need the soft tensor to be tied — the gradient is on
        # the head's parameters via CE loss in the trainer; this method
        # is for inference-time decoding.
        return idx

    def execute_step(self, registers: torch.Tensor,
                     read_idx: torch.Tensor, write_idx: torch.Tensor,
                     write_val: torch.Tensor):
        """Apply one step of register IO.

        registers:  int64 (B, n_registers) — current bank state
        read_idx:   int64 (B,) — index in [0, n_registers]; n_registers = no-read
        write_idx:  int64 (B,) — same convention
        write_val:  int64 (B,) — value in [0, value_range)

        Returns:
          new_registers: int64 (B, n_registers)
          read_value:    int64 (B,)  — register[read_idx] or 0 if no-read
        """
        B, N = registers.shape
        # Read: gather the register at read_idx; clamp to valid range first
        # so the no-read index doesn't index past the bank.
        clamped_read = read_idx.clamp(0, N - 1)
        read_value = registers.gather(1, clamped_read.unsqueeze(1)).squeeze(1)
        # Mask: if no-read, return 0
        is_no_read = (read_idx == N)
        read_value = torch.where(is_no_read, torch.zeros_like(read_value), read_value)

        # Write: scatter write_val into write_idx, conditional on not-no-op
        new_regs = registers.clone()
        is_write = (write_idx < N)
        if is_write.any():
            # For the rows that have a real write, do scatter
            for b in range(B):
                if is_write[b]:
                    new_regs[b, write_idx[b]] = write_val[b]
        return new_regs, read_value


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
                 use_loop_counter: bool = False, loop_counter_max: int = 1024,
                 lc_iteration_token: int = 49,
                 use_register_bank: bool = False,
                 reg_n_registers: int = 16, reg_value_range: int = 16,
                 use_state_feedback: bool = False,
                 sf_value_range: int = 4):
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
        self.loop_counter = (LoopCounter(d_model, max_count=loop_counter_max,
                                          iteration_token=lc_iteration_token)
                             if use_loop_counter else None)

        # Optional discrete register bank — hard read/write semantics
        # for state-machine execution. Used by tasks like
        # tower_of_hanoi_exec where the model has to maintain state
        # across many output positions.
        self.use_register_bank = use_register_bank
        self.register_bank = (RegisterBank(d_model, n_registers=reg_n_registers,
                                           value_range=reg_value_range)
                              if use_register_bank else None)

        # MultiChannelStateFeedback: parameter-free in slot/channel count.
        # Used by tool-use tasks (Hanoi-tool, etc.) where state is a list
        # of small ints supplied per-step from a Python tool.
        self.use_state_feedback = use_state_feedback
        self.state_feedback = (MultiChannelStateFeedback(d_model, value_range=sf_value_range)
                                if use_state_feedback else None)

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

    def forward_step(self, token, counter_value=None, states=None,
                     iter_token_per_pos=None):
        """Single-token forward pass for autoregressive decoding.

        token:               (B, 1) int64
        counter_value:       (B, 1) int64 or None
        iter_token_per_pos:  (B, 1) int64 or None (FIB-decimal etc.)
        states:              list[dict] from init_decode_state()

        Returns (logits: (B, 1, V), new_states).

        For batch_size=1 and a model trained at d_model=64 L=3, this is
        ~50-100x faster than re-running the prefix at every step.
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
        eos_bias = None
        iter_bias = None
        iter_tok_scalar = None
        if self.loop_counter is not None:
            if counter_value is None:
                counter_value = torch.full(token.shape, self.loop_counter.sentinel,
                                           dtype=torch.long, device=token.device)
            x = x + self.loop_counter(counter_value)
            eos_bias = self.loop_counter.get_eos_bias(counter_value)
            iter_bias = self.loop_counter.get_iter_bias(counter_value)
            iter_tok_scalar = self.loop_counter.iteration_token
        x = self.final_norm(x)
        logits = self.head(x)
        if eos_bias is not None:
            addition = torch.zeros_like(logits)
            addition[..., EOS] = eos_bias
            if iter_bias is not None:
                if iter_token_per_pos is not None:
                    addition.scatter_add_(
                        2,
                        iter_token_per_pos.unsqueeze(-1),
                        iter_bias.unsqueeze(-1),
                    )
                elif iter_tok_scalar is not None:
                    addition[..., iter_tok_scalar] = iter_bias
            logits = logits + addition
        return logits, new_states

    def forward(self, tokens, counter_values=None, iter_token_per_pos=None,
                register_read_values=None, state_channels=None):
        """
        register_read_values: int64 (B, L) or None. If the register
        bank is active and oracle-supplied (training), this is the
        per-position read-back value (the integer that was at the
        register read at the PREVIOUS step). It's embedded via the
        bank's value_emb and added to the token embedding so the
        SSM "sees" the register contents.

        state_channels: int64 (B, L, K) or None. K parallel channels
        of state — each channel is a small int (e.g., peg of disk k).
        Used by tool-use tasks where state is multi-dimensional and
        K can vary between examples.

        Return value:
          - if not use_register_bank: token logits (B, L, V) [unchanged]
          - if use_register_bank: a dict with keys
                'token_logits':  (B, L, V)
                'read_logits':   (B, L, n_registers + 1)
                'write_logits':  (B, L, n_registers + 1)
                'val_logits':    (B, L, value_range)
        """
        x = self.embed_norm(self.embed(tokens))
        # Inject register read-feedback as additive input embedding.
        if self.register_bank is not None and register_read_values is not None:
            x = x + self.register_bank.read_feedback(register_read_values)
        # Inject multi-channel state feedback (tool-use tasks).
        if self.state_feedback is not None and state_channels is not None:
            x = x + self.state_feedback(state_channels)

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
        iter_tok_scalar = None
        if self.loop_counter is not None:
            if counter_values is None:
                counter_values = torch.full(tokens.shape, self.loop_counter.sentinel,
                                            dtype=torch.long, device=tokens.device)
            x = x + self.loop_counter(counter_values)
            eos_bias = self.loop_counter.get_eos_bias(counter_values)  # (B, L)
            iter_bias = self.loop_counter.get_iter_bias(counter_values)
            iter_tok_scalar = self.loop_counter.iteration_token

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
            if iter_bias is not None:
                if iter_token_per_pos is not None:
                    # Per-position iter_token (FIB-decimal etc.): scatter
                    # the bias amount to the position-specific token.
                    addition.scatter_add_(
                        2,
                        iter_token_per_pos.unsqueeze(-1),
                        iter_bias.unsqueeze(-1),
                    )
                elif iter_tok_scalar is not None:
                    # Scalar iter_token (HANOIBIN, FIB-unary).
                    addition[..., iter_tok_scalar] = iter_bias
            logits = logits + addition

        # If RegisterBank is active, ALSO compute the three control
        # heads from the same SSM hidden state and return them
        # alongside token logits as a dict. Backwards-compat: when
        # the bank is not active the return type is the plain tensor.
        if self.register_bank is not None:
            r_logits, w_logits, v_logits = self.register_bank.heads(x)
            return {
                "token_logits": logits,
                "read_logits": r_logits,
                "write_logits": w_logits,
                "val_logits": v_logits,
            }
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
