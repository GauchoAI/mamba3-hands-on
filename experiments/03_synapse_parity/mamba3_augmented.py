"""
Augmented Mamba-3 — SSM + Registers + Spike Gates + Persistent Memory.

The brain of a fly: not more parameters, but the right *kinds* of state.

Architecture:
  1. SSM layer (Mamba-3): fast sequential processing, pattern detection
  2. Register bank: small set of explicitly addressable slots (working memory)
  3. Spike gate: threshold mechanism that decides WHEN to write to registers
  4. Persistent memory: slow-decay bank for accumulated intuitions

The SSM sees the stream. When it detects something important (spike fires),
it writes to a register. Registers persist across the sequence. The output
at each step combines SSM processing with register reads.

This separates "what I'm processing now" (SSM state, fast)
from "what I've learned so far" (registers, persistent).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_platform.mamba3_minimal import Mamba3Block, Mamba3Config


class SpikeGate(nn.Module):
    """
    Threshold gate — fires when confidence exceeds a learned threshold.

    Output is near-zero most of the time (no write), near-one when
    something noteworthy is detected (write to register).

    Uses a steep sigmoid with learned bias starting negative, so the
    default is "don't fire." The model must learn to fire.
    """
    def __init__(self, d_input, sharpness=5.0):
        super().__init__()
        self.proj = nn.Linear(d_input, 1)
        self.sharpness = sharpness
        # Initialize bias negative → default is "don't fire"
        nn.init.constant_(self.proj.bias, -2.0)

    def forward(self, x):
        """x: (B, D) → gate: (B, 1) in [0, 1], mostly near 0."""
        return torch.sigmoid(self.sharpness * self.proj(x))


class RegisterBank(nn.Module):
    """
    Explicit addressable memory — like CPU registers.

    n_registers slots of dimension d_reg. Supports:
      - Soft write: gate * attention_over_slots * value
      - Soft read: query-based attention over slots
      - Persistence: registers carry forward unless explicitly written

    This is NOT a tensor operation in the usual sense. It's a structured
    memory that the model learns to use as a tool.
    """
    def __init__(self, d_model, n_registers=8):
        super().__init__()
        self.n_registers = n_registers
        self.d_reg = d_model

        # Write path
        self.write_gate = SpikeGate(d_model)           # WHEN to write
        self.write_addr = nn.Linear(d_model, n_registers)  # WHERE to write
        self.write_value = nn.Linear(d_model, d_model)     # WHAT to write

        # Read path
        self.read_query = nn.Linear(d_model, d_model)
        self.read_scale = d_model ** -0.5

    def forward(self, h_t, registers):
        """
        h_t: (B, D) — current SSM output
        registers: (B, n_reg, D) — current register contents
        Returns: (read_value, updated_registers, write_gate_value)
        """
        B = h_t.shape[0]

        # ── Write ──
        gate = self.write_gate(h_t)                        # (B, 1) — spike decision
        addr = F.softmax(self.write_addr(h_t), dim=-1)     # (B, n_reg) — which register
        value = self.write_value(h_t)                       # (B, D) — what to store

        # Write = gate * outer(addr, value)
        # Only modifies registers when gate fires (near 1)
        write = gate.unsqueeze(-1) * addr.unsqueeze(-1) * value.unsqueeze(1)
        registers = registers + write                       # (B, n_reg, D)

        # ── Read ──
        query = self.read_query(h_t)                        # (B, D)
        # Attention scores over registers
        scores = torch.bmm(registers, query.unsqueeze(-1)).squeeze(-1)  # (B, n_reg)
        scores = scores * self.read_scale
        attn = F.softmax(scores, dim=-1)                    # (B, n_reg)
        read_value = torch.bmm(attn.unsqueeze(1), registers).squeeze(1)  # (B, D)

        return read_value, registers, gate.squeeze(-1)


class PersistentMemory(nn.Module):
    """
    Slow-decay memory bank — accumulated intuitions.

    Like registers but with a decay mechanism. Old memories fade slowly
    unless refreshed. This creates a natural priority: frequently
    reinforced patterns persist, one-off observations decay.
    """
    def __init__(self, d_model, n_slots=16, decay=0.995):
        super().__init__()
        self.n_slots = n_slots
        self.d_mem = d_model
        self.decay = decay

        self.write_gate = SpikeGate(d_model)
        self.write_addr = nn.Linear(d_model, n_slots)
        self.write_value = nn.Linear(d_model, d_model)

        self.read_query = nn.Linear(d_model, d_model)
        self.read_scale = d_model ** -0.5

    def forward(self, h_t, memory):
        """
        h_t: (B, D)
        memory: (B, n_slots, D)
        Returns: (read_value, updated_memory, write_gate_value)
        """
        # Decay existing memory (slow forgetting)
        memory = memory * self.decay

        # Write (same mechanism as registers, but with decay)
        gate = self.write_gate(h_t)
        addr = F.softmax(self.write_addr(h_t), dim=-1)
        value = self.write_value(h_t)
        write = gate.unsqueeze(-1) * addr.unsqueeze(-1) * value.unsqueeze(1)
        memory = memory + write

        # Read
        query = self.read_query(h_t)
        scores = torch.bmm(memory, query.unsqueeze(-1)).squeeze(-1)
        scores = scores * self.read_scale
        attn = F.softmax(scores, dim=-1)
        read_value = torch.bmm(attn.unsqueeze(1), memory).squeeze(1)

        return read_value, memory, gate.squeeze(-1)


class AugmentedMamba3(nn.Module):
    """
    The augmented architecture:

      Input → SSM (Mamba-3) → per-step register read/write → output
                                    ↕
                              Register Bank (working memory)
                              Persistent Memory (long-term)

    The SSM does the fast sequential processing.
    Registers store important findings (spike-gated writes).
    Persistent memory accumulates patterns over longer timescales.
    Output combines all three sources.
    """
    def __init__(self, cfg, n_registers=8, n_memory=16, memory_decay=0.995):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        self.ssm = Mamba3Block(cfg)
        self.registers = RegisterBank(d, n_registers)
        self.memory = PersistentMemory(d, n_memory, memory_decay)

        # Combine SSM output + register read + memory read
        self.combine = nn.Linear(d * 3, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, u):
        """
        u: (B, L, d_model) → (B, L, d_model)
        """
        B, L, D = u.shape

        # Step 1: SSM processes entire sequence
        ssm_out = self.ssm(u)  # (B, L, D)

        # Step 2: Sequential register/memory operations on SSM output
        reg_state = u.new_zeros(B, self.registers.n_registers, D)
        mem_state = u.new_zeros(B, self.memory.n_slots, D)

        outputs = []
        total_reg_spikes = 0
        total_mem_spikes = 0

        for t in range(L):
            h_t = ssm_out[:, t]  # (B, D)

            # Read/write registers
            reg_read, reg_state, reg_gate = self.registers(h_t, reg_state)

            # Read/write persistent memory
            mem_read, mem_state, mem_gate = self.memory(h_t, mem_state)

            # Combine all three sources
            combined = self.combine(torch.cat([h_t, reg_read, mem_read], dim=-1))
            combined = self.norm(combined + h_t)  # residual from SSM

            outputs.append(combined)
            total_reg_spikes += reg_gate.sum().item()
            total_mem_spikes += mem_gate.sum().item()

        # Store spike stats for monitoring
        self._last_reg_spikes = total_reg_spikes / (B * L)
        self._last_mem_spikes = total_mem_spikes / (B * L)

        return torch.stack(outputs, dim=1)  # (B, L, D)


# ── Quick comparison test ────────────────────────────────────────────

if __name__ == "__main__":
    import os
    os.environ["PYTHONUNBUFFERED"] = "1"

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    cfg = Mamba3Config(d_model=64, d_state=16, expand=2, headdim=16)

    # Plain Mamba-3
    plain = Mamba3Block(cfg).to(device)
    n_plain = sum(p.numel() for p in plain.parameters())

    # Augmented Mamba-3
    aug = AugmentedMamba3(cfg, n_registers=8, n_memory=16).to(device)
    n_aug = sum(p.numel() for p in aug.parameters())

    print(f"Plain Mamba-3:     {n_plain:,} params", flush=True)
    print(f"Augmented Mamba-3: {n_aug:,} params (+{n_aug - n_plain:,})", flush=True)

    # Forward pass test
    u = torch.randn(4, 16, cfg.d_model, device=device)

    y_plain = plain(u)
    y_aug = aug(u)

    print(f"\nPlain   — in: {tuple(u.shape)} → out: {tuple(y_plain.shape)}", flush=True)
    print(f"Augment — in: {tuple(u.shape)} → out: {tuple(y_aug.shape)}", flush=True)
    print(f"  reg spikes/step: {aug._last_reg_spikes:.3f}", flush=True)
    print(f"  mem spikes/step: {aug._last_mem_spikes:.3f}", flush=True)

    # Backward pass
    y_aug.sum().backward()
    print(f"\nBackward pass OK.", flush=True)
    print("Ready for training.", flush=True)
