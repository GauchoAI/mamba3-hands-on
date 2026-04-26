"""synapse — register-space invocation of frozen specialist models.

Two designs in one file:

ProjectedBridge (v1) — router's own state is projected into the
specialist's input space (W_send), specialist runs with that
projected state via forward_from_hidden, output is mapped back
through W_recv. Empirically marginal because the specialist's
frozen dynamics expect token-embedding inputs, not arbitrary
continuous router projections.

AttendBridge (v2) — specialist runs on the ORIGINAL input bytes
(its native diet), produces its native hidden state at each
timestep, and the router learns to ATTEND to that state via a
learned recv + a per-timestep gate. No projection from router
into specialist's input space; the specialist sees what it was
trained on. This is the design that matches the biological
intuition: a specialist 'cortical area' activates on its native
input, and a higher-order area reads from it when useful.

Both bridges share the same gating + recv arithmetic; the
difference is whether the router's state is projected into the
specialist's input space or not.

The specialist stays frozen; only the bridges and the router's
own weights are trained. Differentiable end-to-end via the soft
gate. Mental model: each bridge is a synapse between two
register populations. With N specialists, the router sprouts N
synapses but does not grow.
"""
from __future__ import annotations
from pathlib import Path
import torch
import torch.nn as nn

from progressive_model import ProgressiveModel, VOCAB_SIZE


class AttendBridge(nn.Module):
    """v2 synapse: specialist runs on the ORIGINAL input bytes; the
    router learns when to attend to its hidden state.

    No `W_send`. The specialist is invoked once per forward via the
    same `tokens` the router got, in eval / no_grad. Its post-final-norm
    hidden state `(B, L, d_specialist)` is what we read from. The router
    decides — per timestep — how much of that signal to pull in via:

        gate_t  = σ(W_g · x_router_t)              ∈ [0, 1]
        contrib = gate_t · (W_recv · spec_hidden_t)

    Both `W_recv` and `W_g` are tiny (`d_spec×d_router` and `d_router×1`).

    Compared to ProjectedBridge: the specialist sees its NATIVE input
    distribution (token embeddings → its trained dynamics), so its
    hidden state is meaningful rather than a fragile response to a
    learned projection. The router's job becomes "where in the
    sequence is the specialist's expertise relevant?" — which is the
    biological intuition for higher-order areas reading from primary
    ones.
    """

    def __init__(self, d_router: int, d_specialist: int, init_open: bool = True):
        super().__init__()
        self.recv = nn.Linear(d_specialist, d_router)
        self.gate = nn.Linear(d_router, 1)
        # Learnable per-bridge scale, init small so a new synapse starts
        # near-identity (it doesn't try to compete with the router base
        # at step 1). Mirrors the `scale` parameter on each kernel layer
        # (init 0.01) — gives the optimizer the same "open it up gradually"
        # dynamics. Critical for stability when stacking multiple synapses.
        self.scale = nn.Parameter(torch.tensor(0.1))
        if init_open:
            nn.init.normal_(self.recv.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.recv.bias)
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)
        else:
            nn.init.zeros_(self.recv.weight)
            nn.init.zeros_(self.recv.bias)
            nn.init.zeros_(self.gate.weight)
            nn.init.constant_(self.gate.bias, -3.0)

    def forward(self, x_router, spec_hidden):
        """spec_hidden: (B, L, d_spec) — pre-computed once outside this fn."""
        gate = torch.sigmoid(self.gate(x_router))   # (B, L, 1)
        y = self.recv(spec_hidden)                  # (B, L, d_router)
        return self.scale * gate * y


class Bridge(nn.Module):
    """v1 synapse: router register state ↔ specialist register state."""

    def __init__(self, d_router: int, d_specialist: int, init_open: bool = True):
        super().__init__()
        self.send = nn.Linear(d_router, d_specialist)
        self.recv = nn.Linear(d_specialist, d_router)
        self.gate = nn.Linear(d_router, 1)
        if init_open:
            # Open gate at initialization (σ(0) = 0.5) — the synapse
            # contributes from step 1 and the router has to learn to
            # USE it well or close it. Recv init small-random so the
            # specialist's signal actually reaches the router. This is
            # the design that gives the synapse a fair shot when the
            # router-alone path is also non-trivial.
            nn.init.normal_(self.recv.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.recv.bias)
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)
        else:
            # Closed gate at initialization (σ(-3) ≈ 0.05) — synapse
            # is a no-op and the router has to LEARN to open it. Useful
            # when the router-alone path is strong and we want to see
            # whether the synapse pulls extra signal.
            nn.init.zeros_(self.recv.weight)
            nn.init.zeros_(self.recv.bias)
            nn.init.zeros_(self.gate.weight)
            nn.init.constant_(self.gate.bias, -3.0)

    def forward(self, x_router, specialist):
        """Run one synaptic invocation.

        Args:
          x_router: (B, L, d_router) — router state at the synapse layer
          specialist: a frozen ProgressiveModel with a `forward_from_hidden`

        Returns:
          (B, L, d_router) — the additive contribution to the router state
        """
        gate = torch.sigmoid(self.gate(x_router))           # (B, L, 1)
        x_spec = self.send(x_router)                        # (B, L, d_spec)
        # Specialist is frozen — no_grad both saves memory and makes
        # explicit that we're using its dynamics not training them.
        with torch.no_grad():
            y_spec = specialist.forward_from_hidden(x_spec) # (B, L, d_spec)
        y = self.recv(y_spec)                               # (B, L, d_router)
        return gate * y                                     # additive update


class RouterModel(nn.Module):
    """Router with synaptic invocation of frozen specialists.

    The router IS a normal `ProgressiveModel`; the synapse fires once,
    midway through the kernel-layer stack, blending each specialist's
    contribution into the router's hidden state via a Bridge.
    """

    def __init__(
        self,
        router_d_model: int = 32,
        router_d_state: int = 16,
        router_headdim: int = 16,
        router_n_layers: int = 1,
        specialists: list[ProgressiveModel] | None = None,
        bridge_kind: str = "attend",
    ):
        super().__init__()
        self.base = ProgressiveModel(
            d_model=router_d_model,
            d_state=router_d_state,
            expand=2,
            headdim=router_headdim,
        )
        for _ in range(router_n_layers):
            self.base.add_kernel_layer()

        # Freeze specialists; build one synapse per
        specialists = specialists or []
        self.specialists = nn.ModuleList(specialists)
        for sp in self.specialists:
            for p in sp.parameters():
                p.requires_grad = False
            sp.eval()

        self.bridge_kind = bridge_kind
        if bridge_kind == "attend":
            BridgeCls = AttendBridge
        elif bridge_kind == "project":
            BridgeCls = Bridge
        else:
            raise ValueError(f"unknown bridge_kind: {bridge_kind}")
        self.bridges = nn.ModuleList([
            BridgeCls(d_router=router_d_model, d_specialist=sp.d_model,
                      init_open=True)
            for sp in self.specialists
        ])

        # Insert the synapse halfway through the kernel stack. With
        # n_layers=1, that's "after layer 0, before final_norm".
        self.synapse_after_layer = max(0, router_n_layers - 1)

    def _specialist_hiddens(self, tokens):
        """Run each frozen specialist on the original tokens once, in
        no_grad. Returns a list of (B, L, d_spec) tensors, one per
        specialist. Used by the AttendBridge path.

        Specialists trained on a narrow input distribution can NaN
        when fed unfamiliar prefixes (e.g. count_above_threshold
        destabilizes on the `DUAL OR 0 1 ;` prefix from `dual_task`).
        We zero out NaNs so the bridge sees "specialist had nothing
        useful at these positions" rather than poisoning the router
        with NaN. The gate is free to close at those positions if the
        signal is consistently zero.
        """
        outs = []
        for sp in self.specialists:
            with torch.no_grad():
                x_sp = sp.embed_norm(sp.embed(tokens))
                x_sp = sp.forward_from_hidden(x_sp)
                x_sp = torch.nan_to_num(x_sp, nan=0.0, posinf=0.0, neginf=0.0)
            outs.append(x_sp)
        return outs

    def forward(self, tokens):
        # Pre-compute specialist hidden states once (attend mode only;
        # for project mode each Bridge runs its own forward_from_hidden
        # on a router-projected state).
        spec_hiddens = (self._specialist_hiddens(tokens)
                        if self.bridge_kind == "attend" else None)

        x = self.base.embed_norm(self.base.embed(tokens))
        kernel = list(self.base.kernel_layers)
        for i, layer in enumerate(kernel):
            scale = layer["scale"][0]
            x = x + scale * layer["block"](layer["norm"](x))
            if i == self.synapse_after_layer:
                for j, (sp, br) in enumerate(zip(self.specialists, self.bridges)):
                    if self.bridge_kind == "attend":
                        x = x + br(x, spec_hiddens[j])
                    else:
                        x = x + br(x, sp)
        for layer in self.base.cortex_layers:
            scale = layer["scale"][0]
            x = x + scale * layer["block"](layer["norm"](x))
        x = self.base.final_norm(x)
        return self.base.head(x)

    def gate_stats(self, tokens):
        """Mean gate value per synapse on a batch — probe for usage."""
        x = self.base.embed_norm(self.base.embed(tokens))
        for layer in self.base.kernel_layers:
            scale = layer["scale"][0]
            x = x + scale * layer["block"](layer["norm"](x))
        out = []
        for br in self.bridges:
            with torch.no_grad():
                g = torch.sigmoid(br.gate(x))
                out.append(float(g.mean()))
        return out


def load_specialist(pt_path: str | Path, device: str = "cpu") -> ProgressiveModel:
    """Load a saved .pt as a frozen ProgressiveModel ready to be
    invoked from a router. Reconstructs the architecture from the
    checkpoint's `config` and adds the right number of layers."""
    ck = torch.load(str(pt_path), map_location=device, weights_only=False)
    cfg = ck.get("config", {})
    model = ProgressiveModel(
        d_model=cfg.get("d_model", 64),
        d_state=cfg.get("d_state", 16),
        expand=2,
        headdim=cfg.get("headdim", 16),
    ).to(device)
    for _ in range(cfg.get("n_kernel_layers", 1)):
        model.add_kernel_layer()
    model.load_state_dict(ck["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model
