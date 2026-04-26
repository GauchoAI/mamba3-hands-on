"""synapse — register-space invocation of frozen specialist models.

The router learns to project its own SSM state into a specialist's
register space, run the specialist with that projected state as input,
and read the result back into its own state via a learned linear bridge.

No tokens transit. No vocabulary expansion. The bridge is three small
matrices per (router, specialist) pair:

    W_send : (d_router → d_specialist)  what the router asks
    W_recv : (d_specialist → d_router)  how the router reads back
    W_g    : (d_router → 1)              when to invoke (soft gate)

The specialist stays frozen; only the bridges and the router's own
weights are trained. Differentiable end-to-end via the soft gate.

Mental model: each bridge is a synapse between two register populations.
With N specialists, the router sprouts N synapses but does not grow.
"""
from __future__ import annotations
from pathlib import Path
import torch
import torch.nn as nn

from progressive_model import ProgressiveModel, VOCAB_SIZE


class Bridge(nn.Module):
    """One synapse: router register state ↔ specialist register state."""

    def __init__(self, d_router: int, d_specialist: int):
        super().__init__()
        self.send = nn.Linear(d_router, d_specialist)
        self.recv = nn.Linear(d_specialist, d_router)
        self.gate = nn.Linear(d_router, 1)
        # Init the recv path near zero so the synapse starts as a no-op
        # and the router has to LEARN to use the specialist (the gate
        # opens only if it helps the loss). This avoids the synapse
        # initially injecting noise that destabilizes the router.
        nn.init.zeros_(self.recv.weight)
        nn.init.zeros_(self.recv.bias)
        # Negative gate bias so the initial gate is ≈ 0 (closed synapse).
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -3.0)  # σ(-3) ≈ 0.047

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

        self.bridges = nn.ModuleList([
            Bridge(d_router=router_d_model, d_specialist=sp.d_model)
            for sp in self.specialists
        ])

        # Insert the synapse halfway through the kernel stack. With
        # n_layers=1, that's "after layer 0, before final_norm".
        self.synapse_after_layer = max(0, router_n_layers - 1)

    def forward(self, tokens):
        x = self.base.embed_norm(self.base.embed(tokens))
        kernel = list(self.base.kernel_layers)
        # Pre-synapse layers
        for i, layer in enumerate(kernel):
            scale = layer["scale"][0]
            x = x + scale * layer["block"](layer["norm"](x))
            if i == self.synapse_after_layer:
                # Fire all synapses additively (each one already gated)
                for sp, br in zip(self.specialists, self.bridges):
                    x = x + br(x, sp)
        # Cortex (if any) and final norm + head
        for layer in self.base.cortex_layers:
            scale = layer["scale"][0]
            x = x + scale * layer["block"](layer["norm"](x))
        x = self.base.final_norm(x)
        return self.base.head(x)

    def gate_stats(self, tokens):
        """Return mean gate value per synapse on a batch — a probe for
        whether the router is actually using the specialists."""
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
