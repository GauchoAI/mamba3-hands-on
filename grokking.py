"""
Grokking utilities — StableMax + PerpGrad (⊥Grad).

From: "Grokking at the Edge of Numerical Stability" (Prieto et al., 2025)
       arXiv:2501.04697

StableMax: replaces softmax's exponential with a linear ramp, preventing
  floating-point absorption errors that kill gradients (Softmax Collapse).

PerpGrad: projects gradients to be orthogonal to the weight vector,
  preventing Naive Loss Minimization (scaling logits without changing
  predictions). Forces every step to actually change the model's behavior.

Together: 10-12x speedup to grokking, works without weight decay.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── StableMax ───────────────────────────────────────────────────────

def _stable_s(x):
    """Piecewise activation: linear for x>=0, rational for x<0."""
    x = x.clamp(-1e6, 1e6)        # prevent inf from extreme logits
    pos = x + 1                    # linear growth (not exponential)
    neg = 1.0 / (1.0 - x + 1e-8)  # smooth decay toward 0 (epsilon prevents div/0)
    return torch.where(x >= 0, pos, neg)


def stablemax(logits, dim=-1):
    """Drop-in replacement for F.softmax. Numerically stable."""
    s = _stable_s(logits)
    return s / s.sum(dim=dim, keepdim=True)


def stable_cross_entropy(logits, targets, reduction='none'):
    """Cross-entropy using StableMax instead of Softmax.
    Drop-in replacement for F.cross_entropy."""
    probs = stablemax(logits, dim=-1)
    # Clamp to avoid log(0)
    log_probs = torch.log(probs.clamp(min=1e-12))
    # NLL
    loss = F.nll_loss(log_probs, targets, reduction=reduction)
    return loss


class StableMaxCrossEntropy(nn.Module):
    """Module wrapper for stable_cross_entropy."""
    def forward(self, logits, targets):
        return stable_cross_entropy(logits, targets, reduction='mean')


# ── PerpGrad (⊥Grad) ───────────────────────────────────────────────

class PerpGradOptimizer:
    """
    Wraps any optimizer to project gradients orthogonal to weights.

    After loss.backward(), call project() before optimizer.step().
    This removes the Naive Loss Minimization component — gradient
    steps that just scale logits without changing predictions.

    Usage:
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        perp = PerpGradOptimizer(model)

        loss.backward()
        perp.project()     # ← remove NLM component
        opt.step()
    """
    def __init__(self, model):
        self.model = model

    def project(self):
        """Project gradients to be orthogonal to weight vectors."""
        for p in self.model.parameters():
            if p.grad is None:
                continue
            # θ·∇L / (θ·θ)
            theta = p.data
            grad = p.grad.data
            dot = (theta * grad).sum()
            norm_sq = (theta * theta).sum()
            if norm_sq > 1e-12:
                # Remove the component along θ
                grad.sub_(theta, alpha=(dot / norm_sq).item())


# ── Combined loss for the experiment ────────────────────────────────

def compute_loss_stable(logits, tokens, sep_positions, special_tokens):
    """Vectorized loss using StableMax instead of Softmax."""
    B, L, V = logits.shape
    device = logits.device

    mask = torch.zeros(B, L, device=device)
    for b in range(B):
        sep = sep_positions[b]
        mask[b, sep:L-1] = 1.0

    pad_mask = (tokens != special_tokens["<PAD>"]).float()
    target_valid = pad_mask[:, 1:]
    pred_mask = mask[:, :L-1] * target_valid

    if pred_mask.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    logits_flat = logits[:, :L-1].reshape(-1, V)
    targets_flat = tokens[:, 1:].reshape(-1)
    mask_flat = pred_mask.reshape(-1)

    loss_all = stable_cross_entropy(logits_flat, targets_flat, reduction='none')
    loss = (loss_all * mask_flat).sum() / mask_flat.sum()
    return loss


# ── Quick test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    # Test StableMax
    logits = torch.randn(4, 10)
    probs = stablemax(logits)
    print(f"StableMax probs sum: {probs.sum(dim=-1)}")  # should be ~1.0
    assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5)

    # Test with extreme logits (where softmax would collapse)
    extreme = torch.tensor([[100.0, -100.0, 0.0]])
    soft_probs = F.softmax(extreme, dim=-1)
    stable_probs = stablemax(extreme, dim=-1)
    print(f"Softmax  on [100, -100, 0]: {soft_probs}")
    print(f"StableMax on [100, -100, 0]: {stable_probs}")

    # Test stable CE
    targets = torch.tensor([0, 1, 2, 0])
    loss = stable_cross_entropy(logits, targets, reduction='mean')
    print(f"Stable CE loss: {loss.item():.4f}")

    # Test PerpGrad
    model = nn.Linear(10, 5)
    x = torch.randn(4, 10)
    y = model(x)
    loss = y.sum()
    loss.backward()

    perp = PerpGradOptimizer(model)
    # Check gradient has component along weight before projection
    w = model.weight.data
    g = model.weight.grad.data
    dot_before = (w * g).sum().item()

    perp.project()

    # After projection, gradient should be orthogonal to weight
    dot_after = (w * model.weight.grad.data).sum().item()
    print(f"PerpGrad: dot before={dot_before:.4f}, after={dot_after:.6f}")
    assert abs(dot_after) < 1e-4, "Gradient should be orthogonal to weights"

    print("\nAll tests passed.")
