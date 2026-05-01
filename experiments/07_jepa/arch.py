"""ThoughtHead, JEPA loss, SIGreg loss.

W2 of the JEPA-Cortex plan. Three small additions on top of the existing
CortexLM in cortex_counting.py:

  ThoughtHead   — small MLP from student d_model -> teacher D_teacher
  jepa_loss     — predict-the-next-thought regression on student residuals
  sigreg_loss   — Cramér-Wold isotropy via 1-D shadow tests, anti-collapse
                  on the prompt-level intent embedding

These are deliberately stateless and trainer-agnostic. Nothing here imports
the trainer; the trainer imports these.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Thought head
# ---------------------------------------------------------------------------
class ThoughtHead(nn.Module):
    """Project student residual stream to teacher hidden dim.

    Two-layer MLP. A single linear underfits because the student d_model
    (96 to 192 in our regime) is much smaller than the teacher D_teacher
    (1536 for Qwen-2.5-1.5B). The hidden width is set to 4× student
    d_model by default — same factor as a transformer FFN — which is
    enough on the 4070 Ti without inflating param count past the budget.
    """

    def __init__(self, d_model: int, d_teacher: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or 4 * d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_teacher),
        )
        # Keep init small: in early training the byte CE has to dominate
        # so the LM head can lock in basic fluency before we ask the
        # residual to encode teacher-shaped thoughts. Bigger init here
        # caused the LM head to never escape uniform-noise output.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model). Returns (B, L, d_teacher).
        return self.net(x)


# ---------------------------------------------------------------------------
# JEPA loss
# ---------------------------------------------------------------------------
def jepa_loss(student_thoughts: torch.Tensor,    # (B, L, D_teacher)
              teacher_thoughts: torch.Tensor,    # (B, K, D_teacher)
              thought_byte_pos: torch.Tensor,    # (B, K) long — student byte idx
              thought_pad_mask: torch.Tensor,    # (B, K) bool
              stride_bytes: int = 16) -> torch.Tensor:
    """Smooth-L1 between predicted-next-thought and actual-next-thought.

    Mechanics. For each teacher thought at student byte position p, gather
    the student's residual at position p - stride_bytes (the position from
    which the student should have been predicting forward). Compare the
    student's projected residual there to the teacher hidden at p.

    Why predict-the-next instead of same-position. Same-position lets the
    head learn an identity on whatever is locally available; the SSM never
    has to encode anything future-relevant. Shifting the target back by one
    stride forces the student state to carry information about what the
    teacher will be saying STRIDE_BYTES later — that's the JEPA bet, the
    same property that makes LeJEPA's video model develop continuity.

    Why smooth-L1 over MSE / cosine. MSE is sensitive to teacher activation
    outliers from rare BPE tokens (numerals, punctuation, code switches).
    Cosine throws away magnitude, but Qwen activations encode confidence
    in their norms — losing that hurts. Smooth-L1 splits the difference.
    """
    B, K, D = teacher_thoughts.shape
    L = student_thoughts.size(1)
    src = (thought_byte_pos - stride_bytes).clamp(min=0, max=L - 1)  # (B, K)
    idx = src.unsqueeze(-1).expand(-1, -1, D)                         # (B, K, D)
    pred = torch.gather(student_thoughts, 1, idx)                     # (B, K, D)
    raw = F.smooth_l1_loss(pred, teacher_thoughts, reduction="none").mean(-1)
    denom = thought_pad_mask.float().sum().clamp_min(1.0)
    return (raw * thought_pad_mask.float()).sum() / denom


# ---------------------------------------------------------------------------
# Conversational JEPA loss — predict response-end teacher hidden from
# student's end-of-prompt residual.
# ---------------------------------------------------------------------------
def conv_jepa_loss(student_thoughts: torch.Tensor,         # (B, L, D_teacher)
                   teacher_response_hidden: torch.Tensor,  # (B, D_teacher)
                   prompt_end_pos: torch.Tensor,           # (B,) long
                   pad_mask: torch.Tensor | None = None    # (B,) bool
                   ) -> torch.Tensor:
    """Predict the teacher's response-end hidden from the student's
    end-of-prompt residual. The whole point: a model on autopilot — one that
    produces fluent text but ignores the prompt — cannot do this. Its
    end-of-prompt residual is just a function of the prompt's surface form
    and carries no information about the prompt-conditional response. The
    plain `jepa_loss` doesn't catch this because its same-sample shifted
    target lets the head learn local continuation; conv-jepa forces a
    cross-segment latent commitment.

    student_thoughts has already been projected to teacher dim by ThoughtHead.
    """
    B, L, D = student_thoughts.shape
    src = prompt_end_pos.clamp(0, L - 1)
    idx = src.view(B, 1, 1).expand(-1, 1, D)
    pred = student_thoughts.gather(1, idx).squeeze(1)        # (B, D)
    raw = F.smooth_l1_loss(pred, teacher_response_hidden, reduction="none").mean(-1)
    if pad_mask is None:
        return raw.mean()
    m = pad_mask.float()
    return (raw * m).sum() / m.sum().clamp_min(1.0)


# ---------------------------------------------------------------------------
# SIGreg — Cramér-Wold isotropy regularizer
# ---------------------------------------------------------------------------
def sigreg_loss(intent: torch.Tensor, n_directions: int = 1024) -> torch.Tensor:
    """Anti-collapse on the prompt-level intent embedding.

    intent: (B, D_intent). One vector per sample, typically a mean-pool of
        the student residual over the prompt bytes (computed by the trainer
        before this is called).

    Two terms enforce isotropic-Gaussian intent across the batch:

      shape: per-direction KS distance.
          1. Center across batch.
          2. Project onto n_directions random unit vectors.
          3. Standardize each projection and compare to the inverse-CDF
             of N(0, 1) at (i - 0.5) / B. Mean |deviation| over
             directions and rank-positions.
          The Cramér-Wold theorem says: if every 1-D projection is
          Gaussian, the joint distribution is Gaussian. We get an
          isotropy-shape signal from cheap 1-D tests.

      mag: per-dim variance should be 1.
          Standardization in `shape` makes the test scale-invariant —
          a near-zero batch with a tiny bit of noise passes the
          shape test trivially. The magnitude term penalizes that:
          (var_per_dim - 1)^2, averaged over dims. With shape it
          turns "be Gaussian-shaped" into "be standard isotropic Gaussian."

    Failure mode to keep in mind. If the intent space is intrinsically
    low-dimensional (e.g. the corpus is only counting tasks), forcing
    isotropy actively hurts. Mitigated by keeping the topic distribution
    in the teacher corpus wide; if loss-curves show byte CE rising while
    sigreg falls, the corpus diversity is the suspect, not the regularizer.
    """
    if intent.size(0) < 4:
        # Too small a batch for a meaningful KS test; return zero rather
        # than a noisy gradient.
        return intent.new_zeros(())

    z = intent - intent.mean(dim=0, keepdim=True)
    B, D = z.shape

    # Magnitude/scale calibration: per-dim variance should be 1.
    var_per_dim = z.var(dim=0, unbiased=True)
    mag = (var_per_dim - 1.0).pow(2).mean()

    # Shape: project, standardize, compare to N(0, 1) quantiles.
    U = torch.randn(D, n_directions, device=z.device, dtype=z.dtype)
    U = U / U.norm(dim=0, keepdim=True).clamp_min(1e-6)
    s = z @ U                                                       # (B, n_dir)
    s = (s - s.mean(0, keepdim=True)) / s.std(0, keepdim=True).clamp_min(1e-6)
    s_sorted, _ = torch.sort(s, dim=0)
    quantiles = (torch.arange(1, B + 1, device=s.device, dtype=s.dtype) - 0.5) / B
    # icdf for N(0,1) via probit; std_normal = sqrt(2) * erfinv(2*q - 1)
    expected = torch.special.erfinv(2 * quantiles - 1) * (2 ** 0.5)
    shape = (s_sorted - expected.unsqueeze(-1)).abs().mean()

    return shape + mag


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def masked_mean_pool(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool x over positions where mask is True.

    x:    (B, L, D)
    mask: (B, L) bool

    Returns: (B, D). Empty masks yield the zero vector.
    """
    m = mask.float().unsqueeze(-1)        # (B, L, 1)
    s = (x * m).sum(dim=1)
    n = m.sum(dim=1).clamp_min(1.0)
    return s / n


def prompt_mask(prompt_lens: torch.Tensor, L: int) -> torch.Tensor:
    """Build a (B, L) bool mask that is True for prompt positions only."""
    arange = torch.arange(L, device=prompt_lens.device).unsqueeze(0)
    return arange < prompt_lens.unsqueeze(1)
