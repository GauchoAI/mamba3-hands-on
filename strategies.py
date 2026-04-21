"""
Training strategies — drop-in components for the worker.

Each strategy is a few lines. Workers select via config["strategy"].
Evolution mutates between them.
"""
import torch
import torch.nn as nn
import math


# ── SAM (Sharpness-Aware Minimization) ──────────────────────────────

class SAM:
    """
    Finds wide flat valleys instead of sharp narrow ones.
    Two forward passes per step: one to find the worst nearby point,
    one to compute the actual gradient at that point.
    """
    def __init__(self, model, base_optimizer, rho=0.05):
        self.model = model
        self.base_optimizer = base_optimizer
        self.rho = rho

    def step(self, loss_fn):
        """Call with a closure that returns loss."""
        # First pass: compute gradient at current point
        loss = loss_fn()
        loss.backward()

        # Save current params and compute epsilon (perturbation)
        with torch.no_grad():
            old_params = []
            for p in self.model.parameters():
                if p.grad is None:
                    old_params.append(None)
                    continue
                old_params.append(p.data.clone())
                # Move to worst nearby point: p + rho * grad / ||grad||
                grad_norm = p.grad.norm()
                if grad_norm > 1e-12:
                    p.data.add_(p.grad, alpha=self.rho / grad_norm)

        # Second pass: compute gradient at the worst nearby point
        self.base_optimizer.zero_grad()
        loss2 = loss_fn()
        loss2.backward()

        # Restore original params
        with torch.no_grad():
            for p, old in zip(self.model.parameters(), old_params):
                if old is not None:
                    p.data.copy_(old)

        # Step with the gradient from the worst nearby point
        self.base_optimizer.step()
        return loss.item()


# ── Label Smoothing ─────────────────────────────────────────────────

def label_smoothed_cross_entropy(logits, targets, smoothing=0.1, reduction='none'):
    """
    Cross-entropy with label smoothing.
    Instead of [0, 0, 1, 0], target becomes [0.033, 0.033, 0.9, 0.033].
    Prevents overconfidence.
    """
    n_classes = logits.size(-1)
    log_probs = torch.log_softmax(logits, dim=-1)

    # Smooth target: (1-ε) on correct class, ε/(n-1) on others
    nll = -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    smooth = -log_probs.sum(dim=-1) / n_classes

    loss = (1 - smoothing) * nll + smoothing * smooth

    if reduction == 'mean':
        return loss.mean()
    return loss


# ── Lion Optimizer ──────────────────────────────────────────────────

class Lion(torch.optim.Optimizer):
    """
    Google's Lion optimizer. Uses sign of momentum instead of magnitude.
    50% less memory than Adam, often faster on small models.

    From: Chen et al. 2023, "Symbolic Discovery of Optimization Algorithms"
    """
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']

                # Weight decay
                if group['weight_decay'] > 0:
                    p.mul_(1 - group['lr'] * group['weight_decay'])

                # Update: sign of interpolation between momentum and gradient
                update = exp_avg * beta1 + grad * (1 - beta1)
                p.add_(update.sign(), alpha=-group['lr'])

                # Update momentum
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)


# ── Warm Restarts (Cosine Annealing with Restarts) ──────────────────

class WarmRestartScheduler:
    """
    Cosine annealing that periodically resets LR to high.
    Each restart lets the model escape local minima.
    """
    def __init__(self, optimizer, T_0=100, T_mult=2, eta_min=1e-5):
        self.optimizer = optimizer
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.base_lrs = [g['lr'] for g in optimizer.param_groups]
        self.step_count = 0
        self.current_T = T_0

    def step(self):
        self.step_count += 1
        if self.step_count >= self.current_T:
            # Restart!
            self.step_count = 0
            self.current_T = int(self.current_T * self.T_mult)

        # Cosine annealing within current period
        progress = self.step_count / self.current_T
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = self.eta_min + (base_lr - self.eta_min) * \
                                (1 + math.cos(math.pi * progress)) / 2


# ── Noise Injection ─────────────────────────────────────────────────

def inject_noise(model, noise_scale=0.001):
    """Add random noise to all parameters. Helps escape local minima."""
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * noise_scale)


# ── Quick test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    # Test all strategies
    model = nn.Linear(10, 5)

    # SAM
    base_opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sam = SAM(model, base_opt, rho=0.05)
    x = torch.randn(4, 10)
    sam.step(lambda: nn.functional.cross_entropy(model(x), torch.randint(0, 5, (4,))))
    print("SAM: OK")

    # Label smoothing
    logits = torch.randn(4, 10)
    targets = torch.randint(0, 10, (4,))
    loss = label_smoothed_cross_entropy(logits, targets, smoothing=0.1, reduction='mean')
    print(f"Label smoothing: loss={loss.item():.3f} OK")

    # Lion
    model2 = nn.Linear(10, 5)
    lion = Lion(model2.parameters(), lr=1e-4, weight_decay=0.01)
    loss = nn.functional.cross_entropy(model2(x), torch.randint(0, 5, (4,)))
    loss.backward()
    lion.step()
    print("Lion: OK")

    # Warm restarts
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = WarmRestartScheduler(opt, T_0=10)
    for i in range(25):
        sched.step()
    print(f"Warm restarts: lr={opt.param_groups[0]['lr']:.6f} OK")

    # Noise injection
    inject_noise(model, noise_scale=0.001)
    print("Noise injection: OK")

    print("\nAll strategies passed.")
