"""
Diagnostician: reads telemetry from cycle_history, detects signals,
prescribes targeted mutations as bias (not direct intervention).

The diagnostician does NOT bypass the mutation gate. It returns a
diagnostic_bias that the GA uses to make a more informed mutation.
The mutation still goes through champion-challenger.

Usage:
    from diagnostician import Diagnostician
    diag = Diagnostician(db)
    signals = diag.diagnose(task)
    if signals:
        bias = diag.prescribe(signals[0], task, current_config)
        # bias is passed to mutate_config as diagnostic_bias
"""
import json
from collections import defaultdict


class Diagnostician:
    """Reads cycle_history, detects signals, prescribes mutation biases."""

    def __init__(self, db):
        self.db = db

    def diagnose(self, task):
        """Detect all active signals for a task. Returns list of DiagnosticEvents."""
        signals = []
        recent = self.db.get_cycle_history(task, last_n=20)
        if len(recent) < 5:
            return signals

        # Ensure chronological order
        recent.sort(key=lambda r: r.get("cycle", 0))

        signals.extend(self._check_dead_gradients(task, recent))
        signals.extend(self._check_oscillating_loss(task, recent))
        signals.extend(self._check_accuracy_oscillation(task, recent))
        signals.extend(self._check_loss_acc_divergence(task, recent))
        signals.extend(self._check_param_explosion(task, recent))
        signals.extend(self._check_grad_spike(task, recent))
        signals.extend(self._check_mode_collapse(task, recent))
        signals.extend(self._check_convergence_speed(task, recent))

        return signals

    def prescribe(self, signal, task, config):
        """Given a signal, return (prescription_type, params, provenance_entry).

        Returns None if this prescription has been tried too many times.
        """
        prescriptions = self._get_prescriptions(signal["signal"])

        for rx_type, rx_params in prescriptions:
            if self.db.should_prescribe(task, signal["signal"], rx_type):
                provenance = {
                    "source": "diagnostic",
                    "signal": signal["signal"],
                    "prescription": rx_type,
                    "evidence": signal["evidence"],
                }
                return {
                    "type": rx_type,
                    "params": rx_params,
                    "signal": signal["signal"],
                    "provenance": provenance,
                }

        return None  # all prescriptions exhausted

    # ── Signal detectors ───────────────────────────────────────────

    def _check_dead_gradients(self, task, recent):
        grads = [r["grad_norm"] for r in recent if r.get("grad_norm") is not None]
        if not grads:
            return []
        avg = sum(grads) / len(grads)
        if avg < 0.1:
            return [{
                "signal": "dead_grad",
                "task": task,
                "evidence": {"avg_grad_norm": round(avg, 4), "n_cycles": len(grads), "threshold": 0.1},
            }]
        return []

    def _check_oscillating_loss(self, task, recent):
        losses = [r["loss"] for r in recent if r.get("loss") is not None]
        if len(losses) < 10:
            return []
        mean_loss = sum(losses) / len(losses)
        var_loss = sum((l - mean_loss) ** 2 for l in losses) / len(losses)
        trend = losses[-1] - losses[0]
        if var_loss > 0.01 and abs(trend) < 0.001:
            return [{
                "signal": "oscillating_loss",
                "task": task,
                "evidence": {"variance": round(var_loss, 4), "trend": round(trend, 4), "mean": round(mean_loss, 4)},
            }]
        return []

    def _check_accuracy_oscillation(self, task, recent):
        accs = [r["accuracy"] for r in recent if r.get("accuracy") is not None]
        if len(accs) < 10:
            return []
        spread = max(accs) - min(accs)
        best = max(accs)
        if spread > 0.3 and best > 0.7:
            return [{
                "signal": "accuracy_oscillation",
                "task": task,
                "evidence": {"spread": round(spread, 3), "best": round(best, 3), "min": round(min(accs), 3)},
            }]
        return []

    def _check_loss_acc_divergence(self, task, recent):
        if len(recent) < 10:
            return []
        losses = [r["loss"] for r in recent if r.get("loss") is not None]
        accs = [r["accuracy"] for r in recent if r.get("accuracy") is not None]
        if len(losses) < 10 or len(accs) < 10:
            return []
        loss_trend = losses[-1] - losses[0]
        acc_trend = accs[-1] - accs[0]
        if loss_trend < -0.01 and abs(acc_trend) < 0.02:
            return [{
                "signal": "loss_acc_divergence",
                "task": task,
                "evidence": {"loss_trend": round(loss_trend, 4), "acc_trend": round(acc_trend, 4)},
            }]
        return []

    def _check_param_explosion(self, task, recent):
        pnorms = [r["param_norm"] for r in recent if r.get("param_norm") is not None]
        if len(pnorms) < 10:
            return []
        growth = pnorms[-1] / max(pnorms[0], 1e-6)
        if growth > 1.5:
            return [{
                "signal": "param_explosion",
                "task": task,
                "evidence": {"growth_ratio": round(growth, 3), "start": round(pnorms[0], 1), "end": round(pnorms[-1], 1)},
            }]
        return []

    def _check_grad_spike(self, task, recent):
        grads = [r["grad_norm"] for r in recent if r.get("grad_norm") is not None]
        if len(grads) < 5:
            return []
        avg_prev = sum(grads[:-1]) / max(len(grads) - 1, 1)
        if avg_prev > 0 and grads[-1] / max(avg_prev, 1e-6) > 10:
            return [{
                "signal": "grad_spike",
                "task": task,
                "evidence": {"current": round(grads[-1], 2), "avg_previous": round(avg_prev, 2), "ratio": round(grads[-1] / max(avg_prev, 1e-6), 1)},
            }]
        return []

    def _check_mode_collapse(self, task, recent):
        accs = [r["accuracy"] for r in recent if r.get("accuracy") is not None]
        if len(accs) < 15:
            return []
        unique = len(set(round(a, 2) for a in accs))
        if unique <= 2:
            return [{
                "signal": "mode_collapse",
                "task": task,
                "evidence": {"unique_accuracies": unique, "n_cycles": len(accs), "value": round(accs[-1], 2)},
            }]
        return []

    def _check_convergence_speed(self, task, recent):
        accs = [r["accuracy"] for r in recent if r.get("accuracy") is not None]
        if len(accs) < 20:
            return []
        mid = len(accs) // 2
        early_rate = (accs[mid] - accs[0]) / max(mid, 1)
        late_rate = (accs[-1] - accs[mid]) / max(len(accs) - mid, 1)

        if late_rate > early_rate and late_rate > 0.005:
            return [{
                "signal": "accelerating",
                "task": task,
                "evidence": {"early_rate": round(early_rate, 4), "late_rate": round(late_rate, 4)},
            }]
        if early_rate > 0.02 and late_rate < 0.001:
            return [{
                "signal": "early_plateau",
                "task": task,
                "evidence": {"early_rate": round(early_rate, 4), "late_rate": round(late_rate, 4)},
            }]
        if early_rate < 0.001 and late_rate < 0.001 and len(accs) >= 20:
            return [{
                "signal": "never_learned",
                "task": task,
                "evidence": {"early_rate": round(early_rate, 4), "late_rate": round(late_rate, 4), "cycles": len(accs)},
            }]
        return []

    # ── Prescriptions per signal ───────────────────────────────────

    def _get_prescriptions(self, signal):
        """Return ordered list of (type, params) prescriptions for a signal."""
        prescriptions = {
            "dead_grad": [
                ("noise_injection", {"noise_scale": 0.005}),
                ("warm_restart", {"warm_restarts": True}),
                ("teacher_distill", {}),  # best teacher picked at runtime
                ("lr_spike", {"lr_multiply": 10}),
            ],
            "oscillating_loss": [
                ("lr_reduce", {"lr_multiply": 0.3}),
                ("batch_increase", {"batch_size_multiply": 2}),
                ("wd_increase", {"weight_decay_add": 0.05}),
            ],
            "accuracy_oscillation": [
                ("label_smooth", {"loss_fn": "label_smooth"}),
                ("lr_reduce", {"lr_multiply": 0.5}),
            ],
            "loss_acc_divergence": [
                ("focal_loss", {"loss_fn": "focal"}),
                ("perpgrad", {"use_perp": True, "weight_decay": 0.0}),
            ],
            "param_explosion": [
                ("wd_increase", {"weight_decay_add": 0.1}),
                ("lr_reduce", {"lr_multiply": 0.5}),
                ("noise_injection", {"noise_scale": 0.001}),
            ],
            "grad_spike": [
                ("wait", {}),  # do nothing for 3 cycles — might be grokking
            ],
            "mode_collapse": [
                ("noise_injection", {"noise_scale": 0.01}),
                ("lr_spike", {"lr_multiply": 5}),
                ("focal_loss", {"loss_fn": "focal"}),
            ],
            "accelerating": [
                ("protect", {}),  # do NOT mutate — let it run
            ],
            "early_plateau": [
                ("wd_increase", {"weight_decay_add": 0.05}),
                ("noise_injection", {"noise_scale": 0.003}),
            ],
            "never_learned": [
                ("radical", {}),  # needs a completely different config
            ],
        }
        return prescriptions.get(signal, [])

    def apply_prescription(self, config, prescription):
        """Apply a prescription's params to a config. Returns new config."""
        cfg = config.copy()
        params = prescription["params"]

        if prescription["type"] == "protect":
            return None  # signal to NOT mutate

        if prescription["type"] == "wait":
            return None  # signal to NOT mutate

        if prescription["type"] == "radical":
            return None  # signal to use maximum severity GA

        # Apply param changes
        if "noise_scale" in params:
            cfg["noise_scale"] = params["noise_scale"]
        if "warm_restarts" in params:
            cfg["warm_restarts"] = params["warm_restarts"]
        if "loss_fn" in params:
            cfg["loss_fn"] = params["loss_fn"]
        if "use_perp" in params:
            cfg["use_perp"] = params["use_perp"]
        if "weight_decay" in params:
            cfg["weight_decay"] = params["weight_decay"]

        if "lr_multiply" in params:
            cfg["lr"] = cfg.get("lr", 1e-3) * params["lr_multiply"]
        if "batch_size_multiply" in params:
            cfg["batch_size"] = min(4096, int(cfg.get("batch_size", 256) * params["batch_size_multiply"]))
        if "weight_decay_add" in params:
            cfg["weight_decay"] = cfg.get("weight_decay", 0.0) + params["weight_decay_add"]

        # Teacher: pick best available
        if prescription["type"] == "teacher_distill":
            best = self.db.get_best_teachers_for_task(
                prescription.get("_task", ""), min_accuracy=0.01)
            if best:
                cfg["teacher_model"] = best[0][0]

        return cfg
