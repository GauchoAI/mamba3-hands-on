"""Mutation Registry — data-driven GA mutations.

Replaces hardcoded if/random blocks in coordinator.py with a registry
of mutation specs loaded from YAML. Node capabilities auto-extend the registry.
"""

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None


@dataclass
class MutationSpec:
    """One mutation the GA can apply to a config."""
    name: str
    key: str                              # config dict key to mutate
    type: str                             # categorical | continuous_log | discrete_nearby | boolean | toggle | teacher
    probability: float                    # base probability (before plateau amplification)
    max_probability: float = 1.0          # cap after amplification
    values: list | None = None            # for categorical/discrete
    default: Any = None
    min_val: float | None = None          # for continuous
    max_val: float | None = None
    spread: float = 0.3                   # for continuous_log (gaussian std)
    nearby_delta: int | None = None       # for discrete_nearby
    nearby_offsets: list | None = None    # [-1, 0, 1, 2] for layers
    nearby_extra: list | None = None     # [0, 32] extra delta values for d_model
    nearby_probability: float = 0.7       # prob of nearby vs wild
    min_value: int | None = None          # floor for discrete_nearby
    implies: dict | None = None           # value → {key: forced_value}
    side_effects: list | None = None      # [{constrain: key, rule: "min_of_self"}]
    respect_diagnostic: bool = False      # skip if diagnostic already set this key
    source: str = "builtin"               # builtin | node_capability | plugin


class MutationRegistry:
    """Data-driven mutation engine for the GA.

    Usage:
        registry = MutationRegistry()
        registry.load(["registry/mutations.yaml"])
        child, provenance = registry.apply(parent_config, plateau_severity=0.5)
    """

    def __init__(self):
        self.mutations: list[MutationSpec] = []
        self._by_name: dict[str, MutationSpec] = {}
        self._seed_configs: list[dict] = []

    def load(self, mutation_paths: list[str], seed_path: str | None = None):
        """Load mutation specs and seed configs from YAML files."""
        for p in mutation_paths:
            path = Path(p)
            if not path.exists():
                continue
            with open(path) as f:
                if yaml is not None:
                    data = yaml.safe_load(f)
                else:
                    data = _parse_mutations_yaml(path)
            for mdef in data.get("mutations", []):
                spec = self._spec_from_dict(mdef)
                self.register(spec)

        if seed_path:
            self._load_seeds(seed_path)

    def _spec_from_dict(self, d: dict) -> MutationSpec:
        """Build a MutationSpec from a YAML dict."""
        # Convert numeric string values in 'values' list
        values = d.get("values")
        if values:
            values = [self._coerce_value(v) for v in values]

        return MutationSpec(
            name=d["name"],
            key=d["key"],
            type=d["type"],
            probability=float(d.get("probability", 0.1)),
            max_probability=float(d.get("max_probability", 1.0)),
            values=values,
            default=d.get("default"),
            min_val=float(d["min"]) if "min" in d else None,
            max_val=float(d["max"]) if "max" in d else None,
            spread=float(d.get("spread", 0.3)),
            nearby_delta=d.get("nearby_delta"),
            nearby_offsets=d.get("nearby_offsets"),
            nearby_extra=d.get("nearby_extra"),
            nearby_probability=float(d.get("nearby_probability", 0.7)),
            min_value=d.get("min_value"),
            implies=d.get("implies"),
            side_effects=d.get("side_effects"),
            respect_diagnostic=d.get("respect_diagnostic", False),
            source=d.get("source", "builtin"),
        )

    @staticmethod
    def _coerce_value(v):
        """Coerce YAML values: '0.0' → 0.0, '64' → 64, 'adamw' → 'adamw'."""
        if isinstance(v, (int, float, bool)):
            return v
        if isinstance(v, str):
            try:
                if "." in v or "e" in v.lower():
                    return float(v)
                return int(v)
            except (ValueError, TypeError):
                return v
        return v

    def register(self, spec: MutationSpec):
        """Add or replace a mutation spec."""
        self._by_name[spec.name] = spec
        # Replace if exists, else append
        self.mutations = [m for m in self.mutations if m.name != spec.name]
        self.mutations.append(spec)

    def has(self, name: str) -> bool:
        return name in self._by_name

    def extend_from_capabilities(self, manifest: dict):
        """Auto-register mutations based on node capabilities.

        When a new node joins with backends=[rocm, jit], this adds
        'rocm' to the device mutation's values. The GA discovers
        whether it's competitive through champion/challenger.
        """
        backends = manifest.get("backends", [])

        # Extend device values
        device_spec = self._by_name.get("device")
        if device_spec and device_spec.values:
            for backend in backends:
                if backend in ("rocm", "mps", "xla") and backend not in device_spec.values:
                    device_spec.values.append(backend)

        # Extend scan_backend values
        scan_spec = self._by_name.get("scan_backend")
        if scan_spec and scan_spec.values:
            for backend in backends:
                if backend in ("triton", "jit", "flash") and backend not in scan_spec.values:
                    scan_spec.values.append(backend)

        # Register entirely new backend types
        for backend in backends:
            name = f"backend_{backend}"
            if backend not in ("cuda", "cpu", "triton", "jit") and not self.has(name):
                self.register(MutationSpec(
                    name=name,
                    key="backend",
                    type="categorical",
                    probability=0.1,
                    max_probability=0.5,
                    values=["pytorch", backend],
                    source="node_capability",
                ))

    def apply(self, parent_config: dict, plateau_severity: float = 0.0,
              diagnostic_bias: dict | None = None) -> tuple[dict, dict]:
        """Apply mutations to a parent config. Returns (child_config, provenance).

        This is the data-driven replacement for coordinator.mutate_config().
        Same logic, same behavior, but driven by registered MutationSpecs.
        """
        child = parent_config.copy()
        provenance = {}

        # Mark all inherited params
        for k, v in child.items():
            if k not in ("task",):
                provenance[k] = {"source": "inherited", "value": v}

        # Ensure defaults
        child.setdefault("loss_fn", "stable_ce")
        child.setdefault("backend", "pytorch")
        child.setdefault("steps_per_cycle", 200)

        # Apply diagnostic bias first (if provided)
        diagnostic_keys = set()
        if diagnostic_bias:
            self._apply_diagnostic(child, provenance, diagnostic_bias)
            diagnostic_keys = {k for k, v in provenance.items()
                               if v.get("source") == "diagnostic"}

        # Plateau amplifier
        amp = 1.0 + plateau_severity
        _sev = round(plateau_severity, 1)

        # Apply each registered mutation
        for spec in self.mutations:
            # Skip teacher mutations here — handled separately
            if spec.type == "teacher":
                continue

            # Skip if diagnostic already set this key
            if spec.respect_diagnostic and spec.key in diagnostic_keys:
                continue

            # Roll probability (amplified by plateau)
            prob = min(spec.probability * amp, spec.max_probability)
            if random.random() >= prob:
                continue

            # Apply the typed mutation
            old_val = child.get(spec.key, spec.default)
            new_val = self._apply_typed_mutation(spec, child, amp)

            if new_val is not None:
                child[spec.key] = new_val
                provenance[spec.key] = {
                    "source": "ga_mutation", "severity": _sev,
                    "value": new_val, "mutation": spec.name,
                }

                # Apply implications (e.g., cpu → jit)
                if spec.implies and isinstance(spec.implies, dict):
                    val_str = str(new_val)
                    if val_str in spec.implies:
                        for ik, iv in spec.implies[val_str].items():
                            child[ik] = self._coerce_value(iv)
                            provenance[ik] = {
                                "source": "ga_mutation", "severity": _sev,
                                "value": child[ik], "implied_by": spec.name,
                            }

                # Apply side effects (e.g., headdim ≤ d_model)
                if spec.side_effects:
                    for effect in spec.side_effects:
                        if effect.get("rule") == "min_of_self":
                            target = effect["constrain"]
                            if target in child and child[target] > new_val:
                                child[target] = new_val

        # Teacher mutation — special handling (requires DB access)
        teacher_spec = self._by_name.get("teacher_model")
        if teacher_spec:
            prob = min(teacher_spec.probability * amp, teacher_spec.max_probability)
            if random.random() < prob:
                self._apply_teacher_mutation(child, provenance, _sev)

        # Radical mutation when severely stuck
        if plateau_severity >= 2.0 and random.random() < 0.3 and self._seed_configs:
            child = random.choice(self._seed_configs).copy()
            # Also randomize a few keys
            layers_spec = self._by_name.get("n_kernel_layers")
            if layers_spec:
                child["n_kernel_layers"] = self._apply_typed_mutation(
                    layers_spec, child, amp) or child.get("n_kernel_layers", 1)
            child["loss_fn"] = random.choice(["stable_ce", "ce", "focal", "label_smooth"])
            child["optimizer"] = random.choice(["adamw", "lion"])
            child["warm_restarts"] = random.choice([True, False])
            child["noise_scale"] = random.choice([0.0, 0.001, 0.005])

        # Auto-tag remaining mutations
        for k, v in child.items():
            if k in ("task", "steps_per_cycle", "backend"):
                continue
            if k in provenance and provenance[k]["source"] != "inherited":
                continue
            parent_v = parent_config.get(k)
            if v != parent_v and parent_v is not None:
                provenance[k] = {"source": "ga_mutation", "severity": _sev, "value": v}

        return child, provenance

    def _apply_typed_mutation(self, spec: MutationSpec, child: dict, amp: float):
        """Apply one mutation based on its type. Returns new value or None."""
        if spec.type == "categorical":
            return random.choice(spec.values)

        elif spec.type == "toggle":
            current = child.get(spec.key, spec.values[0] if spec.values else None)
            if spec.values and len(spec.values) == 2:
                return spec.values[1] if current == spec.values[0] else spec.values[0]
            return current

        elif spec.type == "boolean":
            current = child.get(spec.key, False)
            return not current

        elif spec.type == "continuous_log":
            current = child.get(spec.key, 1e-3)
            spread = spec.spread * amp
            factor = 2 ** random.gauss(0, spread)
            new_val = current * factor
            if spec.min_val is not None:
                new_val = max(spec.min_val, new_val)
            if spec.max_val is not None:
                new_val = min(spec.max_val, new_val)
            return new_val

        elif spec.type == "discrete_nearby":
            current = child.get(spec.key, spec.values[0] if spec.values else 64)

            if random.random() < spec.nearby_probability:
                # Nearby: offset from current
                if spec.nearby_offsets:
                    offsets = spec.nearby_offsets
                    nearby_vals = [max(spec.min_value or 1, current + off) for off in offsets]
                elif spec.nearby_delta:
                    deltas = spec.nearby_extra if spec.nearby_extra else [0]
                    nearby_vals = [max(spec.min_value or 1, current + d)
                                  for d in [-spec.nearby_delta] + deltas + [spec.nearby_delta, spec.nearby_delta * 2]]
                else:
                    nearby_vals = spec.values
                return random.choice(nearby_vals)
            else:
                # Wild: any value from the full list
                return random.choice(spec.values)

        return None

    def _apply_diagnostic(self, child: dict, provenance: dict, bias: dict):
        """Apply diagnostic prescription to child config."""
        rx_params = bias.get("params", {})
        signal = bias.get("signal", "unknown")
        rx_type = bias.get("type", "unknown")

        for k, v in rx_params.items():
            if k == "lr_multiply":
                child["lr"] = child.get("lr", 1e-3) * v
                provenance["lr"] = {"source": "diagnostic", "signal": signal,
                                    "prescription": rx_type, "value": child["lr"]}
            elif k == "batch_size_multiply":
                child["batch_size"] = min(4096, int(child.get("batch_size", 256) * v))
                provenance["batch_size"] = {"source": "diagnostic", "signal": signal,
                                            "prescription": rx_type, "value": child["batch_size"]}
            elif k == "weight_decay_add":
                child["weight_decay"] = child.get("weight_decay", 0.0) + v
                provenance["weight_decay"] = {"source": "diagnostic", "signal": signal,
                                              "prescription": rx_type, "value": child["weight_decay"]}
            else:
                child[k] = v
                provenance[k] = {"source": "diagnostic", "signal": signal,
                                 "prescription": rx_type, "value": v}

    def _apply_teacher_mutation(self, child: dict, provenance: dict, sev: float):
        """Mutate teacher_model — only teachers that beat current baseline."""
        task = child.get("task", "")
        if not task:
            return

        try:
            from state_db import StateDB
            _tdb = StateDB("three_pop/training.db")
            task_status = _tdb.get_task_status(task)
            current_best = task_status["best_accuracy"] if task_status else 0

            qualified = _tdb.get_best_teachers_for_task(task, min_accuracy=current_best)
            _tdb.close()

            if qualified:
                if child.get("teacher_model"):
                    if random.random() < 0.5:
                        child.pop("teacher_model", None)
                    else:
                        child["teacher_model"] = random.choice(qualified)[0]
                else:
                    child["teacher_model"] = random.choice(qualified)[0]
                provenance["teacher_model"] = {
                    "source": "ga_mutation", "severity": sev,
                    "value": child.get("teacher_model"),
                }
            else:
                child.pop("teacher_model", None)
        except Exception:
            pass

    def _load_seeds(self, path: str):
        """Load seed configs from YAML."""
        p = Path(path)
        if not p.exists():
            return
        with open(p) as f:
            if yaml is not None:
                data = yaml.safe_load(f)
            else:
                data = _parse_seed_configs_yaml(p)
        self._seed_configs = []
        for cfg in data.get("configs", []):
            # Remove 'name' key — it's metadata, not a config param
            c = {k: self._coerce_value(v) for k, v in cfg.items() if k != "name"}
            self._seed_configs.append(c)

    def get_seed_configs(self) -> list[dict]:
        """Return seed configs for initial population."""
        return [c.copy() for c in self._seed_configs]


# ── Fallback YAML parsers (no PyYAML dependency) ──────────────────────

def _parse_mutations_yaml(path: Path) -> dict:
    """Parse mutations.yaml without PyYAML. Handles the specific format we use."""
    text = path.read_text()
    mutations = []
    current = None

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if stripped.startswith("- name:"):
            if current:
                mutations.append(current)
            current = {"name": stripped.split(":", 1)[1].strip()}
        elif current is not None and ":" in stripped:
            key, _, val = stripped.partition(":")
            key = key.strip().lstrip("- ")
            val = val.strip()
            if val.startswith("[") and val.endswith("]"):
                items = val[1:-1].split(",")
                val = [_coerce(item.strip().strip("'\"")) for item in items if item.strip()]
            elif val.startswith("{") and val.endswith("}"):
                # Simple dict: {key: val, key: val}
                pairs = val[1:-1].split(",")
                val = {}
                for pair in pairs:
                    if ":" in pair:
                        dk, _, dv = pair.partition(":")
                        val[dk.strip().strip("'\"")] = _coerce(dv.strip().strip("'\""))
            elif val in ("true", "True"):
                val = True
            elif val in ("false", "False"):
                val = False
            else:
                val = _coerce(val)
            current[key] = val

    if current:
        mutations.append(current)

    # Handle nested implies — the fallback parser flattens them,
    # so we need to restructure implies that are dicts of dicts
    for m in mutations:
        if "implies" in m and isinstance(m["implies"], dict):
            # Check if values are themselves dicts (already parsed correctly)
            # or need restructuring
            pass

    return {"mutations": mutations}


def _parse_seed_configs_yaml(path: Path) -> dict:
    """Parse seed_configs.yaml without PyYAML."""
    text = path.read_text()
    configs = []
    current = None

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if stripped.startswith("- name:"):
            if current:
                configs.append(current)
            current = {"name": stripped.split(":", 1)[1].strip()}
        elif current is not None and ":" in stripped:
            key, _, val = stripped.partition(":")
            key = key.strip()
            val = val.strip()
            if val in ("true", "True"):
                val = True
            elif val in ("false", "False"):
                val = False
            else:
                val = _coerce(val)
            current[key] = val

    if current:
        configs.append(current)

    return {"configs": configs}


def _coerce(v):
    """Coerce string to int/float if possible."""
    if isinstance(v, (int, float, bool)):
        return v
    if isinstance(v, str):
        v = v.strip().strip("'\"")
        if not v:
            return v
        try:
            if "." in v or "e" in v.lower():
                return float(v)
            return int(v)
        except (ValueError, TypeError):
            return v
    return v
