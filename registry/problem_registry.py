"""Problem Registry — discover tasks from YAML manifests instead of hardcoded lists."""

import importlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

try:
    import yaml
except ImportError:
    yaml = None


@dataclass
class ProblemSpec:
    """Declarative specification of a training problem."""
    name: str
    type: str                              # "generator" | "dataset"
    generator_ref: str | None = None       # "module:function"
    dataset_path: str | None = None        # path to JSONL file
    target_accuracy: float = 0.95
    tags: list[str] = field(default_factory=list)
    description: str = ""


class ProblemRegistry:
    """Discovers and serves training problems from YAML manifests.

    Usage:
        registry = ProblemRegistry()
        registry.discover(["problems/"])
        gen_fn = registry.get_generator("parity")
        examples = [gen_fn() for _ in range(1000)]
    """

    def __init__(self):
        self.problems: dict[str, ProblemSpec] = {}
        self._generators: dict[str, Callable] = {}

    def discover(self, dirs: list[str]) -> dict[str, ProblemSpec]:
        """Walk directories for problem.yaml files, register each problem."""
        for d in dirs:
            base = Path(d)
            if not base.exists():
                continue
            for yaml_path in sorted(base.rglob("problem.yaml")):
                self._load_manifest(yaml_path)
        return self.problems

    def _load_manifest(self, path: Path):
        """Parse one problem.yaml and register it."""
        with open(path) as f:
            if yaml is not None:
                data = yaml.safe_load(f)
            else:
                data = self._parse_simple_yaml(f.read())

        name = data["name"]
        ptype = data.get("type", "generator")

        spec = ProblemSpec(
            name=name,
            type=ptype,
            generator_ref=data.get("generator"),
            dataset_path=data.get("dataset"),
            target_accuracy=data.get("target_accuracy", 0.95),
            tags=data.get("tags", []),
            description=data.get("description", ""),
        )

        if ptype == "generator" and not spec.generator_ref:
            raise ValueError(f"Problem '{name}' is type=generator but has no generator field")
        if ptype == "dataset" and not spec.dataset_path:
            raise ValueError(f"Problem '{name}' is type=dataset but has no dataset field")

        self.problems[name] = spec

    def get_generator(self, name: str) -> Callable:
        """Dynamic-import and return the generator function for a problem."""
        if name in self._generators:
            return self._generators[name]

        spec = self.problems.get(name)
        if not spec:
            raise KeyError(f"Problem '{name}' not found in registry. "
                           f"Available: {list(self.problems.keys())}")

        if spec.type == "dataset":
            # Wrap dataset as a generator that yields random examples
            return self._dataset_as_generator(spec)

        # Parse "module:function" reference
        module_path, func_name = spec.generator_ref.rsplit(":", 1)
        module = importlib.import_module(module_path)
        gen_fn = getattr(module, func_name)

        self._generators[name] = gen_fn
        return gen_fn

    def get_dataset(self, name: str) -> Iterator[dict]:
        """Yield examples from a JSONL dataset file."""
        spec = self.problems.get(name)
        if not spec:
            raise KeyError(f"Problem '{name}' not found in registry")
        if spec.type != "dataset" or not spec.dataset_path:
            raise ValueError(f"Problem '{name}' is not a dataset type")

        with open(spec.dataset_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def _dataset_as_generator(self, spec: ProblemSpec) -> Callable:
        """Wrap a JSONL dataset as a callable generator (random sampling)."""
        import random

        # Load all examples into memory (datasets are small for these tasks)
        examples = list(self.get_dataset(spec.name))
        if not examples:
            raise ValueError(f"Dataset '{spec.dataset_path}' is empty")

        def gen_from_dataset():
            return random.choice(examples)

        self._generators[spec.name] = gen_from_dataset
        return gen_from_dataset

    def list_problems(self) -> list[str]:
        """Return sorted list of all discovered problem names."""
        return sorted(self.problems.keys())

    def get_target_accuracy(self, name: str) -> float:
        """Return the target accuracy for a problem."""
        spec = self.problems.get(name)
        return spec.target_accuracy if spec else 0.95

    @staticmethod
    def _parse_simple_yaml(text: str) -> dict:
        """Minimal YAML parser for flat key-value problem manifests.
        Handles: strings, numbers, lists (bracket syntax [a, b, c]).
        Does NOT handle nested dicts, multi-line, or anchors."""
        result = {}
        for line in text.strip().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                continue
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip()
            # Remove quotes
            if (val.startswith('"') and val.endswith('"')) or \
               (val.startswith("'") and val.endswith("'")):
                val = val[1:-1]
            # List: [a, b, c]
            elif val.startswith("[") and val.endswith("]"):
                items = val[1:-1].split(",")
                val = [item.strip().strip("'\"") for item in items if item.strip()]
            # Number
            elif val.replace(".", "", 1).replace("-", "", 1).isdigit():
                val = float(val) if "." in val else int(val)
            result[key] = val
        return result
