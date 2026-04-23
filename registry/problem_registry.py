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
class CurriculumStage:
    """One difficulty level in a curriculum."""
    stage: int
    params: dict                           # generator kwargs for this stage
    advance_at: float = 0.90               # accuracy threshold to advance


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
    params: dict = field(default_factory=dict)          # default generator kwargs
    curriculum: list[CurriculumStage] = field(default_factory=list)  # difficulty stages

    def get_stage(self, stage_num: int) -> CurriculumStage | None:
        """Get a curriculum stage by number (1-indexed)."""
        for s in self.curriculum:
            if s.stage == stage_num:
                return s
        return None

    def get_params_for_stage(self, stage_num: int) -> dict:
        """Get generator params for a given stage. Falls back to defaults."""
        stage = self.get_stage(stage_num)
        if stage:
            merged = dict(self.params)
            merged.update(stage.params)
            return merged
        return dict(self.params)

    @property
    def max_stage(self) -> int:
        """Highest stage number defined."""
        if not self.curriculum:
            return 0
        return max(s.stage for s in self.curriculum)


class ProblemRegistry:
    """Discovers and serves training problems from YAML manifests.

    Usage:
        registry = ProblemRegistry()
        registry.discover(["problems/"])
        gen_fn = registry.get_generator("parity", stage=2)
        examples = [gen_fn() for _ in range(1000)]
    """

    def __init__(self):
        self.problems: dict[str, ProblemSpec] = {}
        self._raw_generators: dict[str, Callable] = {}  # unbound generator fns

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

        # Parse curriculum stages
        curriculum = []
        for i, stage_data in enumerate(data.get("curriculum", []), 1):
            if isinstance(stage_data, dict):
                params = {k: v for k, v in stage_data.items()
                          if k not in ("stage", "advance_at")}
                curriculum.append(CurriculumStage(
                    stage=stage_data.get("stage", i),
                    params=params,
                    advance_at=float(stage_data.get("advance_at", 0.90)),
                ))

        # Parse default params
        params = data.get("params", {})
        if isinstance(params, dict):
            # Flatten {key: {default: val}} to {key: val}
            flat_params = {}
            for k, v in params.items():
                if isinstance(v, dict) and "default" in v:
                    flat_params[k] = v["default"]
                else:
                    flat_params[k] = v
            params = flat_params

        spec = ProblemSpec(
            name=name,
            type=ptype,
            generator_ref=data.get("generator"),
            dataset_path=data.get("dataset"),
            target_accuracy=data.get("target_accuracy", 0.95),
            tags=data.get("tags", []),
            description=data.get("description", ""),
            params=params,
            curriculum=curriculum,
        )

        if ptype == "generator" and not spec.generator_ref:
            raise ValueError(f"Problem '{name}' is type=generator but has no generator field")
        if ptype == "dataset" and not spec.dataset_path:
            raise ValueError(f"Problem '{name}' is type=dataset but has no dataset field")

        self.problems[name] = spec

    def get_generator(self, name: str, stage: int = 0) -> Callable:
        """Return a generator function, optionally bound to curriculum stage params.

        stage=0 means use defaults (no curriculum).
        stage=N means use the params from curriculum stage N.
        """
        spec = self.problems.get(name)
        if not spec:
            raise KeyError(f"Problem '{name}' not found in registry. "
                           f"Available: {list(self.problems.keys())}")

        if spec.type == "dataset":
            return self._dataset_as_generator(spec)

        # Get or import the raw generator function
        if name not in self._raw_generators:
            module_path, func_name = spec.generator_ref.rsplit(":", 1)
            module = importlib.import_module(module_path)
            self._raw_generators[name] = getattr(module, func_name)

        raw_fn = self._raw_generators[name]

        # If no stage or no curriculum, return with default params
        if stage == 0 or not spec.curriculum:
            if spec.params:
                def gen_with_defaults():
                    return raw_fn(**spec.params)
                return gen_with_defaults
            return raw_fn

        # Bind curriculum stage params
        stage_params = spec.get_params_for_stage(stage)
        if stage_params:
            def gen_with_stage():
                return raw_fn(**stage_params)
            return gen_with_stage

        return raw_fn

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

        examples = list(self.get_dataset(spec.name))
        if not examples:
            raise ValueError(f"Dataset '{spec.dataset_path}' is empty")

        def gen_from_dataset():
            return random.choice(examples)

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
        """Minimal YAML parser for problem manifests.
        Handles: strings, numbers, inline lists [a, b], and
        list-of-inline-dicts (curriculum: \\n  - {k: v, ...})."""
        result = {}
        current_list_key = None
        current_list = []

        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            # List item: "  - {stage: 1, min_len: 2, ...}"
            if stripped.startswith("- {") and stripped.endswith("}") and current_list_key:
                inner = stripped[2:]  # strip "- "
                d = ProblemRegistry._parse_inline_dict(inner)
                current_list.append(d)
                continue

            # If we were collecting a list and hit a non-list line, flush
            if current_list_key and not stripped.startswith("-"):
                result[current_list_key] = current_list
                current_list_key = None
                current_list = []

            if ":" not in stripped:
                continue

            key, _, val = stripped.partition(":")
            key = key.strip()
            val = val.strip()

            # Empty value after colon = start of a list block
            if not val:
                current_list_key = key
                current_list = []
                continue

            # Remove quotes
            if (val.startswith('"') and val.endswith('"')) or \
               (val.startswith("'") and val.endswith("'")):
                val = val[1:-1]
            # Inline list: [a, b, c]
            elif val.startswith("[") and val.endswith("]"):
                items = val[1:-1].split(",")
                val = [item.strip().strip("'\"") for item in items if item.strip()]
            # Number
            elif val.replace(".", "", 1).replace("-", "", 1).isdigit():
                val = float(val) if "." in val else int(val)

            result[key] = val

        # Flush any trailing list
        if current_list_key:
            result[current_list_key] = current_list

        return result

    @staticmethod
    def _parse_inline_dict(s: str) -> dict:
        """Parse '{stage: 1, min_len: 2, max_len: 4, advance_at: 0.90}'."""
        s = s.strip()
        if s.startswith("{"):
            s = s[1:]
        if s.endswith("}"):
            s = s[:-1]
        d = {}
        for pair in s.split(","):
            if ":" not in pair:
                continue
            k, _, v = pair.partition(":")
            k = k.strip()
            v = v.strip().strip("'\"")
            # Coerce numbers
            try:
                if "." in v:
                    v = float(v)
                else:
                    v = int(v)
            except (ValueError, TypeError):
                pass
            d[k] = v
        return d
