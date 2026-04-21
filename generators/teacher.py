"""
Adaptive Teacher v2 — continuous difficulty scaling.

The key insight: start trivially easy, master 100%, then inch up.
By the time examples are large, memorization is impossible —
only the algorithm survives.

Each task has a continuous difficulty level [0.0, 1.0] that controls
the generator parameters (sequence length, number range, etc).
Difficulty only advances when FRESH accuracy hits mastery threshold.

Usage:
    teacher = AdaptiveTeacher()
    teacher.observe({"same_different": 0.98, "repeat_count": 0.40, ...})
    examples = teacher.generate(count=10000)
"""
import random
from dataclasses import dataclass, field


@dataclass
class TaskConfig:
    """Per-task configuration with continuous difficulty."""
    weight: float = 1.0
    difficulty: float = 0.0     # continuous [0.0, 1.0]
    accuracy: float = 0.0
    history: list = field(default_factory=list)
    stagnant_count: int = 0     # how many evals with no improvement


# Maps difficulty [0, 1] to generator kwargs via linear interpolation
# Each entry: { param: (min_val, max_val) }
DIFFICULTY_RANGES = {
    "sequence_completion": {
        "max_alpha":   (3, 20),
        "max_period":  (2, 6),
        "min_repeats": (3, 2),
        "max_repeats": (6, 4),
    },
    "same_different": {
        "max_val": (3, 200),
    },
    "odd_one_out": {
        "max_val": (3, 30),
        "min_len": (3, 6),
        "max_len": (4, 12),
    },
    "repeat_count": {
        "max_alpha": (2, 10),
        "min_len":   (3, 6),
        "max_len":   (4, 14),
    },
    "pattern_period": {
        "max_alpha":   (3, 12),
        "max_period":  (2, 6),
        "min_repeats": (3, 2),
        "max_repeats": (5, 3),
    },
    "arithmetic_next": {
        "max_start": (5, 50),
        "max_step":  (2, 20),
        "min_len":   (3, 3),
        "max_len":   (4, 6),
    },
    "geometric_next": {
        "max_base": (2, 8),
        "min_len":  (3, 4),
        "max_len":  (3, 6),
    },
    "mirror_detection": {
        "max_val": (3, 20),
        "min_len": (3, 5),
        "max_len": (3, 9),
    },
}

MASTERY_THRESHOLD = 0.90   # must hit this on FRESH to advance
ADVANCE_RATE = 0.05        # how much difficulty increases per mastery
RETREAT_RATE = 0.02        # how much to back off if struggling
STRUGGLING_THRESHOLD = 0.40

# Tasks unlock in this order. Each must be mastered before the next is introduced.
# Within each task, difficulty scales from 0→1 continuously.
TASK_ORDER = [
    "same_different",         # simplest: compare two values
    "mirror_detection",       # compare sequence to its reverse
    "odd_one_out",            # find the outlier
    "sequence_completion",    # predict next in repeating pattern
    "pattern_period",         # identify the period
    "repeat_count",           # count occurrences
    "arithmetic_next",        # arithmetic reasoning
    "geometric_next",         # multiplicative reasoning
]


class AdaptiveTeacher:
    def __init__(self, sequential_unlock=True):
        """
        Args:
            sequential_unlock: if True, tasks unlock one at a time.
                             if False, all tasks available from start.
        """
        self.sequential_unlock = sequential_unlock
        self.task_configs = {}
        self.unlocked_tasks = set()

        if sequential_unlock:
            # Start with only the first task
            first = TASK_ORDER[0]
            self.task_configs[first] = TaskConfig()
            self.unlocked_tasks.add(first)
            print(f"  🔓 Starting with: {first}", flush=True)
        else:
            for task_type in DIFFICULTY_RANGES:
                self.task_configs[task_type] = TaskConfig()
                self.unlocked_tasks = set(DIFFICULTY_RANGES.keys())

    def _interpolate_params(self, task_type, difficulty):
        """Convert difficulty [0,1] to generator kwargs."""
        ranges = DIFFICULTY_RANGES.get(task_type, {})
        params = {}
        for param, (lo, hi) in ranges.items():
            val = lo + (hi - lo) * difficulty
            params[param] = int(round(val))
        return params

    def _try_unlock_next(self):
        """Unlock the next task if all current tasks are mastered at difficulty >= 0.3."""
        if not self.sequential_unlock:
            return

        # Check if all unlocked tasks are sufficiently mastered
        for task_type in self.unlocked_tasks:
            cfg = self.task_configs.get(task_type)
            if cfg is None or cfg.accuracy < MASTERY_THRESHOLD or cfg.difficulty < 0.3:
                return  # not ready

        # Find next task to unlock
        for task_type in TASK_ORDER:
            if task_type not in self.unlocked_tasks:
                self.unlocked_tasks.add(task_type)
                self.task_configs[task_type] = TaskConfig(weight=1.5)  # extra practice for new task
                print(f"  🔓 Unlocked: {task_type} "
                      f"(all previous tasks mastered at diff≥0.3)", flush=True)
                return

    def observe(self, per_type_accs: dict):
        """Update configs based on FRESH per-type accuracies."""
        for task_type, acc in per_type_accs.items():
            if task_type not in self.task_configs:
                continue  # ignore tasks we haven't unlocked
            if task_type not in self.unlocked_tasks:
                continue

            cfg = self.task_configs[task_type]
            cfg.accuracy = acc
            cfg.history.append(acc)

            # Advance difficulty if mastered
            if acc >= MASTERY_THRESHOLD:
                old_diff = cfg.difficulty
                cfg.difficulty = min(cfg.difficulty + ADVANCE_RATE, 1.0)
                cfg.weight = max(cfg.weight * 0.7 + 0.7 * 0.3, 0.5)  # reduce weight
                cfg.stagnant_count = 0
                if cfg.difficulty > old_diff:
                    print(f"  ↑ {task_type}: difficulty {old_diff:.2f}→{cfg.difficulty:.2f} "
                          f"(mastered at {acc:.0%})", flush=True)

            # Back off if struggling
            elif acc < STRUGGLING_THRESHOLD:
                cfg.stagnant_count += 1
                cfg.weight = cfg.weight * 0.7 + 1.5 * 0.3  # more practice
                if cfg.stagnant_count >= 3 and cfg.difficulty > 0:
                    old_diff = cfg.difficulty
                    cfg.difficulty = max(cfg.difficulty - RETREAT_RATE, 0.0)
                    cfg.stagnant_count = 0
                    print(f"  ↓ {task_type}: difficulty {old_diff:.2f}→{cfg.difficulty:.2f} "
                          f"(struggling at {acc:.0%})", flush=True)

            # Normal learning range
            else:
                cfg.weight = cfg.weight * 0.7 + 1.0 * 0.3
                cfg.stagnant_count = 0

        # Check if we should unlock the next task
        self._try_unlock_next()

    def get_status(self) -> str:
        lines = []
        for task_type, cfg in sorted(self.task_configs.items()):
            d = cfg.difficulty
            w = cfg.weight
            a = cfg.accuracy
            status = "✓" if a >= MASTERY_THRESHOLD else ("…" if a >= STRUGGLING_THRESHOLD else "✗")
            params = self._interpolate_params(task_type, d)
            param_str = " ".join(f"{k}={v}" for k, v in sorted(params.items()))
            lines.append(f"    {status} {task_type}: acc={a:.0%}  diff={d:.2f}  "
                        f"w={w:.1f}  [{param_str}]")
        return "\n".join(lines)

    def generate(self, count: int) -> list:
        """Generate examples with continuous difficulty scaling."""
        import generators.level0_patterns as gen0

        TASK_TO_FN = {
            "sequence_completion": gen0.gen_sequence_completion,
            "same_different": gen0.gen_same_different,
            "odd_one_out": gen0.gen_odd_one_out,
            "repeat_count": gen0.gen_repeat_count,
            "pattern_period": gen0.gen_pattern_period,
            "arithmetic_next": gen0.gen_arithmetic_next,
            "geometric_next": gen0.gen_geometric_next,
            "mirror_detection": gen0.gen_mirror_detection,
        }

        tasks = []
        weights = []
        for task_type, cfg in self.task_configs.items():
            if task_type in TASK_TO_FN and task_type in self.unlocked_tasks:
                tasks.append(task_type)
                weights.append(cfg.weight)

        examples = []
        for _ in range(count):
            task_type = random.choices(tasks, weights=weights, k=1)[0]
            fn = TASK_TO_FN[task_type]
            cfg = self.task_configs[task_type]
            params = self._interpolate_params(task_type, cfg.difficulty)

            try:
                ex = fn(**params)
            except TypeError:
                ex = fn()

            examples.append(ex)

        return examples
