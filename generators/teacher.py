"""
Adaptive Teacher — observes per-type performance and adjusts curriculum.

The teacher controls:
  1. Task weights (which tasks to practice more)
  2. Difficulty parameters (how hard each task should be)
  3. Progression (when to increase difficulty for mastered tasks)

Usage:
    teacher = AdaptiveTeacher()
    teacher.observe({"same_different": 0.98, "repeat_count": 0.40, ...})
    examples = teacher.generate(count=10000)
"""
import random
from dataclasses import dataclass, field


@dataclass
class TaskConfig:
    """Per-task difficulty and weight configuration."""
    weight: float = 1.0          # sampling weight (higher = more practice)
    # Difficulty knobs — each generator uses what it needs
    max_alpha: int = 10          # alphabet size
    max_val: int = 20            # max number value
    min_len: int = 3             # min sequence length
    max_len: int = 8             # max sequence length
    max_period: int = 4          # max pattern period
    max_step: int = 8            # max arithmetic step
    # Tracking
    accuracy: float = 0.0        # last observed accuracy
    history: list = field(default_factory=list)  # accuracy over time
    difficulty_level: int = 0    # 0=easy, 1=medium, 2=hard


# Difficulty presets per task type
DIFFICULTY_PRESETS = {
    "sequence_completion": [
        {"max_alpha": 5,  "max_period": 3, "min_repeats": 3, "max_repeats": 5},   # easy
        {"max_alpha": 10, "max_period": 4, "min_repeats": 2, "max_repeats": 5},   # medium
        {"max_alpha": 15, "max_period": 6, "min_repeats": 2, "max_repeats": 4},   # hard
    ],
    "same_different": [
        {"max_val": 10},
        {"max_val": 50},
        {"max_val": 200},
    ],
    "odd_one_out": [
        {"max_val": 8,  "min_len": 4, "max_len": 6},
        {"max_val": 15, "min_len": 5, "max_len": 8},
        {"max_val": 30, "min_len": 6, "max_len": 12},
    ],
    "repeat_count": [
        {"max_alpha": 4, "min_len": 4, "max_len": 7},      # few symbols, short
        {"max_alpha": 6, "min_len": 5, "max_len": 10},
        {"max_alpha": 10, "min_len": 6, "max_len": 14},
    ],
    "pattern_period": [
        {"max_alpha": 5, "max_period": 3, "min_repeats": 3, "max_repeats": 5},
        {"max_alpha": 8, "max_period": 4, "min_repeats": 2, "max_repeats": 4},
        {"max_alpha": 12, "max_period": 6, "min_repeats": 2, "max_repeats": 3},
    ],
    "arithmetic_next": [
        {"max_start": 10, "max_step": 5,  "min_len": 4, "max_len": 6},
        {"max_start": 20, "max_step": 10, "min_len": 3, "max_len": 6},
        {"max_start": 50, "max_step": 20, "min_len": 3, "max_len": 5},
    ],
    "geometric_next": [
        {"max_base": 3, "min_len": 3, "max_len": 4},
        {"max_base": 5, "min_len": 3, "max_len": 5},
        {"max_base": 8, "min_len": 4, "max_len": 6},
    ],
    "mirror_detection": [
        {"max_val": 5,  "min_len": 3, "max_len": 5},
        {"max_val": 10, "min_len": 4, "max_len": 7},
        {"max_val": 20, "min_len": 5, "max_len": 9},
    ],
}

# Thresholds for progression
MASTERED_THRESHOLD = 0.85    # above this → reduce weight, increase difficulty
LEARNING_THRESHOLD = 0.60    # between this and mastered → normal
STRUGGLING_THRESHOLD = 0.60  # below this → increase weight, decrease difficulty


class AdaptiveTeacher:
    def __init__(self):
        self.task_configs = {}
        for task_type in DIFFICULTY_PRESETS:
            self.task_configs[task_type] = TaskConfig()

    def observe(self, per_type_accs: dict):
        """Update task configs based on observed per-type accuracies."""
        for task_type, acc in per_type_accs.items():
            if task_type not in self.task_configs:
                self.task_configs[task_type] = TaskConfig()

            cfg = self.task_configs[task_type]
            cfg.accuracy = acc
            cfg.history.append(acc)

            # Adjust weights
            if acc >= MASTERED_THRESHOLD:
                cfg.weight = 0.5   # less drilling
                # Progress difficulty if consistently mastered
                if len(cfg.history) >= 3 and all(a >= MASTERED_THRESHOLD for a in cfg.history[-3:]):
                    max_diff = len(DIFFICULTY_PRESETS.get(task_type, [])) - 1
                    if cfg.difficulty_level < max_diff:
                        cfg.difficulty_level += 1
                        cfg.history.clear()  # reset history for new difficulty
                        cfg.weight = 1.5  # more practice at new difficulty
                        print(f"  📈 {task_type}: difficulty {cfg.difficulty_level-1}→{cfg.difficulty_level}",
                              flush=True)
            elif acc >= LEARNING_THRESHOLD:
                cfg.weight = 1.0   # normal
            else:
                cfg.weight = 2.0   # more practice
                # Regress difficulty if struggling
                if len(cfg.history) >= 3 and all(a < STRUGGLING_THRESHOLD for a in cfg.history[-3:]):
                    if cfg.difficulty_level > 0:
                        cfg.difficulty_level -= 1
                        cfg.history.clear()
                        print(f"  📉 {task_type}: difficulty {cfg.difficulty_level+1}→{cfg.difficulty_level}",
                              flush=True)

    def get_status(self) -> str:
        """Return a summary string of the teacher's view."""
        lines = []
        for task_type, cfg in sorted(self.task_configs.items()):
            diff = cfg.difficulty_level
            w = cfg.weight
            a = cfg.accuracy
            status = "✓" if a >= MASTERED_THRESHOLD else ("…" if a >= LEARNING_THRESHOLD else "✗")
            lines.append(f"    {status} {task_type}: acc={a:.0%}  diff={diff}  weight={w:.1f}")
        return "\n".join(lines)

    def generate(self, count: int) -> list:
        """Generate examples with adaptive weights and difficulty."""
        import generators.level0_patterns as gen0

        # Map task types to generator functions
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

        # Build weighted task list
        tasks = []
        weights = []
        for task_type, cfg in self.task_configs.items():
            if task_type in TASK_TO_FN:
                tasks.append(task_type)
                weights.append(cfg.weight)

        examples = []
        for _ in range(count):
            task_type = random.choices(tasks, weights=weights, k=1)[0]
            fn = TASK_TO_FN[task_type]
            cfg = self.task_configs[task_type]

            # Get difficulty params for this task
            presets = DIFFICULTY_PRESETS.get(task_type, [{}])
            diff_idx = min(cfg.difficulty_level, len(presets) - 1)
            params = presets[diff_idx]

            try:
                ex = fn(**params)
            except TypeError:
                # Some generators don't accept all params — call with no args
                ex = fn()

            examples.append(ex)

        return examples
