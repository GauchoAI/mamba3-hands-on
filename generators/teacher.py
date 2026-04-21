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
    # Samples-to-mastery tracking
    unlock_step: int = 0        # global step when this task was unlocked
    mastery_step: int = 0       # global step when first hit MASTERY_THRESHOLD
    examples_seen: int = 0      # total examples of this type seen
    mastered: bool = False       # has it ever hit mastery?


# Maps difficulty [0, 1] to generator kwargs via linear interpolation
# Each entry: { param: (min_val_at_0, max_val_at_1) }
DIFFICULTY_RANGES = {
    # Stage 0: binary foundations
    "parity": {
        "min_len": (3, 4),
        "max_len": (4, 12),
    },
    "binary_pattern_next": {
        "min_repeats": (3, 2),
        "max_repeats": (5, 4),
    },
    # Stage 1: comparison
    "same_different": {
        "max_val": (1, 200),     # start at 1! binary first, then scale up
    },
    "odd_one_out": {
        "max_val": (1, 30),      # start binary, scale up
        "min_len": (3, 6),
        "max_len": (4, 12),
    },
    # Stage 2: pattern detection
    "sequence_completion": {
        "max_alpha":   (3, 20),
        "max_period":  (2, 6),
        "min_repeats": (3, 2),
        "max_repeats": (6, 4),
    },
    "pattern_period": {
        "max_alpha":   (3, 12),
        "max_period":  (2, 6),
        "min_repeats": (3, 2),
        "max_repeats": (5, 3),
    },
    "run_length_next": {
        "max_run": (2, 5),
        "max_val": (2, 5),
    },
    # Stage 3: sequence memory
    "mirror_detection": {
        "max_val": (1, 20),      # start binary, scale up
        "min_len": (3, 5),
        "max_len": (3, 9),
    },
    "repeat_count": {
        "max_alpha": (2, 10),    # 2 = binary alphabet
        "min_len":   (3, 6),
        "max_len":   (4, 14),
    },
    # Stage 4: arithmetic reasoning
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
    "alternating_next": {
        "max_val": (5, 15),
    },
    # Stage 5: logic
    "logic_gate": {
        "max_inputs": (2, 2),   # stays simple — gates are gates
    },
    "logic_chain": {
        "max_depth": (1, 4),    # 1 gate → 4 gates chained
    },
    "modus_ponens": {},         # no difficulty scaling — it's conceptual
}

MASTERY_THRESHOLD = 0.90   # must hit this on FRESH to advance
ADVANCE_RATE = 0.05        # how much difficulty increases per mastery
RETREAT_RATE = 0.02        # how much to back off if struggling
STRUGGLING_THRESHOLD = 0.40

# Tasks unlock in this order. Each builds on capabilities from the previous.
# The curriculum is designed so each task teaches a skill the next one needs.
#
# Stage 0 — Binary foundations (teaches: detect patterns in binary streams)
#   The simplest possible patterns. Like learning to see.
#   SSM state learns to track binary state / parity.
#   parity:              0 1 1 0 1 → DIFF (odd) or SAME (even)
#   binary_pattern_next: 0 0 1 1 0 0 1 1 → 0 (detect binary cycle)
#
# Stage 1 — Comparison (teaches: attend to values, compare)
#   SSM learns to hold values in state and compare them.
#   same_different:     compare two values → SAME/DIFF
#   odd_one_out:        compare N values, find the outlier
#
# Stage 2 — Pattern detection (teaches: detect repetition, periodicity)
#   SSM learns to recognize repeating structure in sequences.
#   Registers could store the detected pattern for reuse.
#   sequence_completion: A B A B A ? → B (detect the cycle, predict next)
#   pattern_period:      1 2 3 1 2 3 → period=3 (count the cycle length)
#   run_length_next:     0 1 1 2 2 2 ? → 3 (detect the run-length pattern)
#
# Stage 3 — Sequence memory (teaches: remember and compare across positions)
#   Requires holding the full sequence in structured memory.
#   Registers/spikes should fire to store key positions.
#   mirror_detection:   1 2 3 2 1 → MIRROR (compare first half to reversed second)
#   repeat_count:       A B A B A → count(A)=3 (accumulate count in register)
#
# Stage 4 — Arithmetic reasoning (teaches: compute, not just compare)
#   Requires the step function to perform actual computation.
#   arithmetic_next:    2 5 8 11 ? → 14 (detect step, apply it)
#   geometric_next:     2 6 18 ? → 54 (detect ratio, apply it)
#   alternating_next:   1 10 2 9 3 8 ? → 4 (two interleaved sequences)
#
# Stage 5 — Logic (teaches: boolean reasoning, circuit evaluation)
#   Modus ponens, truth tables, gate evaluation.
#   This is where it starts to truly reason.
#   logic_gate:         AND 1 0 → 0, OR 1 0 → 1, XOR 1 1 → 0
#   logic_chain:        AND 1 1 OR 0 → OR(AND(1,1), 0) = 1
#   modus_ponens:       IF 1 THEN 1 → 1 (p→q, p, therefore q)
#
TASK_ORDER = [
    # Stage 0: binary foundations
    "parity",
    "binary_pattern_next",
    # Stage 1: comparison
    "same_different",
    "odd_one_out",
    # Stage 2: pattern detection
    "sequence_completion",
    "pattern_period",
    "run_length_next",
    # Stage 3: sequence memory
    "mirror_detection",
    "repeat_count",
    # Stage 4: arithmetic reasoning
    "arithmetic_next",
    "geometric_next",
    "alternating_next",
    # Stage 5: logic
    "logic_gate",
    "logic_chain",
    "modus_ponens",
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
        self.global_step = 0          # track training steps
        self.mastery_log = []         # list of (task, steps_to_master, examples_to_master)
        self.boss_mode = False        # final boss activated?
        self.boss_results = {}        # unseen task results

        if sequential_unlock:
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

    def set_step(self, step):
        """Update global step counter (call from training loop)."""
        self.global_step = step

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
                self.task_configs[task_type] = TaskConfig(
                    weight=1.5, unlock_step=self.global_step
                )
                print(f"  🔓 Unlocked: {task_type} at step {self.global_step} "
                      f"(all previous tasks mastered at diff≥0.3)", flush=True)
                return

        # All tasks unlocked and mastered → final boss!
        if not self.boss_mode:
            self.boss_mode = True
            print(f"\n  🏆 ALL TASKS MASTERED — FINAL BOSS MODE at step {self.global_step}",
                  flush=True)

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

                # Record first mastery
                if not cfg.mastered:
                    cfg.mastered = True
                    cfg.mastery_step = self.global_step
                    steps_to_master = self.global_step - cfg.unlock_step
                    self.mastery_log.append({
                        "task": task_type,
                        "unlock_step": cfg.unlock_step,
                        "mastery_step": self.global_step,
                        "steps_to_master": steps_to_master,
                        "examples_to_master": cfg.examples_seen,
                        "task_index": len(self.mastery_log),
                    })
                    print(f"  ★ {task_type}: MASTERED in {steps_to_master} steps "
                          f"({cfg.examples_seen:,} examples)", flush=True)

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
        for task_type in TASK_ORDER:
            if task_type not in self.task_configs:
                lines.append(f"    🔒 {task_type}: locked")
                continue
            cfg = self.task_configs[task_type]
            d = cfg.difficulty
            w = cfg.weight
            a = cfg.accuracy
            status = "✓" if a >= MASTERY_THRESHOLD else ("…" if a >= STRUGGLING_THRESHOLD else "✗")
            params = self._interpolate_params(task_type, d)
            param_str = " ".join(f"{k}={v}" for k, v in sorted(params.items()))
            mastery_info = ""
            if cfg.mastered:
                steps = cfg.mastery_step - cfg.unlock_step
                mastery_info = f"  mastered_in={steps}steps"
            lines.append(f"    {status} {task_type}: acc={a:.0%}  diff={d:.2f}  "
                        f"w={w:.1f}{mastery_info}  [{param_str}]")
        if self.boss_mode:
            lines.append(f"    🏆 BOSS MODE ACTIVE")
        return "\n".join(lines)

    def get_learning_report(self) -> str:
        """Report on learning-to-learn: do later tasks take fewer examples?"""
        if not self.mastery_log:
            return "  No tasks mastered yet."
        lines = ["  --- Learning-to-Learn Report ---"]
        for entry in self.mastery_log:
            lines.append(f"    Task {entry['task_index']+1}: {entry['task']}"
                        f"  steps={entry['steps_to_master']}"
                        f"  examples={entry['examples_to_master']:,}")
        if len(self.mastery_log) >= 2:
            first = self.mastery_log[0]["steps_to_master"]
            last = self.mastery_log[-1]["steps_to_master"]
            if first > 0:
                ratio = last / first
                trend = "accelerating!" if ratio < 0.7 else (
                    "improving" if ratio < 1.0 else "no acceleration yet")
                lines.append(f"    Ratio (last/first): {ratio:.2f} — {trend}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize teacher state for checkpointing."""
        configs = {}
        for task_type, cfg in self.task_configs.items():
            configs[task_type] = {
                "weight": cfg.weight,
                "difficulty": cfg.difficulty,
                "accuracy": cfg.accuracy,
                "history": cfg.history[-10:],  # keep last 10 only
                "stagnant_count": cfg.stagnant_count,
                "unlock_step": cfg.unlock_step,
                "mastery_step": cfg.mastery_step,
                "examples_seen": cfg.examples_seen,
                "mastered": cfg.mastered,
            }
        return {
            "task_configs": configs,
            "unlocked_tasks": list(self.unlocked_tasks),
            "global_step": self.global_step,
            "mastery_log": self.mastery_log,
            "boss_mode": self.boss_mode,
            "sequential_unlock": self.sequential_unlock,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AdaptiveTeacher":
        """Restore teacher from checkpoint."""
        teacher = cls.__new__(cls)
        teacher.sequential_unlock = d.get("sequential_unlock", True)
        teacher.global_step = d.get("global_step", 0)
        teacher.mastery_log = d.get("mastery_log", [])
        teacher.boss_mode = d.get("boss_mode", False)
        teacher.boss_results = {}
        teacher.unlocked_tasks = set(d.get("unlocked_tasks", []))
        teacher.task_configs = {}
        for task_type, cfg_d in d.get("task_configs", {}).items():
            cfg = TaskConfig(
                weight=cfg_d["weight"],
                difficulty=cfg_d["difficulty"],
                accuracy=cfg_d["accuracy"],
                history=cfg_d.get("history", []),
                stagnant_count=cfg_d.get("stagnant_count", 0),
                unlock_step=cfg_d.get("unlock_step", 0),
                mastery_step=cfg_d.get("mastery_step", 0),
                examples_seen=cfg_d.get("examples_seen", 0),
                mastered=cfg_d.get("mastered", False),
            )
            teacher.task_configs[task_type] = cfg
        print(f"  Teacher restored: {len(teacher.unlocked_tasks)} tasks unlocked, "
              f"step={teacher.global_step}, boss={'ON' if teacher.boss_mode else 'OFF'}",
              flush=True)
        return teacher

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
            "parity": gen0.gen_parity,
            "binary_pattern_next": gen0.gen_binary_pattern_next,
            "run_length_next": gen0.gen_run_length_next,
            "alternating_next": gen0.gen_alternating_next,
            "logic_gate": gen0.gen_logic_gate,
            "logic_chain": gen0.gen_logic_chain,
            "modus_ponens": gen0.gen_modus_ponens,
        }

        # In boss mode, mix in unseen tasks
        if self.boss_mode:
            try:
                from generators.boss_tasks import BOSS_GENERATORS
                for name, fn in BOSS_GENERATORS.items():
                    TASK_TO_FN[name] = fn
                    if name not in self.task_configs:
                        self.task_configs[name] = TaskConfig(weight=2.0)
                        self.unlocked_tasks.add(name)
            except ImportError:
                pass  # boss tasks not yet defined

        tasks = []
        weights = []
        for task_type, cfg in self.task_configs.items():
            if task_type in TASK_TO_FN and task_type in self.unlocked_tasks:
                tasks.append(task_type)
                weights.append(cfg.weight)

        # Count examples per type
        type_counts = {}
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
            type_counts[task_type] = type_counts.get(task_type, 0) + 1

        # Update examples_seen
        for task_type, n in type_counts.items():
            if task_type in self.task_configs:
                self.task_configs[task_type].examples_seen += n

        return examples
