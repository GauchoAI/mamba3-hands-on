---
title: GA tournament
chapter: "01"
status: archival
summary: "GA-era narrative paired with the current live task appendix: {{lab.tasksTracked}} tasks tracked, {{lab.mastered}} mastered, {{lab.workers}} workers, and {{lab.teachers}} teachers."
---

# Chapter 01 — GA tournament

**Status:** archival.

**Synopsis.** A 50-fresh-experiment GA evolving in parallel under an
H100-backed orchestrator, mastering 14 of 15 tasks in roughly 24 hours
of wall-clock. The era that produced `specialist_trainer.py` (the
daily-driver trainer to this day), `three_populations.py`, the
strategies / mutation / external-teacher tooling, and the multi-task
mamba3-augmented + plain comparison runs.

**Key takeaways** (full arc in
[`docs/findings/ga_tournament.md`](../docs/findings/ga_tournament.md)):

1. GPU saturation is the binding constraint long before the population
   is "done" — saturate the card with diverse specialists first, only
   then layer mutation pressure.
2. Teacher mutation beats hand-tuned teacher schedules.
3. Some tasks (`bool_expr_depth3` was the last) are genuinely hard for
   a 130 M SSM and benefit from **architecture mutations**, not just
   hyperparameter ones.

**What's here.** All the GA-era trainers (`specialist_trainer.py`,
`three_populations.py`), survey scripts (`exp_augmented_*`,
`exp_multitask`), strategy / amplification / teacher tooling
(`strategies.py`, `amplify.py`, `external_teacher.py`,
`auto_tuner.py`, `mac_sweep.py`), validation suites (`*_validate.py`),
the progressive-curriculum trainer + model
(`train_progressive.py`, `progressive_model.py`), distillation
(`distill.py`, `distill_from_router.py`, `train_router.py`), and the
miscellaneous proof-of-concept tasks (`crack_mod3*.py`,
`selective_copy.py`, `sort_suite.py`, `formal_language.py`,
`dual_task.py`, `grokking.py`, `length_gen*.py`,
`finetune_hanoi_to_gcd.py`). Plus the historical `three_pop/` data
directory and the `harness_3stage_mamba3/` experimental harness.
