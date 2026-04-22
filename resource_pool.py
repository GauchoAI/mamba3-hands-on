"""
Resource-aware worker pool with genetic algorithm.

Workers are SUBPROCESSES (not threads) — each gets its own GIL and CUDA context.
This is how we achieved 100% GPU saturation before: 35+ independent processes.

Admission gated by CPU/RAM/GPU/VRAM headroom. GA evolves configs per task.
Students start distilling as soon as the first teacher graduates.

Communication: filesystem (metrics.json, config.json, status files) — proven pattern.

Usage:
    from resource_pool import ResourceAwarePool
    pool = ResourceAwarePool(tasks, seed_configs, device, on_cycle, on_graduate)
    pool.start()  # blocks until done or SIGINT
"""
import os
os.environ["PYTHONUNBUFFERED"] = "1"
import sys
sys.path.insert(0, os.path.dirname(__file__))

import json
import time
import subprocess
import signal
import shutil
from pathlib import Path
from collections import defaultdict

from coordinator import (
    EvolutionState, score_experiments, select_parent,
    mutate_config, smart_mutate_config, MutationHistory,
    SEED_CONFIGS, BASE_CONFIG, get_system_resources,
)


# ── Resource monitoring ────────────────────────────────────────────

class ResourceMonitor:
    """System resource monitor. Caches readings for 2s."""

    def __init__(self):
        self._cache = None
        self._cache_time = 0.0

    def snapshot(self):
        now = time.time()
        if self._cache and (now - self._cache_time) < 2.0:
            return self._cache
        try:
            cpu, ram, gpu, vram = get_system_resources()
        except Exception:
            cpu, ram, gpu, vram = 0, 0, 0, 0
        self._cache = {"cpu_pct": cpu, "ram_pct": ram, "gpu_pct": gpu, "vram_pct": vram}
        self._cache_time = now
        return self._cache

    def can_admit(self):
        s = self.snapshot()
        return (s["cpu_pct"] < 85 and s["ram_pct"] < 80 and
                s["gpu_pct"] < 90 and s["vram_pct"] < 90)


# ── Resource-aware pool with GA ────────────────────────────────────

class ResourceAwarePool:
    """Parallel subprocess worker pool with genetic algorithm and resource gating."""

    def __init__(self, tasks, seed_configs, device, on_cycle=None,
                 on_graduate=None, target_acc=0.95, teachers_dir=None,
                 runs_dir="runs_three_pop", max_workers=1):
        self.tasks = list(tasks)
        self.seed_configs = seed_configs
        self.device = device
        self.on_cycle = on_cycle
        self.on_graduate_cb = on_graduate
        self.target_acc = target_acc
        self.teachers_dir = Path(teachers_dir) if teachers_dir else Path("checkpoints/specialists")
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

        # Resource management
        self.max_workers = max_workers
        self.monitor = ResourceMonitor()

        # GA state
        self.evo_state = EvolutionState()
        self.mutation_history = MutationHistory()

        # Worker tracking: subprocess.Popen per worker
        self.processes = {}       # exp_id → Popen
        self.worker_task = {}     # exp_id → task name
        self.worker_config = {}   # exp_id → config dict
        self.worker_parent = {}   # exp_id → parent_id
        self.teacher_tasks = {}   # task → exp_id (graduated)
        self.tasks_remaining = set(tasks)
        self.next_id = 0
        self.pending_configs = [] # (task, config, parent_id) waiting for admission

        # Plateau detection for GA probing
        self.worker_best_at = {}  # exp_id → (best_acc, cycle_when_best)
        self.plateau_threshold = 20  # cycles without improvement before probing
        self.probe_cycles = 5        # cycles per probe (fast: ~5s each)
        self.n_probes = 3            # number of probes to try
        self.probed_tasks = set()    # tasks already probed this round (avoid re-probing)

        # Student
        self.student_proc = None
        self.student_exp_id = None

        # Shutdown
        self._should_stop = False

    def _log(self, msg):
        print(msg, flush=True)

    def _next_exp_id(self):
        eid = f"exp_{self.next_id:04d}"
        self.next_id += 1
        return eid

    def start(self):
        """Run the pool. Blocks until all tasks graduate or stopped."""
        self._log(f"\n{'='*60}")
        self._log(f"RESOURCE POOL — {len(self.tasks)} tasks, GA enabled (subprocess mode)")
        self._log(f"  Device: {self.device}")
        self._log(f"  Seed configs: {len(self.seed_configs)}")
        self._log(f"  Target accuracy: {self.target_acc:.0%}")
        self._log(f"  Runs dir: {self.runs_dir}")
        self._log(f"{'='*60}\n")

        # Seed: one worker per task, round-robin from seed configs
        for i, task in enumerate(self.tasks):
            cfg = self.seed_configs[i % len(self.seed_configs)].copy()
            cfg["task"] = task
            self.pending_configs.append((task, cfg, None))

        # Main loop
        generation = 0
        last_evolve = time.time()
        last_check = time.time()
        evolve_interval = 30
        check_interval = 5

        while self.tasks_remaining and not self._should_stop:
            now = time.time()

            # Admit pending workers if under max and resources allow
            n_active = len(self.processes)
            if self.pending_configs and n_active < self.max_workers and self.monitor.can_admit():
                self._admit_one()

            # Check worker results periodically
            if now - last_check >= check_interval:
                self._check_all_workers()
                self._check_student()
                last_check = now

            # GA evolution periodically
            if now - last_evolve >= evolve_interval:
                self._evolve_generation(generation)
                generation += 1
                last_evolve = now

                # Log status
                s = self.monitor.snapshot()
                n_alive = sum(1 for p in self.processes.values() if p.poll() is None)
                self._log(f"\n[Gen {generation}] Workers={n_alive} "
                         f"Teachers={len(self.teacher_tasks)} "
                         f"Students={'1' if self._student_alive() else '0'} | "
                         f"CPU={s['cpu_pct']:.0f}% RAM={s['ram_pct']:.0f}% "
                         f"GPU={s['gpu_pct']:.0f}% VRAM={s['vram_pct']:.0f}%")

            time.sleep(1)

        # All tasks done
        if not self.tasks_remaining:
            self._log(f"\n{'='*60}")
            self._log(f"ALL {len(self.tasks)} TASKS MASTERED!")
            self._log(f"{'='*60}")

        self._log("Pool shutdown complete.")

    def _admit_one(self):
        """Start one pending worker as a subprocess."""
        task, cfg, parent_id = self.pending_configs.pop(0)
        if task in self.teacher_tasks:
            return

        exp_id = self._next_exp_id()
        run_dir = self.runs_dir / exp_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Write config
        cfg["task"] = task
        config_path = run_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)

        # Weight inheritance from parent
        if parent_id:
            parent_dir = self.runs_dir / parent_id
            parent_ckpt = parent_dir / "checkpoint.pt"
            if parent_ckpt.exists():
                try:
                    parent_cfg = json.load(open(parent_dir / "config.json"))
                    arch_keys = ["d_model", "d_state", "headdim", "n_kernel_layers"]
                    compatible = all(cfg.get(k) == parent_cfg.get(k) for k in arch_keys)
                    if compatible:
                        shutil.copy2(parent_ckpt, run_dir / "checkpoint.pt")
                except Exception:
                    pass
            self.evo_state.register_lineage(exp_id, parent_id)

        # Launch subprocess using specialist_trainer.py
        proc = subprocess.Popen(
            [sys.executable, "-u", "specialist_trainer.py",
             "--task", task,
             "--run-dir", str(run_dir),
             "--d-model", str(cfg.get("d_model", 64)),
             "--d-state", str(cfg.get("d_state", 16)),
             "--headdim", str(cfg.get("headdim", 16)),
             "--layers", str(cfg.get("n_kernel_layers", 3)),
             "--lr", str(cfg.get("lr", 1e-3)),
             "--weight-decay", str(cfg.get("weight_decay", 0.0)),
             "--optimizer", str(cfg.get("optimizer", "adamw")),
             "--loss-fn", str(cfg.get("loss_fn", "ce")),
             "--batch-size", str(cfg.get("batch_size", 256)),
             "--steps-per-cycle", str(cfg.get("steps_per_cycle", 200)),
             "--max-cycles", "10000",
             "--target-acc", str(self.target_acc)],
            stdout=open(run_dir / "stdout.log", "w"),
            stderr=subprocess.STDOUT,
            cwd=str(Path(__file__).parent),
        )

        self.processes[exp_id] = proc
        self.worker_task[exp_id] = task
        self.worker_config[exp_id] = cfg
        self.worker_parent[exp_id] = parent_id
        (run_dir / "status").write_text("running")

        self.probed_tasks.discard(task)  # allow probing again with new config
        self._log(f"  + {exp_id} [{task}]: d={cfg.get('d_model')} "
                 f"L={cfg.get('n_kernel_layers')} lr={cfg.get('lr', 1e-3):.0e} "
                 f"opt={cfg.get('optimizer', 'adamw')} loss={cfg.get('loss_fn', 'ce')}")

    def _check_all_workers(self):
        """Check all running workers for completion/graduation."""
        for exp_id in list(self.processes.keys()):
            proc = self.processes[exp_id]
            task = self.worker_task[exp_id]

            if task in self.teacher_tasks:
                # Task already graduated by another worker — kill this one
                if proc.poll() is None:
                    proc.terminate()
                del self.processes[exp_id]
                continue

            if proc.poll() is not None:
                # Process finished — check if mastered
                self._handle_finished(exp_id)
                continue

            # Still running — read metrics.json
            self._read_worker_metrics(exp_id)

    def _read_worker_metrics(self, exp_id):
        """Read metrics.json (written atomically by specialist_trainer)."""
        metrics_path = self.runs_dir / exp_id / "metrics.json"
        if not metrics_path.exists():
            return

        try:
            with open(metrics_path) as f:
                m = json.load(f)

            task = self.worker_task[exp_id]
            last_cycle = m.get("cycle", 0)
            last_acc = m.get("acc", 0)
            best_acc = m.get("best_acc", 0)
            last_loss = m.get("loss", 0)
            n_params = m.get("n_params", 0)

            if last_cycle > 0:
                self.evo_state.record(exp_id, last_cycle, best_acc)

                # Track plateau: when did best_acc last improve?
                prev_best, prev_cycle = self.worker_best_at.get(exp_id, (0, 0))
                if best_acc > prev_best:
                    self.worker_best_at[exp_id] = (best_acc, last_cycle)
                elif prev_cycle > 0 and (last_cycle - prev_cycle) >= self.plateau_threshold:
                    # Plateau detected — trigger GA probes (once per task per stint)
                    if best_acc < self.target_acc and task not in self.probed_tasks:
                        self.probed_tasks.add(task)
                        self._trigger_probes(exp_id, task, best_acc, last_cycle)

                if self.on_cycle:
                    s = self.monitor.snapshot()
                    n_alive = sum(1 for p in self.processes.values() if p.poll() is None)
                    cfg = self.worker_config.get(exp_id, {})
                    try:
                        self.on_cycle(
                            task=task, cycle=last_cycle, acc=last_acc,
                            best_acc=best_acc, loss=last_loss,
                            exp_id=exp_id, config=cfg,
                            parent_id=self.worker_parent.get(exp_id),
                            n_params=n_params,
                            cpu_pct=s["cpu_pct"], ram_pct=s["ram_pct"],
                            gpu_pct=s["gpu_pct"], vram_pct=s["vram_pct"],
                            n_workers=n_alive,
                            n_teachers=len(self.teacher_tasks),
                            n_students=1 if self._student_alive() else 0,
                            teacher_tasks=dict(self.teacher_tasks),
                            tasks_remaining=list(self.tasks_remaining),
                            lineage=self.evo_state.lineage,
                            generation=self.evo_state.generation,
                        )
                    except Exception as e:
                        self._log(f"  on_cycle error: {e}")

        except (json.JSONDecodeError, IOError):
            pass

    def _trigger_probes(self, stuck_exp_id, task, current_best, current_cycle):
        """Plateau detected. Kill worker, run quick probes, pick best config, continue."""
        self._log(f"\n  🔍 PLATEAU: {task} stuck at {current_best:.0%} for "
                 f"{self.plateau_threshold} cycles. Probing {self.n_probes} mutations...")

        # Kill the stuck worker
        proc = self.processes.get(stuck_exp_id)
        if proc and proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=10)
        if stuck_exp_id in self.processes:
            del self.processes[stuck_exp_id]

        # Get the stuck worker's config and checkpoint
        parent_config = self.worker_config.get(stuck_exp_id, {})
        parent_ckpt = self.runs_dir / stuck_exp_id / "checkpoint.pt"

        # Calculate plateau severity for mutation aggressiveness
        plateau_severity = min(3.0, (current_cycle - self.worker_best_at.get(stuck_exp_id, (0, 0))[1]) / 50)

        # Run N probes sequentially (each gets full GPU)
        probe_results = []
        for i in range(self.n_probes):
            probe_cfg = smart_mutate_config(
                parent_config,
                mutation_history=self.mutation_history,
                plateau_severity=plateau_severity,
            )
            probe_cfg["task"] = task

            probe_id = self._next_exp_id()
            probe_dir = self.runs_dir / probe_id
            probe_dir.mkdir(parents=True, exist_ok=True)

            # Write config
            config_path = probe_dir / "config.json"
            with open(config_path, "w") as f:
                json.dump(probe_cfg, f, indent=2)

            # Copy parent checkpoint for weight inheritance
            if parent_ckpt.exists():
                shutil.copy2(parent_ckpt, probe_dir / "checkpoint.pt")

            # Run probe (short: probe_cycles)
            changed = {k: probe_cfg[k] for k in probe_cfg
                      if k not in ("task", "steps_per_cycle") and probe_cfg.get(k) != parent_config.get(k)}
            self._log(f"    Probe {i+1}/{self.n_probes} [{probe_id}]: {changed}")

            probe_proc = subprocess.Popen(
                [sys.executable, "-u", "specialist_trainer.py",
                 "--task", task,
                 "--run-dir", str(probe_dir),
                 "--d-model", str(probe_cfg.get("d_model", 64)),
                 "--d-state", str(probe_cfg.get("d_state", 16)),
                 "--headdim", str(probe_cfg.get("headdim", 16)),
                 "--layers", str(probe_cfg.get("n_kernel_layers", 3)),
                 "--lr", str(probe_cfg.get("lr", 1e-3)),
                 "--weight-decay", str(probe_cfg.get("weight_decay", 0.0)),
                 "--optimizer", str(probe_cfg.get("optimizer", "adamw")),
                 "--loss-fn", str(probe_cfg.get("loss_fn", "ce")),
                 "--batch-size", str(probe_cfg.get("batch_size", 256)),
                 "--steps-per-cycle", "100",
                 "--max-cycles", str(self.probe_cycles),
                 "--target-acc", str(self.target_acc)],
                stdout=open(probe_dir / "stdout.log", "w"),
                stderr=subprocess.STDOUT,
                cwd=str(Path(__file__).parent),
            )

            # Wait for probe to finish
            probe_proc.wait(timeout=300)

            # Read result
            metrics_path = probe_dir / "metrics.json"
            probe_best = 0.0
            if metrics_path.exists():
                try:
                    with open(metrics_path) as f:
                        m = json.load(f)
                    probe_best = m.get("best_acc", 0)
                except Exception:
                    pass

            probe_results.append((probe_id, probe_cfg, probe_best))
            self._log(f"    Probe {i+1} result: {probe_best:.0%}")

            # Record for mutation history
            self.mutation_history.record(parent_config, probe_cfg, current_best, probe_best)

        # Pick the best probe
        probe_results.sort(key=lambda x: -x[2])
        best_probe_id, best_cfg, best_probe_acc = probe_results[0]

        if best_probe_acc > current_best:
            # Winner found — continue with this config
            self._log(f"  ✓ Probe {best_probe_id} won: {best_probe_acc:.0%} > {current_best:.0%}")
            self._log(f"    Config: { {k: best_cfg[k] for k in best_cfg if k not in ('task', 'steps_per_cycle') and best_cfg.get(k) != parent_config.get(k)} }")
            # Queue the winning config to continue training
            self.pending_configs.insert(0, (task, best_cfg, best_probe_id))
        else:
            # No improvement — continue with original config but reset plateau counter
            self._log(f"  ✗ No probe beat {current_best:.0%}. Continuing with original config.")
            self.pending_configs.insert(0, (task, parent_config, stuck_exp_id))
            # Reset plateau tracking so we don't probe again immediately
            self.worker_best_at.pop(stuck_exp_id, None)

    def _handle_finished(self, exp_id):
        """A worker process finished. Check if it mastered the task."""
        task = self.worker_task[exp_id]
        run_dir = self.runs_dir / exp_id

        # Check if specialist checkpoint exists (saved on mastery)
        ckpt_path = Path("checkpoints/specialists") / f"{task}.pt"
        cache_path = Path("checkpoints/specialists") / f"{task}_cache.pt"

        graduated = False
        if ckpt_path.exists():
            try:
                import torch
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                acc = ckpt.get("accuracy", 0)
                if acc >= self.target_acc:
                    graduated = True
            except Exception:
                pass

        if graduated and task not in self.teacher_tasks:
            self.teacher_tasks[task] = exp_id
            self.tasks_remaining.discard(task)

            # Copy to teachers dir
            if ckpt_path.exists():
                shutil.copy2(ckpt_path, self.teachers_dir / f"{task}.pt")
            if cache_path.exists():
                shutil.copy2(cache_path, self.teachers_dir / f"{task}_cache.pt")

            # Kill other workers for this task
            for eid, proc in list(self.processes.items()):
                if self.worker_task.get(eid) == task and eid != exp_id:
                    if proc.poll() is None:
                        proc.terminate()

            # Notify
            if self.on_graduate_cb:
                try:
                    self.on_graduate_cb(task, exp_id, self.worker_config.get(exp_id, {}))
                except Exception:
                    pass

            self._log(f"\n  🎓 GRADUATED: {task} → Teacher (exp={exp_id})"
                     f" | {len(self.teacher_tasks)}/{len(self.tasks)} teachers"
                     f" | {len(self.tasks_remaining)} remaining\n")

            # Cross-task config transfer: winning config seeds remaining tasks
            winning_config = self.worker_config.get(exp_id, {})
            updated = 0
            for i, (t, cfg, pid) in enumerate(self.pending_configs):
                if t in self.tasks_remaining:
                    new_cfg = winning_config.copy()
                    new_cfg["task"] = t
                    self.pending_configs[i] = (t, new_cfg, exp_id)
                    updated += 1
            if updated:
                self._log(f"  📋 Cross-task transfer: {task}'s config seeded {updated} pending tasks")

            # Start student on first graduation
            if len(self.teacher_tasks) == 1 and not self._student_alive():
                self._start_student()
            elif self._student_alive():
                self._log(f"  📚 Student will pick up new teacher {task} on next cycle")

        # Clean up
        if exp_id in self.processes:
            del self.processes[exp_id]

    def _evolve_generation(self, generation):
        """GA: score workers, select parents, mutate, queue children."""
        # Build results from running workers
        results = []
        for exp_id, proc in self.processes.items():
            if proc.poll() is not None:
                continue
            task = self.worker_task[exp_id]
            if task in self.teacher_tasks:
                continue
            h = self.evo_state.history.get(exp_id, [])
            best = h[-1][1] if h else 0.0
            cycle = h[-1][0] if h else 0
            results.append({
                "exp_id": exp_id,
                "task": task,
                "best_fresh": best,
                "cycle": cycle,
                "config": self.worker_config.get(exp_id, {}),
                "type_accs": {task: best},
                "status": "running",
            })

        if not results:
            return

        scored = score_experiments(results, self.evo_state)

        # Record mutation outcomes for history-informed future mutations
        for r in scored:
            parent_id = self.worker_parent.get(r["exp_id"])
            if parent_id:
                parent_h = self.evo_state.history.get(parent_id, [])
                parent_best = parent_h[-1][1] if parent_h else 0.0
                if parent_best > 0 and r["best_fresh"] > 0:
                    parent_cfg = self.worker_config.get(parent_id, {})
                    child_cfg_r = self.worker_config.get(r["exp_id"], {})
                    self.mutation_history.record(parent_cfg, child_cfg_r,
                                                parent_best, r["best_fresh"])

        # Per task: select parent, smart mutate, queue child
        for task in list(self.tasks_remaining):
            task_workers = [r for r in scored if r["task"] == task]
            if not task_workers:
                continue

            # Plateau detection
            plateau_severity = 0.0
            for w in task_workers:
                m = self.evo_state.get_momentum(w["exp_id"])
                if m <= 0 and w["cycle"] > 20:
                    plateau_severity = min(3.0, (w["cycle"] - 20) / 30)

            parent, reason = select_parent(task_workers, self.evo_state, generation)
            if parent:
                child_cfg = smart_mutate_config(
                    parent["config"],
                    mutation_history=self.mutation_history,
                    plateau_severity=plateau_severity,
                )
                child_cfg["task"] = task
                self.pending_configs.append((task, child_cfg, parent["exp_id"]))

        self.evo_state.generation = generation

    def _start_student(self):
        """Start a student distillation subprocess."""
        self._log(f"  📚 Starting student distillation with {len(self.teacher_tasks)} teachers")

        exp_id = "student_000"
        run_dir = self.runs_dir / exp_id
        run_dir.mkdir(parents=True, exist_ok=True)

        proc = subprocess.Popen(
            [sys.executable, "-u", "distill.py",
             "--runs-dir", str(self.runs_dir),
             "--student-d-model", "64",
             "--student-d-state", "16",
             "--student-headdim", "16",
             "--student-layers", "3",
             "--lr", "1e-3",
             "--cycles", "500",
             "--steps-per-cycle", "50",
             "--pcgrad"],
            stdout=open(run_dir / "stdout.log", "w"),
            stderr=subprocess.STDOUT,
            cwd=str(Path(__file__).parent),
        )

        self.student_proc = proc
        self.student_exp_id = exp_id

    def _student_alive(self):
        return self.student_proc is not None and self.student_proc.poll() is None

    def _check_student(self):
        """Check student progress and push to Firebase."""
        if not self._student_alive():
            return
        # Student logs its own progress — we could parse it here
        # For now, distill.py pushes directly to Firebase

    def shutdown(self):
        """Graceful shutdown — terminate all workers."""
        self._should_stop = True
        for exp_id, proc in self.processes.items():
            if proc.poll() is None:
                proc.terminate()
        if self._student_alive():
            self.student_proc.terminate()
