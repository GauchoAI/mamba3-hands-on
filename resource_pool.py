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
    EvolutionState, score_experiments, select_parent, mutate_config,
    SEED_CONFIGS, get_system_resources,
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
                 runs_dir="runs_three_pop"):
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
        self.monitor = ResourceMonitor()

        # GA state
        self.evo_state = EvolutionState()

        # Worker tracking: subprocess.Popen per worker
        self.processes = {}       # exp_id → Popen
        self.worker_task = {}     # exp_id → task name
        self.worker_config = {}   # exp_id → config dict
        self.worker_parent = {}   # exp_id → parent_id
        self.teacher_tasks = {}   # task → exp_id (graduated)
        self.tasks_remaining = set(tasks)
        self.next_id = 0
        self.pending_configs = [] # (task, config, parent_id) waiting for admission

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

            # Admit pending workers if resources allow
            if self.pending_configs and self.monitor.can_admit():
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
             "--d-model", str(cfg.get("d_model", 64)),
             "--d-state", str(cfg.get("d_state", 16)),
             "--headdim", str(cfg.get("headdim", 16)),
             "--layers", str(cfg.get("n_kernel_layers", 1)),
             "--lr", str(cfg.get("lr", 1e-3)),
             "--weight-decay", str(cfg.get("weight_decay", 0.0)),
             "--optimizer", str(cfg.get("optimizer", "adamw")),
             "--loss-fn", str(cfg.get("loss_fn", "ce")),
             "--batch-size", str(cfg.get("batch_size", 256)),
             "--steps-per-cycle", str(cfg.get("steps_per_cycle", 200)),
             "--max-cycles", "500",
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

            # Still running — read latest output to check progress
            self._read_worker_progress(exp_id)

    def _read_worker_progress(self, exp_id):
        """Read stdout.log tail to get latest accuracy for a running worker."""
        log_path = self.runs_dir / exp_id / "stdout.log"
        if not log_path.exists():
            return

        try:
            # Read last 2KB of log
            with open(log_path, 'rb') as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - 2048))
                tail = f.read().decode('utf-8', errors='ignore')

            # Parse last cycle line: "[task] cycle N  loss=X  acc=Y%  best=Z%"
            task = self.worker_task[exp_id]
            best_acc = 0.0
            last_acc = 0.0
            last_cycle = 0
            last_loss = 0.0

            for line in tail.split('\n'):
                if f'[{task}] cycle' in line:
                    parts = line.strip().split()
                    for p in parts:
                        if p.startswith('acc='):
                            last_acc = float(p.replace('acc=', '').replace('%', '')) / 100
                        elif p.startswith('best='):
                            best_acc = float(p.replace('best=', '').replace('%', '')) / 100
                        elif p.startswith('loss='):
                            last_loss = float(p.replace('loss=', ''))
                    # Extract cycle number
                    try:
                        idx = parts.index('cycle')
                        last_cycle = int(parts[idx + 1])
                    except (ValueError, IndexError):
                        pass

                if '★' in line and 'MASTERED' in line:
                    best_acc = max(best_acc, self.target_acc)

            if last_cycle > 0:
                # Record for GA scoring
                self.evo_state.record(exp_id, last_cycle, best_acc)

                # Push to Firebase via callback
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
                            n_params=0,
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

        except Exception:
            pass

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

        # Per task: select parent, mutate, queue child
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
                child_cfg = mutate_config(parent["config"], plateau_severity=plateau_severity)
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
