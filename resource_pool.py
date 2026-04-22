"""
Resource-aware worker pool with genetic algorithm.

Runs multiple specialist workers in parallel (threads, shared CUDA context).
Admission gated by CPU/RAM/GPU/VRAM headroom. GA evolves configs per task.
Students start distilling as soon as the first teacher graduates.

Usage:
    from resource_pool import ResourceAwarePool
    pool = ResourceAwarePool(tasks, seed_configs, device, on_cycle, on_graduate)
    pool.start()  # blocks until done or SIGINT
"""
import os
os.environ["PYTHONUNBUFFERED"] = "1"
import sys
sys.path.insert(0, os.path.dirname(__file__))

import time
import random
import threading
import traceback
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from progressive_model import ProgressiveModel, ByteTokenizer, VOCAB_SIZE, PAD
from specialist_trainer import (
    load_generators, GENERATORS, create_model_and_optimizer,
    run_single_cycle, precompute_teacher_cache, get_loss_fn,
)
from coordinator import (
    EvolutionState, score_experiments, select_parent, mutate_config,
    SEED_CONFIGS, get_gpu_usage, get_system_resources,
)


# ── Resource monitoring ────────────────────────────────────────────

@dataclass
class ResourceSnapshot:
    cpu_pct: float = 0.0
    ram_pct: float = 0.0
    gpu_pct: float = 0.0
    vram_pct: float = 0.0
    timestamp: float = 0.0


class ResourceMonitor:
    """System resource monitor. Caches readings for 1s."""

    def __init__(self):
        self._cache = None
        self._cache_time = 0.0

    def snapshot(self) -> ResourceSnapshot:
        now = time.time()
        if self._cache and (now - self._cache_time) < 1.0:
            return self._cache
        try:
            cpu, ram, gpu, vram = get_system_resources()
        except Exception:
            cpu, ram, gpu, vram = 0, 0, 0, 0
        self._cache = ResourceSnapshot(cpu, ram, gpu, vram, now)
        self._cache_time = now
        return self._cache


class AdmissionController:
    """Gate new workers based on resource headroom."""

    HEADROOM = {
        'cpu_pct': 85.0,
        'ram_pct': 80.0,
        'gpu_pct': 90.0,
        'vram_pct': 90.0,
    }

    def __init__(self, monitor: ResourceMonitor):
        self.monitor = monitor

    def can_admit(self) -> bool:
        s = self.monitor.snapshot()
        return (s.cpu_pct < self.HEADROOM['cpu_pct'] and
                s.ram_pct < self.HEADROOM['ram_pct'] and
                s.gpu_pct < self.HEADROOM['gpu_pct'] and
                s.vram_pct < self.HEADROOM['vram_pct'])


# ── Worker thread ──────────────────────────────────────────────────

class WorkerThread(threading.Thread):
    """Trains one specialist (one task, one config) until graduation or stopped."""

    def __init__(self, exp_id, task, config, device, pool):
        super().__init__(daemon=True, name=f"worker-{exp_id}")
        self.exp_id = exp_id
        self.task = task
        self.config = config
        self.device = device
        self.pool = pool
        self.parent_id = config.get("_parent_id")

        self.model = None
        self.opt = None
        self.cycle = 0
        self.best_acc = 0.0
        self.last_acc = 0.0
        self.last_loss = 0.0
        self.graduated = False
        self.error = None
        self._stop_event = threading.Event()

    def run(self):
        try:
            gen_fn = GENERATORS.get(self.task)
            if not gen_fn:
                self.error = f"Unknown task: {self.task}"
                return

            tok = ByteTokenizer()
            cfg = {k: v for k, v in self.config.items() if not k.startswith('_')}
            self.model, self.opt, perp, scheduler, loss_fn, n_params = \
                create_model_and_optimizer(cfg, self.device)
            noise = cfg.get("noise_scale", 0.0)

            self.pool._log(f"[{self.exp_id}] {self.task} started: "
                          f"d={cfg.get('d_model')} L={cfg.get('n_kernel_layers')} "
                          f"lr={cfg.get('lr', 1e-3):.0e} opt={cfg.get('optimizer', 'adamw')} "
                          f"loss={cfg.get('loss_fn', 'ce')} ({n_params:,} params)")

            while not self._stop_event.is_set():
                self.cycle += 1
                acc, loss, elapsed = run_single_cycle(
                    self.model, self.opt, gen_fn, tok, cfg, self.device,
                    loss_fn, perp=perp, scheduler=scheduler, noise=noise,
                )
                self.last_acc = acc
                self.last_loss = loss
                self.best_acc = max(self.best_acc, acc)

                # Callback (thread-safe)
                with self.pool.callback_lock:
                    self.pool._on_worker_cycle(self, acc, self.best_acc, loss, n_params)

                # Check graduation
                if acc >= self.pool.target_acc:
                    self.graduated = True
                    self.pool._log(f"  ★ [{self.exp_id}] {self.task} MASTERED "
                                  f"at {acc:.0%} in {self.cycle} cycles!")
                    # Save checkpoint + teacher cache
                    self._save_and_graduate(gen_fn, tok, cfg, n_params)
                    break

        except Exception as e:
            self.error = e
            self.pool._log(f"  ERROR [{self.exp_id}] {self.task}: {e}\n"
                          f"{traceback.format_exc()}")

    def _save_and_graduate(self, gen_fn, tok, cfg, n_params):
        ckpt_dir = Path("checkpoints/specialists")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"{self.task}.pt"
        torch.save({
            "model": self.model.state_dict(),
            "task": self.task,
            "config": cfg,
            "accuracy": self.best_acc,
            "cycles": self.cycle,
            "n_params": n_params,
        }, ckpt_path)

        teacher_data = precompute_teacher_cache(self.model, gen_fn, tok, self.device)
        cache_path = ckpt_dir / f"{self.task}_cache.pt"
        torch.save(teacher_data, cache_path)
        self.pool._log(f"  Saved {self.task} teacher ({len(teacher_data)} cached examples)")

    def stop(self):
        self._stop_event.set()


# ── Student thread ─────────────────────────────────────────────────

class StudentThread(threading.Thread):
    """Distills from available teachers. Hot-adds new teachers as they graduate."""

    def __init__(self, config, device, pool):
        super().__init__(daemon=True, name="student")
        self.config = config
        self.device = device
        self.pool = pool
        self.teacher_models = {}  # task → model (frozen)
        self.teacher_lock = threading.Lock()
        self.cycle = 0
        self.best_fresh = 0.0
        self.last_fresh = 0.0
        self.last_loss = 0.0
        self.type_accs = {}
        self._stop_event = threading.Event()

    def add_teacher(self, task, ckpt_path, config):
        """Hot-add a new teacher. Thread-safe."""
        try:
            model = ProgressiveModel(
                d_model=config.get("d_model", 64),
                d_state=config.get("d_state", 16),
                expand=2,
                headdim=config.get("headdim", 16),
            ).to(self.device)
            for _ in range(config.get("n_kernel_layers", 3)):
                model.add_kernel_layer()
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            model.eval()
            with self.teacher_lock:
                self.teacher_models[task] = model
            self.pool._log(f"  📚 Student: added teacher for {task} "
                          f"({len(self.teacher_models)} total)")
        except Exception as e:
            self.pool._log(f"  ERROR adding teacher {task}: {e}")

    def run(self):
        try:
            from distill import distillation_loss, pcgrad_project
            tok = ByteTokenizer()

            cfg = {k: v for k, v in self.config.items() if not k.startswith('_')}
            student = ProgressiveModel(
                d_model=cfg.get("d_model", 64),
                d_state=cfg.get("d_state", 16),
                expand=2,
                headdim=cfg.get("headdim", 16),
            ).to(self.device)
            for _ in range(cfg.get("n_kernel_layers", 3)):
                student.add_kernel_layer()
            student.set_mode("kernel")

            n_params = sum(p.numel() for p in student.parameters())
            opt = torch.optim.AdamW(student.parameters(), lr=cfg.get("lr", 1e-3),
                                     weight_decay=cfg.get("weight_decay", 0.0))

            self.pool._log(f"  📚 Student started: d={cfg.get('d_model')} "
                          f"L={cfg.get('n_kernel_layers')} ({n_params:,} params)")

            while not self._stop_event.is_set():
                self.cycle += 1

                # Snapshot current teachers
                with self.teacher_lock:
                    teachers = dict(self.teacher_models)
                if not teachers:
                    time.sleep(1)
                    continue

                t0 = time.time()
                student.train()

                # Generate data for each teacher's task
                by_task = {}
                for task in teachers:
                    gen_fn = GENERATORS.get(task)
                    if gen_fn:
                        examples = []
                        for _ in range(500):
                            ex = gen_fn()
                            tokens, sep = tok.encode_curriculum(ex)
                            examples.append((tokens, sep, list(ex["output"].encode("utf-8"))))
                        by_task[task] = examples

                # PCGrad: per-task gradients
                task_grads_list = []
                total_loss = 0.0
                batch_size = min(32, cfg.get("batch_size", 64))

                for task, teacher in teachers.items():
                    examples = by_task.get(task, [])
                    if not examples:
                        continue

                    # Sample batch
                    batch_exs = random.sample(examples, min(batch_size, len(examples)))
                    max_len = max(len(t) for t, _, _ in batch_exs)
                    token_tensor = torch.full((len(batch_exs), max_len), PAD,
                                            dtype=torch.long, device=self.device)
                    seps = []
                    for i, (tokens, sep, _) in enumerate(batch_exs):
                        token_tensor[i, :len(tokens)] = torch.tensor(tokens)
                        seps.append(sep)

                    with torch.no_grad():
                        teacher_logits = teacher(token_tensor)
                    student_logits = student(token_tensor)

                    B, L, V = student_logits.shape
                    loss = torch.tensor(0.0, device=self.device)
                    count = 0
                    for b in range(B):
                        sep = seps[b]
                        for t in range(sep, L - 1):
                            target = token_tensor[b, t + 1]
                            if target == PAD:
                                break
                            loss += distillation_loss(
                                student_logits[b, t].unsqueeze(0),
                                teacher_logits[b, t].unsqueeze(0),
                                target.unsqueeze(0),
                            )
                            count += 1
                    if count > 0:
                        loss = loss / count
                        opt.zero_grad()
                        loss.backward()
                        task_grad = {name: p.grad.clone()
                                    for name, p in student.named_parameters()
                                    if p.grad is not None}
                        task_grads_list.append(task_grad)
                        total_loss += loss.item()

                # PCGrad projection
                if task_grads_list:
                    combined = pcgrad_project(task_grads_list)
                    opt.zero_grad()
                    for name, p in student.named_parameters():
                        if name in combined:
                            p.grad = combined[name]
                    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                    opt.step()

                cycle_loss = total_loss / max(len(task_grads_list), 1)

                # Evaluate
                student.eval()
                type_accs = {}
                with torch.no_grad():
                    for task in teachers:
                        gen_fn = GENERATORS.get(task)
                        if not gen_fn:
                            continue
                        correct = total = 0
                        for _ in range(50):
                            ex = gen_fn()
                            tokens, sep = tok.encode_curriculum(ex)
                            out_bytes = list(ex["output"].encode("utf-8"))
                            t = torch.tensor([tokens], dtype=torch.long, device=self.device)
                            logits = student(t)
                            ok = True
                            for j, expected in enumerate(out_bytes):
                                p = sep + j
                                if p < logits.shape[1]:
                                    if logits[0, p].argmax().item() != expected:
                                        ok = False
                                        break
                                else:
                                    ok = False
                            if ok:
                                correct += 1
                            total += 1
                        type_accs[task] = correct / max(total, 1)

                fresh = sum(type_accs.values()) / max(len(type_accs), 1)
                self.best_fresh = max(self.best_fresh, fresh)
                self.last_fresh = fresh
                self.last_loss = cycle_loss
                self.type_accs = type_accs
                elapsed = time.time() - t0

                with self.pool.callback_lock:
                    self.pool._on_student_cycle(self, fresh, cycle_loss, type_accs, n_params)

                if self.cycle % 5 == 0:
                    self.pool._log(f"  📚 Student cycle {self.cycle}: "
                                  f"fresh={fresh:.0%} best={self.best_fresh:.0%} "
                                  f"loss={cycle_loss:.3f} teachers={len(teachers)} "
                                  f"{elapsed:.1f}s")

        except Exception as e:
            self.pool._log(f"  ERROR [student]: {e}\n{traceback.format_exc()}")

    def stop(self):
        self._stop_event.set()


# ── Resource-aware pool with GA ────────────────────────────────────

class ResourceAwarePool:
    """Parallel worker pool with genetic algorithm and resource gating."""

    def __init__(self, tasks, seed_configs, device, on_cycle=None,
                 on_graduate=None, target_acc=0.95, teachers_dir=None):
        self.tasks = list(tasks)
        self.seed_configs = seed_configs
        self.device = device
        self.on_cycle = on_cycle
        self.on_graduate_cb = on_graduate
        self.target_acc = target_acc
        self.teachers_dir = Path(teachers_dir) if teachers_dir else Path("checkpoints/specialists")

        # Resource management
        self.monitor = ResourceMonitor()
        self.admission = AdmissionController(self.monitor)

        # GA state
        self.evo_state = EvolutionState()
        self.evo_lock = threading.Lock()

        # Worker tracking
        self.workers = {}           # exp_id → WorkerThread
        self.worker_lock = threading.Lock()
        self.callback_lock = threading.Lock()
        self.teacher_tasks = {}     # task → exp_id (graduated)
        self.tasks_remaining = set(tasks)
        self.next_id = 0
        self.pending_configs = []   # (task, config) waiting for admission
        self.max_retries = 3
        self.task_retries = defaultdict(int)

        # Student
        self.student = None

        # Shutdown
        self._stop_event = threading.Event()

    def _log(self, msg):
        print(msg, flush=True)
        sys.stdout.flush()

    def _next_exp_id(self):
        eid = f"exp_{self.next_id:04d}"
        self.next_id += 1
        return eid

    def start(self):
        """Run the pool. Blocks until all tasks graduate or stopped."""
        # Load generators once before spawning threads (thread-safe init)
        load_generators()

        self._log(f"\n{'='*60}")
        self._log(f"RESOURCE POOL — {len(self.tasks)} tasks, GA enabled")
        self._log(f"  Device: {self.device}")
        self._log(f"  Seed configs: {len(self.seed_configs)}")
        self._log(f"  Target accuracy: {self.target_acc:.0%}")
        self._log(f"{'='*60}\n")

        # Seed: one worker per task, round-robin from seed configs
        for i, task in enumerate(self.tasks):
            cfg = self.seed_configs[i % len(self.seed_configs)].copy()
            cfg["_parent_id"] = None
            self.pending_configs.append((task, cfg))

        # Main loop: admit workers, check graduations, evolve
        generation = 0
        last_evolve = time.time()
        evolve_interval = 30  # seconds between GA generations

        while self.tasks_remaining and not self._stop_event.is_set():
            # Admit pending workers if resources allow
            self._admit_pending()

            # Check for graduations and crashes
            self._check_workers()

            # GA evolution: periodically score + mutate
            if time.time() - last_evolve > evolve_interval:
                self._evolve_generation(generation)
                generation += 1
                last_evolve = time.time()

            # Don't spin — brief sleep
            time.sleep(0.5)

        # All tasks done or stopped
        if not self.tasks_remaining:
            self._log(f"\n{'='*60}")
            self._log(f"ALL {len(self.tasks)} TASKS MASTERED!")
            self._log(f"{'='*60}")

        # Wait for student to finish a few more cycles
        if self.student and self.student.is_alive():
            self._log("Letting student finish...")
            self.student._stop_event.wait(timeout=60)

        self._log("Pool shutdown complete.")

    def _admit_pending(self):
        """Try to start one pending worker if resources allow."""
        if not self.pending_configs:
            return
        if not self.admission.can_admit():
            return

        task, cfg = self.pending_configs.pop(0)
        if task in self.teacher_tasks:
            return  # already graduated

        exp_id = self._next_exp_id()
        cfg["_parent_id"] = cfg.get("_parent_id")

        # Register lineage
        parent_id = cfg.get("_parent_id")
        if parent_id:
            with self.evo_lock:
                self.evo_state.register_lineage(exp_id, parent_id)

        worker = WorkerThread(exp_id, task, cfg, self.device, self)
        with self.worker_lock:
            self.workers[exp_id] = worker
        worker.start()

    def _check_workers(self):
        """Check for graduated or crashed workers."""
        with self.worker_lock:
            finished = [(eid, w) for eid, w in self.workers.items()
                       if not w.is_alive()]

        for exp_id, worker in finished:
            if worker.graduated:
                self._handle_graduation(worker)
            elif worker.error:
                self._log(f"  CRASH [{exp_id}] {worker.task}: {worker.error}")
                self._handle_crash(worker)
            else:
                self._log(f"  DIED [{exp_id}] {worker.task}: no error, no graduation "
                         f"(cycle={worker.cycle}, alive={worker.is_alive()})")
                self._handle_crash(worker)
            with self.worker_lock:
                del self.workers[exp_id]

    def _handle_graduation(self, worker):
        """Worker mastered its task → becomes teacher."""
        task = worker.task
        if task in self.teacher_tasks:
            return  # another worker already graduated this task

        self.teacher_tasks[task] = worker.exp_id
        self.tasks_remaining.discard(task)

        # Stop all other workers for this task
        with self.worker_lock:
            for eid, w in list(self.workers.items()):
                if w.task == task and w.exp_id != worker.exp_id:
                    w.stop()
                    self._log(f"  Stopped {eid} ({task} already graduated)")

        # Remove pending configs for this task
        self.pending_configs = [(t, c) for t, c in self.pending_configs if t != task]

        # Notify callback
        if self.on_graduate_cb:
            try:
                self.on_graduate_cb(task, worker.exp_id, worker.config)
            except Exception as e:
                self._log(f"  ERROR in on_graduate: {e}")

        # Start student if first teacher
        if len(self.teacher_tasks) == 1 and self.student is None:
            self._start_student()
        elif self.student and self.student.is_alive():
            # Hot-add teacher to student
            ckpt_path = Path("checkpoints/specialists") / f"{task}.pt"
            if ckpt_path.exists():
                self.student.add_teacher(task, ckpt_path, worker.config)

        self._log(f"\n  🎓 GRADUATED: {task} → Teacher ({worker.best_acc:.0%})"
                  f" | {len(self.teacher_tasks)}/{len(self.tasks)} teachers"
                  f" | {len(self.tasks_remaining)} remaining\n")

    def _handle_crash(self, worker):
        """Worker crashed — retry if allowed."""
        task = worker.task
        self.task_retries[task] += 1
        if self.task_retries[task] <= self.max_retries:
            self._log(f"  Retrying {task} (attempt {self.task_retries[task]})")
            cfg = worker.config.copy()
            self.pending_configs.append((task, cfg))
        else:
            self._log(f"  GIVING UP on {task} after {self.max_retries} retries")

    def _start_student(self):
        """Start the student distiller."""
        # Use a moderate config for the student
        student_cfg = {
            "d_model": 64, "d_state": 16, "headdim": 16,
            "n_kernel_layers": 3, "lr": 1e-3, "weight_decay": 0.0,
        }
        self.student = StudentThread(student_cfg, self.device, self)

        # Load all available teachers
        for task, exp_id in self.teacher_tasks.items():
            ckpt_path = Path("checkpoints/specialists") / f"{task}.pt"
            if ckpt_path.exists():
                # Get config from checkpoint
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                cfg = ckpt.get("config", student_cfg)
                self.student.add_teacher(task, ckpt_path, cfg)

        self.student.start()
        self._log(f"  📚 Student started with {len(self.teacher_tasks)} teachers")

    def _evolve_generation(self, generation):
        """GA: score workers, select parents, mutate, spawn children."""
        with self.worker_lock:
            active = list(self.workers.values())
        if not active:
            return

        # Build results for scoring
        results = []
        for w in active:
            results.append({
                "exp_id": w.exp_id,
                "task": w.task,
                "best_fresh": w.best_acc,
                "cycle": w.cycle,
                "config": w.config,
                "type_accs": {w.task: w.last_acc},
                "status": "running",
            })

        with self.evo_lock:
            scored = score_experiments(results, self.evo_state)

            # Plateau detection per task
            for task in self.tasks_remaining:
                task_workers = [r for r in scored if r["task"] == task]
                if not task_workers:
                    continue
                best = max(w["best_fresh"] for w in task_workers)

                # Check if stuck
                plateau_severity = 0.0
                for w in task_workers:
                    m = self.evo_state.get_momentum(w["exp_id"])
                    if m <= 0 and w["cycle"] > 20:
                        plateau_severity = min(3.0, (w["cycle"] - 20) / 30)

                # Select parent and mutate
                parent, reason = select_parent(
                    task_workers, self.evo_state, generation)
                if parent:
                    child_cfg = mutate_config(
                        parent["config"],
                        plateau_severity=plateau_severity,
                    )
                    child_cfg["_parent_id"] = parent["exp_id"]
                    self.pending_configs.append((task, child_cfg))

            # Update best ever for plateau detection
            all_best = max((w.best_acc for w in active), default=0)
            if all_best > self.evo_state.best_ever:
                self.evo_state.best_ever = all_best
                self.evo_state.best_ever_gen = generation
            self.evo_state.generation = generation

        # Log generation summary
        s = self.monitor.snapshot()
        n_workers = len(active)
        n_teachers = len(self.teacher_tasks)
        n_students = 1 if (self.student and self.student.is_alive()) else 0
        self._log(f"\n[Gen {generation}] Workers={n_workers} Teachers={n_teachers} "
                  f"Students={n_students} | "
                  f"CPU={s.cpu_pct:.0f}% RAM={s.ram_pct:.0f}% "
                  f"GPU={s.gpu_pct:.0f}% VRAM={s.vram_pct:.0f}%")

    def _on_worker_cycle(self, worker, acc, best_acc, loss, n_params):
        """Called by worker thread after each cycle. Pushes to Firebase."""
        if self.on_cycle:
            try:
                s = self.monitor.snapshot()
                self.on_cycle(
                    task=worker.task,
                    cycle=worker.cycle,
                    acc=acc,
                    best_acc=best_acc,
                    loss=loss,
                    exp_id=worker.exp_id,
                    config=worker.config,
                    parent_id=worker.parent_id,
                    n_params=n_params,
                    cpu_pct=s.cpu_pct,
                    ram_pct=s.ram_pct,
                    gpu_pct=s.gpu_pct,
                    vram_pct=s.vram_pct,
                    n_workers=len(self.workers),
                    n_teachers=len(self.teacher_tasks),
                    n_students=1 if (self.student and self.student.is_alive()) else 0,
                    teacher_tasks=dict(self.teacher_tasks),
                    tasks_remaining=list(self.tasks_remaining),
                    lineage=self.evo_state.lineage,
                    generation=self.evo_state.generation,
                )
            except Exception as e:
                self._log(f"  on_cycle error: {e}")

    def _on_student_cycle(self, student, fresh, loss, type_accs, n_params):
        """Called by student thread after each cycle."""
        # Push student data to Firebase via the same on_cycle
        if self.on_cycle:
            try:
                s = self.monitor.snapshot()
                self.on_cycle(
                    task="_student",
                    cycle=student.cycle,
                    acc=fresh,
                    best_acc=student.best_fresh,
                    loss=loss,
                    exp_id="student",
                    config=student.config,
                    parent_id=None,
                    n_params=n_params,
                    cpu_pct=s.cpu_pct,
                    ram_pct=s.ram_pct,
                    gpu_pct=s.gpu_pct,
                    vram_pct=s.vram_pct,
                    n_workers=len(self.workers),
                    n_teachers=len(self.teacher_tasks),
                    n_students=1,
                    teacher_tasks=dict(self.teacher_tasks),
                    tasks_remaining=list(self.tasks_remaining),
                    lineage=self.evo_state.lineage,
                    generation=self.evo_state.generation,
                    student_type_accs=type_accs,
                )
            except Exception as e:
                self._log(f"  student on_cycle error: {e}")

    def shutdown(self):
        """Graceful shutdown."""
        self._stop_event.set()
        with self.worker_lock:
            for w in self.workers.values():
                w.stop()
        if self.student:
            self.student.stop()
