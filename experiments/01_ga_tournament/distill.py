"""
Distillation: merge specialist knowledge into one generalist.

The MAP produced specialists (genetic algorithm, 49 workers).
This is the REDUCE: train one student from ALL specialists' dark knowledge.

Each specialist's full output distribution encodes structure that
the Python grader (right/wrong) never could. The student is forced
to find shared internal primitives because it must reproduce all
specialists' behaviors in one set of weights.

Usage:
    python distill.py                           # auto-find specialists, train student
    python distill.py --student-d-model 96      # bigger student
    python distill.py --temperature 4.0         # softer distributions
"""
import os
os.environ["PYTHONUNBUFFERED"] = "1"
import sys
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict

from progressive_model import ProgressiveModel, ByteTokenizer, VOCAB_SIZE, PAD, BOS, EOS, SEP
from generators.teacher import AdaptiveTeacher
from metrics_db import MetricsWriter


# ── PCGrad: project conflicting task gradients ──────────────────────

def pcgrad_project(task_grads):
    """
    Given a list of per-task gradient dicts, project away conflicts.
    Returns the combined gradient with conflicts resolved.

    From Yu et al. 2020: if two task gradients have negative cosine
    similarity, project each onto the other's orthogonal complement.
    """
    import copy
    projected = [copy.deepcopy(g) for g in task_grads]

    for i in range(len(projected)):
        for j in range(len(projected)):
            if i == j:
                continue
            # Flatten both
            gi = torch.cat([p.flatten() for p in projected[i].values()])
            gj = torch.cat([p.flatten() for p in task_grads[j].values()])

            dot = (gi * gj).sum()
            if dot < 0:
                # Conflict — project gi onto orthogonal complement of gj
                proj = dot / (gj.norm() ** 2 + 1e-12)
                idx = 0
                for name in projected[i]:
                    numel = projected[i][name].numel()
                    projected[i][name] -= proj * task_grads[j][name]
                    idx += numel

    # Average the projected gradients
    combined = {}
    for name in projected[0]:
        combined[name] = sum(p[name] for p in projected) / len(projected)
    return combined


# ── Find best specialist per task ───────────────────────────────────

def find_specialists(runs_dir="runs"):
    """Find the best checkpoint for each task from the experiment zoo."""
    runs = Path(runs_dir)
    task_best = {}  # task → (exp_id, accuracy, checkpoint_path, config)

    for exp_dir in sorted(runs.iterdir()):
        if not exp_dir.is_dir():
            continue
        metrics_path = exp_dir / "metrics.json"
        ckpt_path = exp_dir / "checkpoint.pt"
        if not metrics_path.exists() or not ckpt_path.exists():
            continue

        try:
            m = json.load(open(metrics_path))
            type_accs = m.get("type_accs", {})
            config = m.get("config", {})
            for task, acc in type_accs.items():
                if task not in task_best or acc > task_best[task][1]:
                    task_best[task] = (exp_dir.name, acc, str(ckpt_path), config)
        except Exception:
            continue

    return task_best


def load_specialist(ckpt_path, config, device):
    """Load a specialist model from checkpoint."""
    model = ProgressiveModel(
        d_model=config.get("d_model", 64),
        d_state=config.get("d_state", 16),
        expand=2,
        headdim=config.get("headdim", 16),
    ).to(device)
    for _ in range(config.get("n_kernel_layers", 1)):
        model.add_kernel_layer()

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


# ── Distillation loss ───────────────────────────────────────────────

def distillation_loss(student_logits, teacher_logits, hard_targets,
                      temperature=3.0, alpha=0.7):
    """
    Combine soft (teacher distribution) and hard (ground truth) losses.

    soft_loss: KL divergence between student and teacher softmax at temperature T
    hard_loss: standard cross-entropy on the correct answer

    alpha controls the mix: 0.7 means 70% teacher knowledge, 30% hard label.
    Temperature softens distributions to expose dark knowledge.
    """
    # Soft targets from teacher
    soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
    soft_student = F.log_softmax(student_logits / temperature, dim=-1)
    soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (temperature ** 2)

    # Hard targets
    hard_loss = F.cross_entropy(student_logits, hard_targets)

    return alpha * soft_loss + (1 - alpha) * hard_loss


# ── Main distillation loop ──────────────────────────────────────────

def distill(args):
    # Device
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}", flush=True)

    # Find specialists
    print(f"\nFinding specialists in {args.runs_dir}...", flush=True)
    specialists = find_specialists(args.runs_dir)
    print(f"Found specialists for {len(specialists)} tasks:", flush=True)
    for task, (exp_id, acc, path, cfg) in sorted(specialists.items()):
        print(f"  {task}: {acc:.0%} ({exp_id}, d={cfg.get('d_model','?')}, "
              f"L={cfg.get('n_kernel_layers','?')})", flush=True)

    if not specialists:
        print("No specialists found!")
        return

    # Load specialist models
    print(f"\nLoading specialist models...", flush=True)
    teacher_models = {}
    for task, (exp_id, acc, path, cfg) in specialists.items():
        if acc < 0.1:  # skip tasks with < 10% accuracy
            continue
        try:
            model = load_specialist(path, cfg, device)
            teacher_models[task] = model
            print(f"  Loaded {task} teacher ({exp_id})", flush=True)
        except Exception as e:
            print(f"  Failed to load {task}: {e}", flush=True)

    print(f"\n{len(teacher_models)} teachers loaded.", flush=True)

    # Create student
    student = ProgressiveModel(
        d_model=args.student_d_model,
        d_state=args.student_d_state,
        expand=2,
        headdim=args.student_headdim,
    ).to(device)
    for _ in range(args.student_layers):
        student.add_kernel_layer()
    student.set_mode("kernel")

    n_params = sum(p.numel() for p in student.parameters())
    print(f"\nStudent: d={args.student_d_model}, L={args.student_layers}, "
          f"{n_params:,} params", flush=True)

    # Optimizer
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

    # Data generator
    teacher_gen = AdaptiveTeacher(sequential_unlock=False)  # all tasks available
    tok = ByteTokenizer()

    # Metrics
    metrics = MetricsWriter()
    metrics.register_experiment("student", {
        "d_model": args.student_d_model,
        "d_state": args.student_d_state,
        "n_kernel_layers": args.student_layers,
        "type": "distilled_student",
    }, n_params)

    print(f"\nTraining for {args.cycles} cycles, {args.steps_per_cycle} steps/cycle",
          flush=True)
    print(f"Temperature: {args.temperature}, Alpha: {args.alpha}", flush=True)
    print(f"PCGrad: {'ON' if args.pcgrad else 'OFF'}", flush=True)
    print(flush=True)

    best_fresh = 0.0

    for cycle in range(1, args.cycles + 1):
        t0 = time.time()
        student.train()

        # Generate mixed-task data
        raw = teacher_gen.generate(5000)
        by_task = defaultdict(list)
        for ex in raw:
            by_task[ex["type"]].append(ex)

        cycle_loss = 0.0
        n_steps = 0

        for step in range(args.steps_per_cycle):
            if args.pcgrad:
                # Per-task gradients for PCGrad
                task_grads_list = []
                total_loss = 0.0

                for task, task_teacher in teacher_models.items():
                    examples = by_task.get(task, [])
                    if not examples:
                        continue

                    # Sample batch for this task
                    batch_exs = [examples[i % len(examples)]
                                for i in range(min(args.batch_size // len(teacher_models), 16))]
                    batch_tokens = []
                    batch_seps = []
                    for ex in batch_exs:
                        tokens, sep = tok.encode_curriculum(ex)
                        batch_tokens.append(tokens)
                        batch_seps.append(sep)

                    max_len = max(len(t) for t in batch_tokens)
                    token_tensor = torch.full((len(batch_tokens), max_len), PAD,
                                            dtype=torch.long, device=device)
                    for i, t in enumerate(batch_tokens):
                        token_tensor[i, :len(t)] = torch.tensor(t)

                    # Get teacher distribution
                    with torch.no_grad():
                        teacher_logits = task_teacher(token_tensor)

                    # Get student distribution
                    student_logits = student(token_tensor)

                    # Compute distillation loss on output positions
                    B, L, V = student_logits.shape
                    loss = torch.tensor(0.0, device=device)
                    count = 0
                    for b in range(B):
                        sep = batch_seps[b]
                        for t in range(sep, L - 1):
                            target = token_tensor[b, t + 1]
                            if target == PAD:
                                break
                            loss += distillation_loss(
                                student_logits[b, t].unsqueeze(0),
                                teacher_logits[b, t].unsqueeze(0),
                                target.unsqueeze(0),
                                temperature=args.temperature,
                                alpha=args.alpha,
                            )
                            count += 1
                    if count > 0:
                        loss = loss / count

                    # Compute per-task gradient
                    opt.zero_grad()
                    loss.backward()
                    task_grad = {name: p.grad.clone() for name, p in student.named_parameters()
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
                    cycle_loss += total_loss / len(task_grads_list)
                    n_steps += 1

            else:
                # Simple: mixed batch, all tasks together
                all_exs = []
                for task_examples in by_task.values():
                    all_exs.extend(task_examples[:args.batch_size // len(by_task)])

                if not all_exs:
                    continue

                batch_exs = all_exs[:args.batch_size]
                batch_tokens = []
                batch_seps = []
                batch_tasks = []
                for ex in batch_exs:
                    tokens, sep = tok.encode_curriculum(ex)
                    batch_tokens.append(tokens)
                    batch_seps.append(sep)
                    batch_tasks.append(ex["type"])

                max_len = max(len(t) for t in batch_tokens)
                token_tensor = torch.full((len(batch_tokens), max_len), PAD,
                                        dtype=torch.long, device=device)
                for i, t in enumerate(batch_tokens):
                    token_tensor[i, :len(t)] = torch.tensor(t)

                # Get teacher distributions (task-specific)
                student_logits = student(token_tensor)
                B, L, V = student_logits.shape

                loss = torch.tensor(0.0, device=device)
                count = 0
                for b in range(B):
                    task = batch_tasks[b]
                    if task not in teacher_models:
                        # No teacher — use hard labels only
                        sep = batch_seps[b]
                        for t in range(sep, L - 1):
                            target = token_tensor[b, t + 1]
                            if target == PAD:
                                break
                            loss += F.cross_entropy(student_logits[b, t].unsqueeze(0),
                                                   target.unsqueeze(0))
                            count += 1
                    else:
                        # Distill from specialist
                        with torch.no_grad():
                            t_logits = teacher_models[task](token_tensor[b:b+1])
                        sep = batch_seps[b]
                        for t in range(sep, L - 1):
                            target = token_tensor[b, t + 1]
                            if target == PAD:
                                break
                            loss += distillation_loss(
                                student_logits[b, t].unsqueeze(0),
                                t_logits[0, t].unsqueeze(0),
                                target.unsqueeze(0),
                                temperature=args.temperature,
                                alpha=args.alpha,
                            )
                            count += 1

                if count > 0:
                    loss = loss / count
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                    opt.step()
                    cycle_loss += loss.item()
                    n_steps += 1

        cycle_loss /= max(n_steps, 1)
        elapsed = time.time() - t0

        # Evaluate student
        from generators.level0_patterns import generate_dataset
        eval_raw = generate_dataset(300)
        eval_by_type = defaultdict(list)
        for ex in eval_raw:
            eval_by_type[ex["type"]].append(ex)

        type_accs = {}
        student.eval()
        with torch.no_grad():
            for task_type, examples in eval_by_type.items():
                correct = 0
                total = 0
                for ex in examples[:30]:
                    tokens, sep_pos = tok.encode_curriculum(ex)
                    out_bytes = list(ex["output"].encode("utf-8"))
                    t = torch.tensor([tokens], dtype=torch.long, device=device)
                    logits = student(t)
                    ok = True
                    for j, expected in enumerate(out_bytes):
                        p = sep_pos + j
                        if p < logits.shape[1]:
                            if logits[0, p].argmax().item() != expected:
                                ok = False
                                break
                        else:
                            ok = False
                    if ok:
                        correct += 1
                    total += 1
                type_accs[task_type] = correct / max(total, 1)
        student.train()

        fresh = sum(type_accs.values()) / max(len(type_accs), 1)
        best_fresh = max(best_fresh, fresh)

        # Print
        print(f"[Cycle {cycle}] loss={cycle_loss:.3f}  fresh={fresh:.1%}  "
              f"best={best_fresh:.1%}  {elapsed:.1f}s", flush=True)
        for task, acc in sorted(type_accs.items()):
            if acc > 0:
                specialist_acc = specialists.get(task, (None, 0))[1]
                bar = "█" * int(acc * 20)
                spec_bar = "░" * int(specialist_acc * 20)
                print(f"    {task:25s} {acc:.0%} {bar}  (specialist: {specialist_acc:.0%})",
                      flush=True)

        # Log
        metrics.log_cycle("student", cycle, cycle_loss, fresh, best_fresh, elapsed_s=elapsed)
        metrics.log_tasks("student", cycle, type_accs)

        # Firebase
        try:
            from lab_platform.firebase_push import push_experiment_cycle, evt_new_best
            push_experiment_cycle("student", cycle, fresh, cycle_loss, type_accs)
            if fresh == best_fresh and fresh > 0:
                evt_new_best("student", fresh, best_fresh,
                            {"type": "distilled", "d_model": args.student_d_model,
                             "layers": args.student_layers})
        except Exception:
            pass

        # Checkpoint
        if cycle % 10 == 0 or fresh == best_fresh:
            ckpt_dir = Path("checkpoints")
            ckpt_dir.mkdir(exist_ok=True)
            torch.save({
                "model": student.state_dict(),
                "cycle": cycle,
                "fresh": fresh,
                "best_fresh": best_fresh,
                "config": {
                    "d_model": args.student_d_model,
                    "d_state": args.student_d_state,
                    "n_kernel_layers": args.student_layers,
                    "type": "distilled_student",
                },
            }, ckpt_dir / "student_best.pt")

    print(f"\n{'='*60}", flush=True)
    print(f"DISTILLATION COMPLETE", flush=True)
    print(f"  Best fresh: {best_fresh:.1%}", flush=True)
    print(f"  Student: d={args.student_d_model}, L={args.student_layers}, "
          f"{n_params:,} params", flush=True)
    print(f"  Teachers: {len(teacher_models)} specialists", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distill specialists into one generalist")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--student-d-model", type=int, default=64)
    parser.add_argument("--student-d-state", type=int, default=16)
    parser.add_argument("--student-headdim", type=int, default=16)
    parser.add_argument("--student-layers", type=int, default=3)
    parser.add_argument("--cycles", type=int, default=200)
    parser.add_argument("--steps-per-cycle", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--pcgrad", action="store_true", default=True)
    parser.add_argument("--no-pcgrad", action="store_false", dest="pcgrad")
    args = parser.parse_args()
    distill(args)
