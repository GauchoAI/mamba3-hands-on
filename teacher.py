"""Teacher loader + logits extractor for ptxd's distillation path.

Specialist_trainer.py looked up a teacher per task via ModelRegistry, ran
its forward on each batch, and blended a KL term at α=0.3, T=3.0. This
module mirrors that behaviour but emits teacher logits as the
(pos, logits) tuples that batch_writer expects, so ptxd's kd_apply
kernel can consume them.

Two design choices:
  * Use a live teacher (load .pt → ProgressiveModel → forward) rather
    than `_cache.pt`, so the teacher logits match the *current* batch
    distribution exactly. _cache.pt was generated at a fixed moment
    with a fixed sampler — staleness risk if the curriculum advances.
  * Compute logits at supervised positions only. The Hinton blend in
    ptxd's kd_apply only consumes those rows, so shipping the rest
    would be wasted bandwidth.

If a task has no teacher in the ModelRegistry, `find_teacher_for_task`
returns None and ptxd_specialist falls back to plain CE.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))


def find_teacher_for_task(task):
    """Return (teacher_pt_path, teacher_config, teacher_acc) or None.

    Looks up ModelRegistry. Returns None if no teacher is registered, the
    teacher file is missing, or any import fails (e.g. running on a node
    without the registry available).
    """
    try:
        from registry.model_registry import ModelRegistry
    except Exception as e:
        sys.stderr.write(f"[teacher] ModelRegistry unavailable: {e}\n")
        return None
    try:
        reg = ModelRegistry()
        if not reg.has_teacher(task):
            return None
        # Use .local_dir/{task}.pt as the canonical path; ModelRegistry
        # also caches teachers from other nodes here.
        teacher_path = Path(reg.local_dir) / f"{task}.pt"
        if not teacher_path.exists():
            return None
        # Use get_teacher to discover config/accuracy without loading the
        # full model on this thread (we'll load it ourselves below to keep
        # device control).
        result = reg.get_teacher(task, device="cpu")
        if not result:
            return None
        _model, cfg, acc = result
        return (str(teacher_path), cfg, acc)
    except Exception as e:
        sys.stderr.write(f"[teacher] discovery for task={task!r} failed: {e}\n")
        return None


def load_teacher_model(teacher_pt_path, device="cuda"):
    """Load a ProgressiveModel from a saved .pt checkpoint, ready for
    inference. Returns (model, device_actually_used). Falls back to CPU
    if `device='cuda'` was requested but CUDA is unusable (driver
    mismatch, no GPU, etc.) — ptxd has its own CUDA context for the
    student so we can lose teacher GPU acceleration without breaking
    training, just slower per-batch teacher forward."""
    import torch
    from progressive_model import ProgressiveModel
    ck = torch.load(teacher_pt_path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = ProgressiveModel(
        d_model=cfg["d_model"],
        d_state=cfg["d_state"],
        expand=2,
        headdim=cfg["headdim"],
    )
    for _ in range(cfg["n_kernel_layers"]):
        model.add_kernel_layer()
    model.load_state_dict(ck["model"], strict=False)
    model.eval()
    actual_device = "cpu"
    if device != "cpu":
        try:
            model = model.to(device)
            # Smoke-test the forward path (catches CUDA driver mismatches
            # at this layer rather than at first batch).
            with torch.no_grad():
                _ = model(torch.zeros((1, 4), dtype=torch.long, device=device))
            actual_device = device
        except Exception as e:
            sys.stderr.write(f"[teacher] {device} forward failed ({e}); falling back to CPU\n")
            model = model.cpu()
            actual_device = "cpu"
    return model, actual_device


def compute_teacher_logits_for_examples(
    teacher_model, examples, vocab_size, batch_size=64, device="cuda",
):
    """Run the teacher forward on every example and return per-example
    teacher_logits in the (pos, logits) format batch_writer expects.

    `examples` is a list of (tokens, targets) — the same shape task_runner
    produces. For each example we forward the FULL token sequence through
    the teacher and extract logits at every supervised position
    (target != IGNORE).

    Batched in groups of `batch_size` for speed; pads to max-length within
    a batch since ProgressiveModel handles padding via the same convention
    specialist_trainer used.
    """
    import torch
    from progressive_model import PAD

    IGNORE = 0xFFFFFFFF
    teacher_logits_per_ex = []
    n = len(examples)
    i = 0
    while i < n:
        batch = examples[i:i + batch_size]
        max_len = max(len(toks) for toks, _ in batch)
        padded = torch.full((len(batch), max_len), PAD,
                            dtype=torch.long, device=device)
        for j, (toks, _) in enumerate(batch):
            padded[j, :len(toks)] = torch.tensor(toks, dtype=torch.long, device=device)

        with torch.no_grad():
            t_logits = teacher_model(padded)  # (B, L, V_t)

        for j, (toks, tgts) in enumerate(batch):
            # Per-example slots: collect teacher logits at supervised positions.
            # Note vocab_size of teacher should match ptxd's; if it doesn't,
            # we slice / pad. Both are 260 in production today.
            slots = []
            for pos, tgt in enumerate(tgts):
                if tgt == IGNORE:
                    continue
                # The teacher's logits at position `pos` predict the next
                # token — same convention ptxd uses. Convert to a Python list
                # of floats; the f32 precision is what kd_apply consumes.
                row = t_logits[j, pos].cpu().tolist()
                if len(row) < vocab_size:
                    row = row + [0.0] * (vocab_size - len(row))
                elif len(row) > vocab_size:
                    row = row[:vocab_size]
                slots.append((pos, row))
            teacher_logits_per_ex.append(slots)
        i += batch_size
    return teacher_logits_per_ex


if __name__ == "__main__":
    # Smoke test: discover a teacher for parity (or report none).
    import sys
    task = sys.argv[1] if len(sys.argv) > 1 else "parity"
    found = find_teacher_for_task(task)
    if found is None:
        print(f"no teacher registered for task={task!r}")
        sys.exit(0)
    path, cfg, acc = found
    print(f"teacher for {task!r}: {path}  acc={acc:.0%}  d={cfg.get('d_model')} L={cfg.get('n_kernel_layers')}")
