"""Real per-tile trainer for the curriculum corpus.

Trains a single persistent CortexLM (1M-param byte-level Mamba-3) tile by
tile. State persists across calls in checkpoints/student.pt — each tile
is one curriculum step, not a fresh run.

Loss for v1: byte-level next-token cross-entropy on the flattened record
(concept ⧺ question ⧺ solution ⧺ paraphrase). Paraphrase-invariance JEPA
loss is wired but kept off by default — flip --jepa-weight > 0 to enable.

The trainer plugs into orchestrator.py via train_on_tile() — same shape
as the previous stub, so the wait-on-empty/priority-queue plumbing works
unchanged.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from cortex_counting import CortexLM, CortexLMConfig
from curriculum import Tile, TileRegistry


# --- Config ---

# 4L d_model=192 ≈ 1M params, matches the jepa/ baseline.
DEFAULT_CFG_KWARGS = dict(
    n_layers=4,
    d_model=192,
    max_seq_len=512,
    use_counter=True,
    n_counters=2,
)
CHECKPOINT_NAME = "student.pt"


# --- Data ---

def format_record(rec: dict) -> bytes:
    """Flatten one textbook example to a byte string for next-byte training."""
    parts = [
        rec.get("concept", ""),
        rec.get("question", ""),
        rec.get("solution", ""),
        rec.get("paraphrase", ""),
    ]
    return ("\n\n".join(p for p in parts if p) + "\n\n").encode("utf-8", errors="ignore")


def format_pair(rec: dict) -> tuple[bytes, bytes]:
    """For paraphrase-invariance: two byte strings sharing the same prompt
    but ending in solution vs paraphrase. Pooled embeddings should align."""
    concept = rec.get("concept", "")
    question = rec.get("question", "")
    prompt = (concept + "\n\n" + question + "\n\n").encode("utf-8", errors="ignore")
    sol = rec.get("solution", "").encode("utf-8", errors="ignore")
    par = rec.get("paraphrase", "").encode("utf-8", errors="ignore")
    return prompt + sol, prompt + par


def load_jsonl_records(path: Path) -> list[dict]:
    out = []
    if not path.exists():
        return out
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("_error"):
                continue
            if not all(rec.get(k) for k in ("concept", "question", "solution", "paraphrase")):
                continue
            out.append(rec)
    return out


def make_byte_batch(byte_seqs: list[bytes], batch_size: int, max_len: int,
                    device: str) -> torch.Tensor:
    sampled = random.choices(byte_seqs, k=batch_size)
    seqs = [list(s[:max_len]) for s in sampled]
    L = max(2, max(len(s) for s in seqs))
    L = min(L, max_len)
    padded = [s[:L] + [0] * (L - len(s)) for s in seqs]
    return torch.tensor(padded, dtype=torch.long, device=device)


def make_pair_batch(pair_seqs: list[tuple[bytes, bytes]], batch_size: int,
                    max_len: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    sampled = random.choices(pair_seqs, k=batch_size)
    a_list = [list(s[0][:max_len]) for s in sampled]
    b_list = [list(s[1][:max_len]) for s in sampled]
    L = max(2, max(max(len(a) for a in a_list), max(len(b) for b in b_list)))
    L = min(L, max_len)
    a_pad = [a[:L] + [0] * (L - len(a)) for a in a_list]
    b_pad = [b[:L] + [0] * (L - len(b)) for b in b_list]
    return (
        torch.tensor(a_pad, dtype=torch.long, device=device),
        torch.tensor(b_pad, dtype=torch.long, device=device),
    )


# --- Model lifecycle ---

@dataclass
class TrainerState:
    model: CortexLM
    optimizer: torch.optim.Optimizer
    step: int


def get_or_init(cfg: CortexLMConfig, ckpt_path: Path, device: str, lr: float
                ) -> TrainerState:
    model = CortexLM(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    step = 0
    if ckpt_path.exists():
        try:
            ck = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ck["model"])
            if "optimizer" in ck:
                optimizer.load_state_dict(ck["optimizer"])
            step = int(ck.get("step", 0))
            print(f"  [trainer] resumed from step {step}", flush=True)
        except Exception as e:
            print(f"  [trainer] checkpoint load failed ({e}); starting fresh", flush=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  [trainer] model: {n_params:,} params on {device}", flush=True)
    return TrainerState(model=model, optimizer=optimizer, step=step)


def save_state(state: TrainerState, ckpt_path: Path) -> None:
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = ckpt_path.with_suffix(".tmp")
    torch.save(
        {
            "model": state.model.state_dict(),
            "optimizer": state.optimizer.state_dict(),
            "step": state.step,
        },
        tmp,
    )
    tmp.replace(ckpt_path)


# --- Loss ---

def byte_ce_loss(model: CortexLM, batch: torch.Tensor) -> torch.Tensor:
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    logits = model(inputs)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="mean",
    )


def paraphrase_invariance_loss(model: CortexLM, a: torch.Tensor, b: torch.Tensor
                               ) -> torch.Tensor:
    """Minimise distance between the two pooled-intent embeddings of a pair
    that share prompt but end in solution vs paraphrase. Uses the JEPA
    return path of CortexLM."""
    _, _, _, intent_a = model(a, return_jepa=True)
    _, _, _, intent_b = model(b, return_jepa=True)
    # Cosine-distance with stop-gradient on the target half (BYOL-style),
    # alternating which half is the target across calls is overkill for v1;
    # symmetric MSE is fine for proving the wiring works.
    return F.mse_loss(intent_a, intent_b)


# --- Per-tile training step ---

def train_on_tile(
    tile: Tile,
    jsonl_path: Path,
    registry: TileRegistry,
    model_root: Path,
    device: str = "cpu",
    steps: int = 80,
    batch_size: int = 8,
    lr: float = 3e-4,
    jepa_weight: float = 0.0,
    max_seq_len: int | None = None,
) -> dict:
    """Train the global student on one tile's records for `steps` updates,
    then save. Returns a result dict for the registry/log."""
    cfg_kwargs = dict(DEFAULT_CFG_KWARGS)
    if max_seq_len:
        cfg_kwargs["max_seq_len"] = max_seq_len
    cfg = CortexLMConfig(**cfg_kwargs)
    ckpt_path = model_root / CHECKPOINT_NAME

    records = load_jsonl_records(jsonl_path)
    if not records:
        return {"tile_id": tile.id, "skipped": "no_records"}

    byte_seqs = [format_record(r) for r in records]
    pair_seqs = [format_pair(r) for r in records] if jepa_weight > 0 else []

    state = get_or_init(cfg, ckpt_path, device, lr)
    state.model.train()

    losses_ce: list[float] = []
    losses_jepa: list[float] = []
    t0 = time.time()
    for _ in range(steps):
        batch = make_byte_batch(byte_seqs, batch_size, cfg.max_seq_len, device)
        ce = byte_ce_loss(state.model, batch)
        loss = ce
        jepa_val = 0.0
        if jepa_weight > 0 and pair_seqs:
            a, b = make_pair_batch(pair_seqs, batch_size, cfg.max_seq_len, device)
            jepa = paraphrase_invariance_loss(state.model, a, b)
            jepa_val = float(jepa.item())
            loss = ce + jepa_weight * jepa
        state.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(state.model.parameters(), 1.0)
        state.optimizer.step()
        state.step += 1
        losses_ce.append(float(ce.item()))
        if jepa_val:
            losses_jepa.append(jepa_val)

    save_state(state, ckpt_path)
    dt = time.time() - t0
    avg_ce = sum(losses_ce) / len(losses_ce)
    final_ce = sum(losses_ce[-min(20, len(losses_ce)) :]) / min(20, len(losses_ce))
    avg_jepa = (sum(losses_jepa) / len(losses_jepa)) if losses_jepa else None

    status = registry.get(tile.id)
    status.last_trained_at = time.time()
    status.n_validated = len(records)
    # Heuristic acc: shrink loss into [0,1]. Rough but useful for the dashboard.
    status.student_acc = 1.0 / (1.0 + final_ce)
    registry.set(status)

    return {
        "tile_id": tile.id,
        "n_examples": len(records),
        "steps": steps,
        "global_step": state.step,
        "avg_ce_loss": round(avg_ce, 4),
        "final_ce_loss": round(final_ce, 4),
        "avg_jepa_loss": round(avg_jepa, 4) if avg_jepa is not None else None,
        "elapsed_s": round(dt, 2),
        "device": device,
    }


def best_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
