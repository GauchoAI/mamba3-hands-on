"""
Register Inspector: observe the SSM's hidden state as it processes tokens.

Loads a checkpoint, runs examples through the model with state capture,
produces a compact inspection report showing how registers activate
per token.

The report is small enough for Firebase (~500 bytes per example).
Saved alongside checkpoint as {task}_inspection.json.

Usage:
    from register_inspector import inspect_model
    report = inspect_model("parity", n_examples=5)
    # report is a dict ready for Firebase push
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import json
import time
import torch
import torch.nn.functional as F
from pathlib import Path


def ssm_scan_with_states(inp, decay, C, x, z, D):
    """Like ssm_scan_jit but also returns intermediate hidden states.

    Returns (y, states) where states is (B, L, H, hD, dS).
    Uses pure Python — NOT for training, only for inspection.
    """
    B, L, H, hD, dS = inp.shape
    h = torch.zeros(B, H, hD, dS, device=inp.device, dtype=inp.dtype)
    y = torch.empty(B, L, H, hD, device=inp.device, dtype=inp.dtype)
    states = torch.empty(B, L, H, hD, dS, device=inp.device, dtype=inp.dtype)

    z_silu = z * torch.sigmoid(z)

    for t in range(L):
        h = decay[:, t, :, None, None] * h + inp[:, t]
        states[:, t] = h
        y_t = (h * C[:, t, :, None, :]).sum(dim=-1)
        y_t = y_t + D[None, :, None] * x[:, t]
        y[:, t] = y_t * z_silu[:, t]

    return y, states


def inspect_example(model, tokens, sep, device):
    """Run one example through the model, capture register states.

    Returns per-token metrics: state_norm, state_delta, top_registers.
    """
    model.eval()

    # We need to intercept the SSM scan call inside the model.
    # Monkey-patch ssm_scan temporarily to capture states.
    from lab_platform import ssm_triton
    original_scan = ssm_triton.ssm_scan
    captured_states = {}

    def capturing_scan(inp, decay, C, x, z, D):
        y, states = ssm_scan_with_states(inp, decay, C, x, z, D)
        captured_states["states"] = states.detach().cpu()
        return y

    try:
        ssm_triton.ssm_scan = capturing_scan

        t = torch.tensor([tokens], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(t)

        if "states" not in captured_states:
            return None

        states = captured_states["states"][0]  # (L, H, hD, dS)
        L = states.shape[0]

        # Compute per-token metrics
        per_token = []
        prev_norm = 0.0
        for i in range(L):
            h_t = states[i]  # (H, hD, dS)
            norm = h_t.norm().item()
            delta = abs(norm - prev_norm)

            # Top registers: which (head, dim) indices have highest activation
            flat = h_t.abs().reshape(-1)
            top_k = min(5, flat.shape[0])
            top_vals, top_idx = flat.topk(top_k)

            # Get predicted token at this position
            pred_byte = logits[0, i].argmax().item() if i < logits.shape[1] else -1
            confidence = F.softmax(logits[0, i], dim=-1).max().item() if i < logits.shape[1] else 0

            # Decode token
            tok_char = chr(tokens[i]) if tokens[i] < 256 else {256: "BOS", 257: "EOS", 258: "SEP", 259: "PAD"}.get(tokens[i], "?")

            per_token.append({
                "pos": i,
                "token": tok_char,
                "token_id": tokens[i],
                "state_norm": round(norm, 2),
                "state_delta": round(delta, 2),
                "top_registers": top_idx.tolist(),
                "top_activations": [round(v, 3) for v in top_vals.tolist()],
                "predicted": pred_byte,
                "confidence": round(confidence, 3),
                "is_output": i >= sep,
            })
            prev_norm = norm

        return per_token

    finally:
        ssm_triton.ssm_scan = original_scan


def inspect_model(task, n_examples=5, device=None):
    """Run inspection on a task's current checkpoint.

    Returns a compact report dict suitable for Firebase.
    """
    from progressive_model import ProgressiveModel, ByteTokenizer
    from specialist_trainer import load_generators
    import specialist_trainer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load checkpoint
    ckpt_path = Path("checkpoints/specialists") / f"{task}.pt"
    if not ckpt_path.exists():
        return None

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})

    model = ProgressiveModel(
        d_model=config.get("d_model", 64),
        d_state=config.get("d_state", 16),
        expand=2,
        headdim=config.get("headdim", 16),
    ).to(device)
    for _ in range(config.get("n_kernel_layers", 3)):
        model.add_kernel_layer()
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Load generator
    load_generators()
    gen_fn = specialist_trainer.GENERATORS.get(task)
    if not gen_fn:
        return None
    tok = ByteTokenizer()

    # Inspect examples
    examples = []
    for _ in range(n_examples):
        ex = gen_fn()
        tokens, sep = tok.encode_curriculum(ex)
        expected = ex["output"]

        per_token = inspect_example(model, tokens, sep, device)
        if per_token is None:
            continue

        # Check if model got it right
        pred_bytes = []
        for j, expected_byte in enumerate(expected.encode("utf-8")):
            p = sep + j
            if p < len(per_token):
                pred_bytes.append(per_token[p]["predicted"])
        predicted = bytes(b for b in pred_bytes if b < 256).decode("utf-8", errors="replace")
        correct = predicted == expected

        # Summary metrics
        output_tokens = [t for t in per_token if t["is_output"]]
        input_tokens = [t for t in per_token if not t["is_output"]]

        # Where did the biggest state change happen?
        max_delta_pos = max(range(len(per_token)), key=lambda i: per_token[i]["state_delta"])

        examples.append({
            "input": ex["input"],
            "expected": expected,
            "predicted": predicted,
            "correct": correct,
            "max_delta_pos": max_delta_pos,
            "max_delta_token": per_token[max_delta_pos]["token"],
            "max_delta_value": per_token[max_delta_pos]["state_delta"],
            "input_final_norm": round(input_tokens[-1]["state_norm"], 2) if input_tokens else 0,
            "output_avg_confidence": round(
                sum(t["confidence"] for t in output_tokens) / max(len(output_tokens), 1), 3),
            "per_token": per_token,
        })

    # Build report
    n_correct = sum(1 for e in examples if e["correct"])
    report = {
        "task": task,
        "accuracy": round(n_correct / max(len(examples), 1), 3),
        "n_examples": len(examples),
        "config": {
            "d_model": config.get("d_model", 64),
            "n_kernel_layers": config.get("n_kernel_layers", 3),
            "d_state": config.get("d_state", 16),
        },
        "checkpoint_accuracy": round(ckpt.get("accuracy", 0), 3),
        "checkpoint_cycles": ckpt.get("cycles", 0),
        "examples": examples,
        "timestamp": time.time(),
    }

    return report


def save_and_push(task, report, push_firebase=True):
    """Save inspection report to disk and optionally push to Firebase."""
    if not report:
        return

    # Save alongside checkpoint
    path = Path("checkpoints/specialists") / f"{task}_inspection.json"
    with open(path, "w") as f:
        # Save full report to disk
        json.dump(report, f, indent=2)
    print(f"  Inspection saved → {path}", flush=True)

    # Push compact version to Firebase (no per_token details — too large)
    if push_firebase:
        try:
            from lab_platform import firebase_push as fb
            compact = {
                "task": report["task"],
                "accuracy": report["accuracy"],
                "config": report["config"],
                "checkpoint_accuracy": report["checkpoint_accuracy"],
                "checkpoint_cycles": report["checkpoint_cycles"],
                "n_examples": report["n_examples"],
                "examples": [{
                    "input": e["input"],
                    "expected": e["expected"],
                    "predicted": e["predicted"],
                    "correct": e["correct"],
                    "max_delta_pos": e["max_delta_pos"],
                    "max_delta_token": e["max_delta_token"],
                    "max_delta_value": e["max_delta_value"],
                    "input_final_norm": e["input_final_norm"],
                    "output_avg_confidence": e["output_avg_confidence"],
                    # Per-token: only state_norm and state_delta (compact)
                    "state_norms": [t["state_norm"] for t in e["per_token"]],
                    "state_deltas": [t["state_delta"] for t in e["per_token"]],
                    "tokens": [t["token"] for t in e["per_token"]],
                } for e in report["examples"]],
                "timestamp": report["timestamp"],
            }
            fb._put(f"mamba3/state/inspections/{task}", compact)
            print(f"  Inspection pushed to Firebase", flush=True)
        except Exception as e:
            print(f"  Firebase push failed: {e}", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="parity")
    parser.add_argument("--n-examples", type=int, default=5)
    args = parser.parse_args()

    report = inspect_model(args.task, args.n_examples)
    if report:
        print(f"\nInspection: {report['task']}")
        print(f"  Accuracy: {report['accuracy']:.0%}")
        for ex in report["examples"]:
            print(f"\n  Input: {ex['input']} → expected={ex['expected']} "
                  f"predicted={ex['predicted']} {'✓' if ex['correct'] else '✗'}")
            print(f"  Max state change at '{ex['max_delta_token']}' "
                  f"(pos {ex['max_delta_pos']}, delta={ex['max_delta_value']:.1f})")
            norms = " ".join(f"{t['state_norm']:.0f}" for t in ex["per_token"])
            print(f"  State norms: {norms}")
        save_and_push(args.task, report)
    else:
        print("No checkpoint found or inspection failed.")
