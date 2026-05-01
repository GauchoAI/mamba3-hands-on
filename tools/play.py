#!/usr/bin/env python3
"""
🐀 Ratatouille — Interactive CLI for the Mamba-3 kernel.

Play with the trained model. Feed it inputs, see what it thinks.

Usage:
    python play.py                              # auto-find best checkpoint
    python play.py --checkpoint path/to/ckpt.pt # specific checkpoint
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import json
import time
import torch
from pathlib import Path
from progressive_model import ProgressiveModel, ByteTokenizer, VOCAB_SIZE, PAD, BOS, EOS, SEP


# ── Colors ──────────────────────────────────────────────────────────

class C:
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    RED = "\033[91m"
    RESET = "\033[0m"


def colored(text, color):
    return f"{color}{text}{C.RESET}"


# ── Model loading ───────────────────────────────────────────────────

def find_checkpoint():
    """Find the best checkpoint in checkpoints/ or runs/."""
    candidates = list(Path("checkpoints").glob("best_parity*.pt"))
    candidates += list(Path("checkpoints").glob("progressive_best.pt"))
    candidates += list(Path("runs").glob("*/checkpoint.pt"))

    if not candidates:
        return None

    # Prefer best_parity if it exists
    for c in candidates:
        if "best_parity" in c.name:
            return c

    return candidates[0]


def load_model(ckpt_path):
    """Load model from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})

    model = ProgressiveModel(
        d_model=cfg.get("d_model", 64),
        d_state=cfg.get("d_state", 16),
        expand=2,
        headdim=cfg.get("headdim", 16),
    )
    for _ in range(cfg.get("n_kernel_layers", 1)):
        model.add_kernel_layer()

    model.load_state_dict(ckpt["model"])
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    return model, cfg, n_params, ckpt


# ── Inference ───────────────────────────────────────────────────────

def predict(model, input_text, device="cpu"):
    """Run inference and return prediction + timing."""
    tok = ByteTokenizer()
    input_bytes = list(input_text.encode("utf-8"))
    tokens = [BOS] + input_bytes + [SEP]

    t = torch.tensor([tokens], dtype=torch.long, device=device)

    # Time the forward pass
    if device == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(t)
    if device == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - t0

    # Predict bytes after SEP
    sep_pos = len(tokens) - 1
    predicted_bytes = []
    confidences = []
    for i in range(10):  # max 10 output bytes
        pos = sep_pos + i
        if pos >= logits.shape[1]:
            break
        probs = torch.softmax(logits[0, pos], dim=-1)
        pred_byte = probs.argmax().item()
        confidence = probs[pred_byte].item()

        if pred_byte == EOS or pred_byte == PAD:
            break
        predicted_bytes.append(pred_byte)
        confidences.append(confidence)

    # Decode
    output_text = bytes(b for b in predicted_bytes if b < 256).decode("utf-8", errors="replace")

    # Tokens per second
    n_tokens = logits.shape[1]
    tps = n_tokens / elapsed if elapsed > 0 else 0

    return output_text, confidences, elapsed, tps, n_tokens


# ── Task helpers ────────────────────────────────────────────────────

TASKS = {
    "parity": {
        "desc": "Count 1s in a binary sequence. Even → S, Odd → D",
        "examples": ["0 1 1 0", "1 0 1", "0 0 0 1 1", "1 1 1 1"],
        "hint": "Enter a sequence of 0s and 1s separated by spaces",
    },
    "same_different": {
        "desc": "Are two numbers the same? Same → S, Different → D",
        "examples": ["3 3", "2 5", "0 0", "7 3"],
        "hint": "Enter two numbers separated by a space",
    },
    "mirror": {
        "desc": "Is the sequence a palindrome? Yes → M, No → N",
        "examples": ["1 2 3 2 1", "1 2 3 4", "5 5", "3 1 1 3"],
        "hint": "Enter a sequence of numbers separated by spaces",
    },
    "arithmetic": {
        "desc": "What comes next in the arithmetic sequence?",
        "examples": ["2 5 8 11 ?", "10 8 6 4 ?", "1 3 5 7 ?"],
        "hint": "Enter numbers with ? at the end",
    },
    "logic": {
        "desc": "Evaluate a logic gate",
        "examples": ["AND 1 0", "OR 0 1", "XOR 1 1", "NOT 0"],
        "hint": "Enter GATE followed by values (0 or 1)",
    },
    "free": {
        "desc": "Type anything — see what the model makes of it",
        "examples": [],
        "hint": "Enter any text",
    },
}


# ── CLI ─────────────────────────────────────────────────────────────

def print_banner(cfg, n_params, ckpt_path):
    print()
    print(colored("  ╔══════════════════════════════════════════╗", C.CYAN))
    print(colored("  ║", C.CYAN) + colored("  🐀 Ratatouille", C.BOLD + C.YELLOW) +
          colored(" — Mamba-3 Kernel CLI   ", C.DIM) + colored("║", C.CYAN))
    print(colored("  ╚══════════════════════════════════════════╝", C.CYAN))
    print()
    print(f"  {colored('Model:', C.DIM)} d={cfg.get('d_model','?')}, "
          f"L={cfg.get('n_kernel_layers','?')}, "
          f"{n_params:,} params")
    print(f"  {colored('Strategy:', C.DIM)} wd={cfg.get('weight_decay','?')}, "
          f"loss={cfg.get('loss_fn','stable_ce')}")
    print(f"  {colored('Checkpoint:', C.DIM)} {ckpt_path}")
    print()


def print_tasks():
    print(colored("  Available tasks:", C.BOLD))
    print()
    for key, task in TASKS.items():
        print(f"    {colored(key, C.GREEN):30s} {task['desc']}")
    print()
    print(f"  {colored('Commands:', C.BOLD)}")
    print(f"    {colored('bench', C.YELLOW)}     Run benchmark (tokens/sec)")
    print(f"    {colored('test', C.YELLOW)}      Run test suite on all tasks")
    print(f"    {colored('quit', C.YELLOW)}      Exit")
    print()


def run_benchmark(model, device):
    """Benchmark tokens/sec."""
    from generators.level0_patterns import gen_parity
    tok = ByteTokenizer()
    print(colored("  Running benchmark...", C.DIM))

    times = []
    for _ in range(50):
        ex = gen_parity()
        _, _, elapsed, tps, n_tok = predict(model, ex["input"], device)
        times.append((elapsed, tps, n_tok))

    avg_ms = sum(t[0] for t in times) / len(times) * 1000
    avg_tps = sum(t[1] for t in times) / len(times)
    print(f"  {colored('Benchmark:', C.BOLD)} {avg_ms:.1f}ms/inference, "
          f"{avg_tps:.0f} tokens/sec ({len(times)} runs)")
    print()


def run_test(model, device):
    """Test on generated examples."""
    from generators.level0_patterns import (
        gen_parity, gen_same_different, gen_mirror_detection,
        gen_arithmetic_next, gen_logic_gate,
    )
    generators = {
        "parity": gen_parity,
        "same_different": gen_same_different,
        "mirror": gen_mirror_detection,
        "arithmetic": gen_arithmetic_next,
        "logic": gen_logic_gate,
    }

    print(colored("  Test suite:", C.BOLD))
    print()

    for task_name, gen_fn in generators.items():
        correct = 0
        total = 20
        for _ in range(total):
            try:
                ex = gen_fn()
                output, _, _, _, _ = predict(model, ex["input"], device)
                if output.strip() == ex["output"].strip():
                    correct += 1
            except Exception:
                pass

        pct = correct / total
        bar = "█" * int(pct * 20) + "░" * (20 - int(pct * 20))
        color = C.GREEN if pct >= 0.8 else (C.YELLOW if pct >= 0.5 else C.RED)
        print(f"    {task_name:20s} {colored(bar, color)} {correct}/{total} ({pct:.0%})")

    print()


def interactive(model, device):
    """Main interactive loop."""
    current_task = "parity"

    while True:
        task = TASKS[current_task]
        prompt = (f"  {colored(current_task, C.GREEN + C.BOLD)} "
                  f"{colored('>', C.DIM)} ")

        try:
            user_input = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue

        # Commands
        if user_input.lower() == "quit" or user_input.lower() == "q":
            break
        elif user_input.lower() == "help" or user_input.lower() == "h":
            print_tasks()
            continue
        elif user_input.lower() == "bench":
            run_benchmark(model, device)
            continue
        elif user_input.lower() == "test":
            run_test(model, device)
            continue
        elif user_input.lower() in TASKS:
            current_task = user_input.lower()
            task = TASKS[current_task]
            print(f"  {colored('Switched to:', C.DIM)} {task['desc']}")
            if task["examples"]:
                print(f"  {colored('Examples:', C.DIM)} {', '.join(task['examples'][:3])}")
            print(f"  {colored('Hint:', C.DIM)} {task['hint']}")
            print()
            continue

        # Inference
        output, confidences, elapsed, tps, n_tok = predict(model, user_input, device)

        # Display
        conf_str = ""
        if confidences:
            avg_conf = sum(confidences) / len(confidences)
            conf_str = f" {colored(f'({avg_conf:.0%} confident)', C.DIM)}"

        print(f"  {colored('→', C.YELLOW)} {colored(output, C.BOLD + C.MAGENTA)}"
              f"{conf_str}"
              f"  {colored(f'{elapsed*1000:.1f}ms, {tps:.0f} tok/s', C.DIM)}")
        print()


# ── Main ────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="🐀 Ratatouille — Mamba-3 Kernel CLI")
    parser.add_argument("--checkpoint", "-c", type=str, default=None)
    parser.add_argument("--device", "-d", type=str, default=None)
    args = parser.parse_args()

    # Find checkpoint
    ckpt_path = args.checkpoint
    if not ckpt_path:
        found = find_checkpoint()
        if not found:
            print("No checkpoint found. Download one from the H100:")
            print("  scp -P 32783 root@ssh2.vast.ai:/root/mamba3-hands-on/runs/exp_059/checkpoint.pt checkpoints/best_parity.pt")
            return
        ckpt_path = str(found)

    # Device
    if args.device:
        device = args.device
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Load
    model, cfg, n_params, ckpt = load_model(ckpt_path)
    model = model.to(device)

    # Banner
    print_banner(cfg, n_params, ckpt_path)
    print(f"  {colored('Device:', C.DIM)} {device}")
    print()
    print_tasks()

    # Show examples for default task
    task = TASKS["parity"]
    print(f"  {colored('Current task:', C.DIM)} parity — {task['desc']}")
    print(f"  {colored('Try:', C.DIM)} {', '.join(task['examples'][:3])}")
    print(f"  {colored('Type a task name to switch, or enter input directly.', C.DIM)}")
    print()

    interactive(model, device)

    print(colored("  👋 Adiós!", C.YELLOW))
    print()


if __name__ == "__main__":
    main()
