"""Cortex composition demo: bilingual LM + attached counter primitive.

Side-by-side comparison:
  baseline LM (no counter) vs cortex (LM + 1k-param CounterPrimitive),
  on counting prompts at training-distribution N and OOD N.

The bilingual LM was trained on data/bilingual.txt (Tatoeba en-es +
~5% unary cortex mixin at N ∈ [1, 30]). Both models have 472,960
parameters of LM. The cortex model adds 1,028 parameters (counter
adapters: inc_proj, reset_proj, read_proj) — 0.22% extra.

Key claim being tested:
  If we attach a 1k-param plugin to a FROZEN 473k-param language-
  trained LM and fine-tune ONLY the plugin, does the plugin extend
  the LM's counting capability past the training distribution?
  Yes -> existence proof for train-free composition.

Run after counter-attach training:
    python demo_cortex.py
    python demo_cortex.py --lm-ckpt path --counter-ckpt path
"""
from __future__ import annotations
import argparse
import os
import sys
import torch
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cortex_counting import (
    CortexLM, CortexLMConfig, CounterPrimitive, parse_count_output,
)


def load_baseline(path: str, device: str) -> CortexLM:
    """Load the bilingual LM with no primitives — pure baseline."""
    sd = torch.load(path, map_location=device, weights_only=False)
    m = CortexLM(sd["cfg"]).to(device).eval()
    m.load_state_dict(sd["model"])
    return m


def load_cortex(path: str, device: str) -> CortexLM:
    """Load the counter-attached bilingual LM (frozen LM + trained counter)."""
    sd = torch.load(path, map_location=device, weights_only=False)
    cfg = sd["cfg"]
    counter = CounterPrimitive(
        d_model=cfg.d_model, layer=0, n_counters=2,
        readout="unbounded", injection_scale=10.0,
    )
    m = CortexLM(cfg, primitives=[counter]).to(device).eval()
    m.load_state_dict(sd["model"])
    if m.counter is not None:
        m.counter.hard_gates_inference = True   # the byte-perfect mode
    return m


@torch.no_grad()
def count_eval(model: CortexLM, n: int) -> tuple[bool, int | None, str]:
    """Prompt with N stars + ':', return (ok, parsed_count, head_of_output)."""
    prompt = "*" * n + ":"
    pb = list(prompt.encode("utf-8"))
    gen = model.generate_greedy(pb, max_new=n + 4, max_ctx=8192)
    text = bytes(gen).decode("utf-8", errors="replace")
    parsed = parse_count_output(text)
    ok = (parsed == n)
    return ok, parsed, text[:50]


@torch.no_grad()
def language_probe(model: CortexLM, prompt: str, max_new: int = 80) -> str:
    pb = list(prompt.encode("utf-8"))
    gen = model.generate_greedy(pb, max_new=max_new, max_ctx=128)
    text = bytes(gen).decode("utf-8", errors="replace")
    return text[len(prompt):]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lm-ckpt", default="checkpoints/lm/step_FINAL.pt")
    ap.add_argument("--counter-ckpt", default="checkpoints/lm_counter/step_FINAL.pt")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")

    if not Path(args.counter_ckpt).exists():
        print(f"\nNo counter checkpoint at {args.counter_ckpt}.")
        print("Run train_counter_attach.py first; this demo is the eval step.")
        return

    base = load_baseline(args.lm_ckpt, device)
    cortex = load_cortex(args.counter_ckpt, device)

    print("\n" + "=" * 72)
    print("Cortex composition demo")
    print("=" * 72)
    print(f"  baseline:  bilingual LM, no primitives  ({sum(p.numel() for p in base.parameters()):,} params)")
    print(f"  cortex:    bilingual LM + counter        ({sum(p.numel() for p in cortex.parameters()):,} params)")
    n_extra = sum(p.numel() for p in cortex.parameters()) - sum(p.numel() for p in base.parameters())
    print(f"  difference: {n_extra:,} params (the counter adapter)")
    print(f"  trained on: N ∈ [1, 30] only")
    print(f"  cortex hard_gates_inference: {cortex.counter.hard_gates_inference}")

    # ============ Counting eval ============
    print("\n" + "-" * 72)
    print("Part 1 — counting at training and OOD lengths")
    print("-" * 72)
    print(f"{'N':>5}  {'baseline':>20}  {'cortex+counter':>20}  {'OOD?':>8}")
    table_rows = []
    for n in [3, 10, 30, 50, 100, 200, 500]:
        b_ok, b_p, b_t = count_eval(base, n)
        c_ok, c_p, c_t = count_eval(cortex, n)
        b_str = "OK ✓" if b_ok else f"FAIL→{b_p}"
        c_str = "OK ✓" if c_ok else f"FAIL→{c_p}"
        ood = "(OOD)" if n > 30 else ""
        row = f"{n:>5}  {b_str:>20}  {c_str:>20}  {ood:>8}"
        print(row)
        table_rows.append((n, b_ok, b_p, c_ok, c_p))

    # ============ Bilingual probes ============
    print("\n" + "-" * 72)
    print("Part 2 — bilingual probes (does the LM still talk?)")
    print("-" * 72)
    bilingual_prompts = [
        "The cat ", "El gato ",
        "¿Dónde ", "Where does ",
        "I have ", "Tengo ",
    ]
    for p in bilingual_prompts:
        b_text = language_probe(base, p, max_new=60).replace("\n", " ↵ ")[:70]
        c_text = language_probe(cortex, p, max_new=60).replace("\n", " ↵ ")[:70]
        print(f"  {p!r:>14}")
        print(f"     baseline → {b_text!r}")
        print(f"     cortex   → {c_text!r}")
        print()

    # ============ Summary ============
    n_baseline_ood_ok = sum(1 for n, b, _, _, _ in table_rows if n > 30 and b)
    n_cortex_ood_ok   = sum(1 for n, _, _, c, _ in table_rows if n > 30 and c)
    print("-" * 72)
    print("Summary:")
    print(f"  baseline  OOD lengths (N>30) byte-perfect: {n_baseline_ood_ok}/4")
    print(f"  cortex    OOD lengths (N>30) byte-perfect: {n_cortex_ood_ok}/4")
    print(f"  in-distribution (N≤30) for both should be 3/3")
    print()


if __name__ == "__main__":
    main()
