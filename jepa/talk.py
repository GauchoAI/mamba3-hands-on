"""Interactive REPL over a trained JEPA-Cortex checkpoint.

W5.1 of the plan. Loads a light or best ckpt, enables hard-gates inference
on any CounterPrimitive present (per feedback_hard_gates_at_inference.md
this is free OOD accuracy), and reads stdin.

Usage:
    python talk.py --ckpt checkpoints/jepa_cortex/gpu0/best.pt
    > Cuenta de 1 a 20, uno por linea.
    1
    2
    ...
    > what is 7 + 5?
    ...
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import torch

from cortex_counting import CortexLM, CortexLMConfig, CounterPrimitive


def load_model(ckpt_path: str, device: str) -> CortexLM:
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = payload["config"]
    keys = {f for f in CortexLMConfig.__dataclass_fields__}
    cfg = CortexLMConfig(**{k: cfg_dict[k] for k in cfg_dict if k in keys})
    model = CortexLM(cfg).to(device)
    own = set(model.state_dict().keys())
    sd = {k: v for k, v in payload["model"].items() if k in own}
    missing = own - set(sd.keys())
    if missing:
        print(f"[warn] missing {len(missing)} keys in ckpt; "
              f"first few: {sorted(missing)[:5]}", file=sys.stderr)
    model.load_state_dict(sd, strict=False)
    model.eval()
    # Toggle hard gates on any CounterPrimitive — costless at eval, eliminates
    # OOD slippage from soft sigmoid drift.
    for p in model.primitives:
        if isinstance(p, CounterPrimitive):
            p.hard_gates_inference = True
    return model


def stream_generate(model: CortexLM, prompt: bytes,
                    max_new: int, temperature: float,
                    device: str) -> bytes:
    """Greedy by default; if temperature>0 we sample. Streams to stdout."""
    toks = torch.tensor([list(prompt)], dtype=torch.long, device=device)
    out = bytearray()
    with torch.no_grad():
        for _ in range(max_new):
            logits = model(toks)[:, -1] / max(temperature, 1e-6)
            if temperature <= 0:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = torch.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)
            byte = int(nxt.item())
            out.append(byte)
            sys.stdout.buffer.write(bytes([byte]))
            sys.stdout.buffer.flush()
            toks = torch.cat([toks, nxt], dim=1)
            # Stop on double-newline as a soft EOS — the teacher uses
            # newline-terminated lines, so two in a row usually means done.
            if len(out) >= 2 and out[-1] == 10 and out[-2] == 10:
                break
    return bytes(out)


def repl(model: CortexLM, max_new: int, temperature: float, device: str):
    print("# JEPA-Cortex REPL — Ctrl-D / Ctrl-C to exit\n", flush=True)
    while True:
        try:
            line = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line.strip():
            continue
        prompt = (line + "\n").encode("utf-8")
        stream_generate(model, prompt, max_new, temperature, device)
        print()  # newline after stream


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0 = greedy (default); >0 = sample")
    ap.add_argument("--prompt", default="",
                    help="One-shot prompt; if given, prints and exits")
    args = ap.parse_args()

    model = load_model(args.ckpt, args.device)
    print(f"# loaded {args.ckpt}", flush=True)

    if args.prompt:
        prompt = (args.prompt + "\n").encode("utf-8")
        stream_generate(model, prompt, args.max_new, args.temperature,
                        args.device)
        print()
        return
    repl(model, args.max_new, args.temperature, args.device)


if __name__ == "__main__":
    main()
