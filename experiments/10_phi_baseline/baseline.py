"""Phi baseline probe — what can phi do out-of-the-box?

Loads a phi checkpoint (default: phi-3-mini-4k-instruct), runs a battery
of small probes in five categories — identity, greeting, counting,
multilingual, reasoning — and writes the prompts + completions to
baseline.md as a permanent record.

Run on the Mac:

    .venv/bin/python 10_phi_baseline/baseline.py
    .venv/bin/python 10_phi_baseline/baseline.py --model microsoft/phi-2
    .venv/bin/python 10_phi_baseline/baseline.py --device cpu  # if MPS OOMs

The first run downloads the model (~7.6 GB for phi-3-mini, ~5.5 GB for
phi-2). Subsequent runs are cache hits.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


HERE = Path(__file__).resolve().parent


PROBES: list[tuple[str, str, str]] = [
    # (category, label, prompt)
    ("identity", "who-are-you",       "Who are you? Answer in one sentence."),
    ("identity", "what-model",        "What language model are you and who made you?"),

    ("greeting", "en",                "Hello! How are you today?"),
    ("greeting", "es",                "¡Hola! ¿Cómo estás hoy?"),
    ("greeting", "fr",                "Bonjour! Comment vas-tu aujourd'hui?"),
    ("greeting", "de",                "Hallo! Wie geht es dir heute?"),
    ("greeting", "pt",                "Olá! Como vais hoje?"),
    ("greeting", "it",                "Ciao! Come stai oggi?"),
    ("greeting", "ja",                "こんにちは!お元気ですか?"),
    ("greeting", "zh",                "你好!你今天怎么样?"),
    ("greeting", "ar",                "مرحبا! كيف حالك اليوم؟"),
    ("greeting", "ru",                "Привет! Как дела сегодня?"),

    ("counting", "to-10",             "Count from 1 to 10."),
    ("counting", "to-30",             "Count from 1 to 30."),
    ("counting", "even-to-20",        "List the even numbers from 2 to 20."),
    ("counting", "primes-under-30",   "List all prime numbers less than 30."),
    ("counting", "fib-10",            "List the first 10 Fibonacci numbers."),
    ("counting", "spanish-1-10",      "Cuenta del 1 al 10 en español."),
    ("counting", "japanese-1-10",     "1から10まで日本語で数えてください。"),

    ("multi",    "translate-en-es",   "Translate to Spanish: 'The cat sat on the mat.'"),
    ("multi",    "translate-en-ja",   "Translate to Japanese: 'I want to learn programming.'"),
    ("multi",    "write-haiku",       "Write a haiku about autumn rain."),
    ("multi",    "write-spanish",     "Escribe una frase original en español sobre el café."),

    ("reasoning","arith-1",           "What is 17 * 24? Show your work."),
    ("reasoning","arith-2",           "If a train leaves at 3:15 PM and travels for 2 hours 50 minutes, when does it arrive?"),
    ("reasoning","logic",             "Alice is taller than Bob. Bob is taller than Carol. Who is the shortest?"),
    ("reasoning","hanoi",             "How many moves are needed to solve Tower of Hanoi with 5 disks?"),
]


def build_prompt(tokenizer, user_text: str) -> str:
    """Use the model's chat template if it has one, else raw text."""
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return user_text


def generate(model, tokenizer, prompt: str, max_new_tokens: int, device: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/Phi-3-mini-4k-instruct",
                    help="HF model id. Try microsoft/phi-2 (5.5GB) for a "
                         "lighter option.")
    ap.add_argument("--device", default=None,
                    help="cuda / mps / cpu. Auto-detects if omitted.")
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--out", default=str(HERE / "baseline.md"))
    ap.add_argument("--out-json", default=str(HERE / "baseline.json"))
    args = ap.parse_args()

    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32

    print(f"[load] {args.model} on {device} ({dtype})")
    t0 = time.time()
    # No trust_remote_code: transformers 5.x ships a built-in Phi3 impl that
    # handles the new `rope_scaling.rope_type` config. The vendored
    # modeling_phi3.py on the Hub still expects the old `["type"]` key and
    # blows up with KeyError under transformers >= 5.0.
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[load] done in {time.time() - t0:.1f}s — {n_params/1e9:.2f}B params")

    results: list[dict] = []
    for i, (cat, label, prompt) in enumerate(PROBES, 1):
        full_prompt = build_prompt(tokenizer, prompt)
        t0 = time.time()
        try:
            completion = generate(model, tokenizer, full_prompt,
                                  args.max_new_tokens, device)
            err = None
        except Exception as e:
            completion = ""
            err = repr(e)
        dt = time.time() - t0
        print(f"[{i:2d}/{len(PROBES)}] {cat:9s} {label:18s} ({dt:.1f}s) "
              f"→ {completion[:70]!r}{'…' if len(completion) > 70 else ''}")
        results.append({
            "category": cat,
            "label": label,
            "prompt": prompt,
            "completion": completion,
            "elapsed_s": round(dt, 2),
            "error": err,
        })

    out_md = Path(args.out)
    lines = [
        f"# Phi baseline — `{args.model}`",
        "",
        f"- params: {n_params/1e9:.2f}B",
        f"- device: {device} ({dtype})",
        f"- decoding: greedy, max_new_tokens={args.max_new_tokens}",
        f"- chat template: {'yes' if tokenizer.chat_template else 'no'}",
        "",
    ]
    by_cat: dict[str, list[dict]] = {}
    for r in results:
        by_cat.setdefault(r["category"], []).append(r)
    for cat, rs in by_cat.items():
        lines.append(f"## {cat}")
        lines.append("")
        for r in rs:
            lines.append(f"### {r['label']}  _({r['elapsed_s']}s)_")
            lines.append("")
            lines.append(f"**Prompt:** {r['prompt']}")
            lines.append("")
            if r["error"]:
                lines.append(f"**Error:** `{r['error']}`")
            else:
                lines.append("**Completion:**")
                lines.append("")
                lines.append("```")
                lines.append(r["completion"])
                lines.append("```")
            lines.append("")
    out_md.write_text("\n".join(lines))
    Path(args.out_json).write_text(json.dumps({
        "model": args.model,
        "n_params": n_params,
        "device": device,
        "dtype": str(dtype),
        "results": results,
    }, indent=2, ensure_ascii=False))
    print(f"[done] wrote {out_md} and {args.out_json}")


if __name__ == "__main__":
    main()
