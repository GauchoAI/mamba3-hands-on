#!/usr/bin/env python3
"""Mamba-3 Inference Engine — fast inference for specialists.

Our equivalent of llama.cpp for Mamba-3 models. Optimized for:
- Fast single inference (teacher serving during distillation)
- Batched inference (evaluation, dataset scoring)
- HTTP server (cross-node distillation)
- torch.compile for maximum throughput

Usage:
    # Evaluate a checkpoint
    python export/mamba_inference.py --checkpoint checkpoints/specialists/parity.pt --eval

    # Benchmark throughput
    python export/mamba_inference.py --checkpoint checkpoints/specialists/parity.pt --bench

    # HTTP server for cross-node distillation
    python export/mamba_inference.py --checkpoint checkpoints/specialists/parity.pt --serve --port 8090

    # Batch inference
    python export/mamba_inference.py --checkpoint checkpoints/specialists/parity.pt --batch data/test.jsonl
"""

import json
import sys
import os
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_specialist(checkpoint_path: str, device: str = "cpu", compile: bool = False):
    """Load a specialist model from a .pt checkpoint.

    Args:
        compile: Use torch.compile for faster inference (warmup cost, then ~2x faster)
    """
    import torch
    from progressive_model import ProgressiveModel

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    task = ckpt.get("task", "unknown")

    model = ProgressiveModel(
        d_model=config.get("d_model", 64),
        d_state=config.get("d_state", 16),
        expand=2,
        headdim=config.get("headdim", 16),
    ).to(device)

    for _ in range(config.get("n_kernel_layers", 1)):
        model.add_kernel_layer()

    model.load_state_dict(ckpt["model"])
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    accuracy = ckpt.get("accuracy", 0)

    # Compile for speed
    if compile:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print(f"  torch.compile enabled (reduce-overhead)")
        except Exception as e:
            print(f"  torch.compile failed: {e} (falling back to eager)")

    print(f"Loaded: {task} ({n_params:,} params, {accuracy:.0%} accuracy)")
    print(f"  Config: d={config.get('d_model')}, L={config.get('n_kernel_layers')}, "
          f"dS={config.get('d_state')}, hd={config.get('headdim')}")
    print(f"  Device: {device}")

    return model, config, task


def predict(model, example: dict, device: str = "cpu"):
    """Run inference on a single example. Returns predicted output and logits."""
    import torch
    from progressive_model import ByteTokenizer

    tok = ByteTokenizer()
    tokens, sep = tok.encode_curriculum(example)
    t = torch.tensor([tokens], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(t)

    output_logits = logits[0, sep:]
    predicted_ids = output_logits.argmax(dim=-1).tolist()
    predicted = bytes(b for b in predicted_ids if b < 256).decode("utf-8", errors="replace").strip()

    return {
        "input": example["input"],
        "expected": example.get("output", ""),
        "predicted": predicted,
        "correct": predicted == example.get("output", ""),
        "logits": output_logits.cpu().tolist(),
    }


def predict_batch(model, examples: list, device: str = "cpu"):
    """Run batched inference. Much faster than single predict() calls.

    Pads all examples to the same length and runs a single forward pass.
    """
    import torch
    from progressive_model import ByteTokenizer, PAD

    tok = ByteTokenizer()
    encoded = []
    for ex in examples:
        tokens, sep = tok.encode_curriculum(ex)
        encoded.append((tokens, sep, ex))

    # Pad to max length
    max_len = max(len(t[0]) for t in encoded)
    batch_tokens = torch.full((len(encoded), max_len), PAD, dtype=torch.long, device=device)
    seps = []
    for i, (tokens, sep, _) in enumerate(encoded):
        batch_tokens[i, :len(tokens)] = torch.tensor(tokens)
        seps.append(sep)

    # Single forward pass
    with torch.no_grad():
        all_logits = model(batch_tokens)  # (B, L, V)

    # Extract predictions
    results = []
    for i, (tokens, sep, ex) in enumerate(encoded):
        output_logits = all_logits[i, sep:len(tokens)]
        predicted_ids = output_logits.argmax(dim=-1).tolist()
        predicted = bytes(b for b in predicted_ids if b < 256).decode("utf-8", errors="replace").strip()
        results.append({
            "input": ex["input"],
            "expected": ex.get("output", ""),
            "predicted": predicted,
            "correct": predicted == ex.get("output", ""),
        })

    return results


def evaluate(model, task: str, n_examples: int = 100, device: str = "cpu",
             problems_dir: str = "problems", batch_size: int = 64):
    """Evaluate model on fresh examples using batched inference."""
    from registry.problem_registry import ProblemRegistry

    registry = ProblemRegistry()
    registry.discover([problems_dir])
    gen_fn = registry.get_generator(task)

    correct = 0
    total = 0
    t0 = time.time()

    # Process in batches
    remaining = n_examples
    while remaining > 0:
        bs = min(batch_size, remaining)
        examples = [gen_fn() for _ in range(bs)]
        results = predict_batch(model, examples, device)
        for r in results:
            if r["correct"]:
                correct += 1
            total += 1
        remaining -= bs

    elapsed = time.time() - t0
    accuracy = correct / max(total, 1)
    throughput = total / elapsed

    print(f"\nEvaluation: {correct}/{total} = {accuracy:.1%}")
    print(f"  Time: {elapsed:.2f}s ({throughput:.0f} examples/sec)")
    return accuracy


def benchmark(model, task: str, device: str = "cpu", problems_dir: str = "problems",
              n_warmup: int = 10, n_bench: int = 100):
    """Benchmark inference throughput."""
    import torch
    from registry.problem_registry import ProblemRegistry

    registry = ProblemRegistry()
    registry.discover([problems_dir])
    gen_fn = registry.get_generator(task)

    # Generate fixed examples
    examples = [gen_fn() for _ in range(n_bench)]

    # Warmup (especially important for torch.compile)
    print(f"Warming up ({n_warmup} examples)...")
    for ex in examples[:n_warmup]:
        predict(model, ex, device)

    # Single inference benchmark
    print(f"Benchmarking single inference ({n_bench} examples)...")
    t0 = time.time()
    for ex in examples:
        predict(model, ex, device)
    single_elapsed = time.time() - t0
    single_throughput = n_bench / single_elapsed

    # Batched inference benchmark
    batch_sizes = [16, 32, 64, 128]
    print(f"\nBenchmarking batched inference...")
    for bs in batch_sizes:
        t0 = time.time()
        for i in range(0, n_bench, bs):
            batch = examples[i:i+bs]
            predict_batch(model, batch, device)
        batch_elapsed = time.time() - t0
        batch_throughput = n_bench / batch_elapsed
        speedup = batch_throughput / single_throughput
        print(f"  batch={bs:>4d}: {batch_throughput:>6.0f} ex/s ({speedup:.1f}x vs single)")

    print(f"\nSingle: {single_throughput:.0f} ex/s ({single_elapsed/n_bench*1000:.1f}ms/ex)")


def serve(model, task: str, device: str = "cpu", port: int = 8090):
    """Start HTTP inference server for cross-node distillation."""
    from http.server import HTTPServer, BaseHTTPRequestHandler

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))

            if self.path == "/predict":
                result = predict(model, body.get("example", body), device)
                if not body.get("include_logits"):
                    result.pop("logits", None)
                self._json(200, result)
            elif self.path == "/batch":
                examples = body.get("examples", [])
                results = predict_batch(model, examples, device)
                self._json(200, {"results": results})
            elif self.path == "/logits":
                result = predict(model, body.get("example", body), device)
                self._json(200, result)
            else:
                self._json(404, {"error": "Not found"})

        def do_GET(self):
            if self.path == "/health":
                self._json(200, {"status": "ok", "task": task, "device": device})
            elif self.path == "/info":
                n = sum(p.numel() for p in model.parameters())
                self._json(200, {"task": task, "params": n, "device": device})
            else:
                self._json(404, {"error": "Not found"})

        def _json(self, code, data):
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data, default=str).encode())

        def log_message(self, format, *args):
            pass

    server = HTTPServer(("0.0.0.0", port), Handler)
    print(f"\nServing {task} on http://0.0.0.0:{port}")
    print(f"  POST /predict  — single inference")
    print(f"  POST /batch    — batched inference")
    print(f"  POST /logits   — with full logits (distillation)")
    print(f"  GET  /health   — health check")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")


def main():
    parser = argparse.ArgumentParser(description="Mamba-3 inference engine")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for speed")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--n-examples", type=int, default=100)
    parser.add_argument("--bench", action="store_true", help="Benchmark throughput")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--batch", type=str, default=None, help="JSONL file")
    parser.add_argument("--problems-dir", type=str, default="problems")
    args = parser.parse_args()

    model, config, task = load_specialist(args.checkpoint, args.device, args.compile)

    if args.bench:
        benchmark(model, task, args.device, args.problems_dir)
    elif args.eval:
        evaluate(model, task, args.n_examples, args.device, args.problems_dir)
    elif args.batch:
        with open(args.batch) as f:
            examples = [json.loads(line.strip()) for line in f if line.strip()]
        results = predict_batch(model, examples, args.device)
        correct = sum(1 for r in results if r["correct"])
        for r in results:
            s = "✓" if r["correct"] else "✗"
            print(f"  {s} {r['input']} → {r['predicted']} (expected {r['expected']})")
        print(f"\n{correct}/{len(results)} correct ({correct/max(len(results),1):.0%})")
    elif args.serve:
        serve(model, task, args.device, args.port)
    else:
        # Interactive
        from registry.problem_registry import ProblemRegistry
        registry = ProblemRegistry()
        registry.discover([args.problems_dir])
        gen_fn = registry.get_generator(task)
        print("\nInteractive mode (Ctrl+C to quit):")
        while True:
            try:
                ex = gen_fn()
                r = predict(model, ex, args.device)
                s = "✓" if r["correct"] else "✗"
                print(f"  {s} {ex['input']} → {r['predicted']} (expected {ex['output']})")
            except KeyboardInterrupt:
                break


if __name__ == "__main__":
    main()
