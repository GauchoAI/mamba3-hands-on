#!/usr/bin/env python3
"""Mamba-3 Inference Server — run specialist checkpoints for distillation.

This is our equivalent of llama-server: loads a trained specialist checkpoint
and serves inference via HTTP or direct Python API. Used for cross-node
distillation — a teacher trained on H100 can serve predictions to a student
training on Mac Mini.

Usage:
    # Direct inference (Python API)
    python export/mamba_inference.py --checkpoint checkpoints/specialists/parity.pt --eval

    # HTTP server (for cross-node distillation)
    python export/mamba_inference.py --checkpoint checkpoints/specialists/parity.pt --serve --port 8090

    # Batch inference on a dataset
    python export/mamba_inference.py --checkpoint checkpoints/specialists/parity.pt --batch data/test.jsonl
"""

import json
import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_specialist(checkpoint_path: str, device: str = "cpu"):
    """Load a specialist model from a .pt checkpoint."""
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

    print(f"Loaded: {task} ({n_params:,} params, {accuracy:.0%} accuracy)")
    print(f"  Config: d={config.get('d_model')}, L={config.get('n_kernel_layers')}, "
          f"dS={config.get('d_state')}, hd={config.get('headdim')}")
    print(f"  Device: {device}")

    return model, config, task


def predict(model, example: dict, device: str = "cpu"):
    """Run inference on a single example. Returns predicted output and logits."""
    import torch
    from progressive_model import ByteTokenizer, VOCAB_SIZE

    tok = ByteTokenizer()
    tokens, sep = tok.encode_curriculum(example)
    t = torch.tensor([tokens], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(t)  # (1, seq_len, vocab_size)

    # Get prediction at each output position (after SEP)
    output_logits = logits[0, sep:]  # (output_len, vocab)
    predicted_ids = output_logits.argmax(dim=-1).tolist()

    # Decode predicted bytes
    predicted = bytes(b for b in predicted_ids if b < 256).decode("utf-8", errors="replace").strip()

    return {
        "input": example["input"],
        "expected": example.get("output", ""),
        "predicted": predicted,
        "correct": predicted == example.get("output", ""),
        "logits": output_logits.cpu().tolist(),
    }


def evaluate(model, task: str, n_examples: int = 100, device: str = "cpu",
             problems_dir: str = "problems"):
    """Evaluate model on fresh examples from the problem generator."""
    from registry.problem_registry import ProblemRegistry

    registry = ProblemRegistry()
    registry.discover([problems_dir])
    gen_fn = registry.get_generator(task)

    correct = 0
    total = 0
    for _ in range(n_examples):
        example = gen_fn()
        result = predict(model, example, device)
        if result["correct"]:
            correct += 1
        total += 1

    accuracy = correct / max(total, 1)
    print(f"\nEvaluation: {correct}/{total} = {accuracy:.1%}")
    return accuracy


def serve(model, task: str, device: str = "cpu", port: int = 8090):
    """Start HTTP inference server for cross-node distillation."""
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import torch

    class InferenceHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path == "/predict":
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length))
                example = body.get("example", body)
                result = predict(model, example, device)
                # Don't send full logits over HTTP unless requested
                if not body.get("include_logits"):
                    result.pop("logits", None)
                self._respond(200, result)
            elif self.path == "/batch":
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length))
                examples = body.get("examples", [])
                results = [predict(model, ex, device) for ex in examples]
                for r in results:
                    r.pop("logits", None)
                self._respond(200, {"results": results})
            elif self.path == "/logits":
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length))
                example = body.get("example", body)
                result = predict(model, example, device)
                self._respond(200, result)  # includes logits
            else:
                self._respond(404, {"error": "Not found"})

        def do_GET(self):
            if self.path == "/health":
                self._respond(200, {"status": "ok", "task": task, "device": device})
            elif self.path == "/info":
                config = {}
                n_params = sum(p.numel() for p in model.parameters())
                self._respond(200, {
                    "task": task,
                    "params": n_params,
                    "device": device,
                })
            else:
                self._respond(404, {"error": "Not found"})

        def _respond(self, code, data):
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data, default=str).encode())

        def log_message(self, format, *args):
            pass  # suppress access logs

    server = HTTPServer(("0.0.0.0", port), InferenceHandler)
    print(f"\nServing {task} on http://0.0.0.0:{port}")
    print(f"  POST /predict  — single example inference")
    print(f"  POST /batch    — batch inference")
    print(f"  POST /logits   — inference with full logits (for distillation)")
    print(f"  GET  /health   — health check")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Mamba-3 inference server")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to specialist .pt checkpoint")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device for inference (cpu, cuda, mps)")
    parser.add_argument("--eval", action="store_true",
                       help="Evaluate on 100 fresh examples")
    parser.add_argument("--n-examples", type=int, default=100,
                       help="Number of examples for evaluation")
    parser.add_argument("--serve", action="store_true",
                       help="Start HTTP inference server")
    parser.add_argument("--port", type=int, default=8090,
                       help="HTTP server port")
    parser.add_argument("--batch", type=str, default=None,
                       help="Path to JSONL file for batch inference")
    parser.add_argument("--problems-dir", type=str, default="problems",
                       help="Problems directory for evaluation")
    args = parser.parse_args()

    model, config, task = load_specialist(args.checkpoint, args.device)

    if args.eval:
        evaluate(model, task, args.n_examples, args.device, args.problems_dir)

    if args.batch:
        results = []
        with open(args.batch) as f:
            for line in f:
                example = json.loads(line.strip())
                result = predict(model, example, args.device)
                result.pop("logits", None)
                results.append(result)
                status = "✓" if result["correct"] else "✗"
                print(f"  {status} {result['input']} → {result['predicted']} (expected {result['expected']})")
        correct = sum(1 for r in results if r["correct"])
        print(f"\n{correct}/{len(results)} correct ({correct/max(len(results),1):.0%})")

    if args.serve:
        serve(model, task, args.device, args.port)

    if not args.eval and not args.serve and not args.batch:
        # Interactive mode
        print("\nInteractive mode. Enter inputs (Ctrl+C to quit):")
        from registry.problem_registry import ProblemRegistry
        registry = ProblemRegistry()
        registry.discover([args.problems_dir])
        gen_fn = registry.get_generator(task)
        while True:
            try:
                example = gen_fn()
                result = predict(model, example, args.device)
                status = "✓" if result["correct"] else "✗"
                print(f"  {status} {example['input']} → {result['predicted']} (expected {example['output']})")
            except KeyboardInterrupt:
                break


if __name__ == "__main__":
    main()
