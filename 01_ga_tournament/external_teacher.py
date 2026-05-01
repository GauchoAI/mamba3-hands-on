"""
External Teacher: use a large language model (via llama.cpp) or an existing
specialist as a teacher for distillation.

Can be used as a mutation: config["teacher_model"] = "qwen-math-1.5b"
or config["teacher_model"] = "specialist:same_different"

The teacher produces output distributions (logits) for our task examples.
These logits are richer than hard labels — they encode the model's
uncertainty, its internal representations, its "dark knowledge."

Usage:
    # Start llama-server in the background first:
    # llama-server -m models/qwen-math-1.5b.gguf -ngl 99 --port 8081

    from external_teacher import ExternalTeacher
    teacher = ExternalTeacher("qwen-math-1.5b", port=8081)
    logits = teacher.get_logits("What is the parity of [1,0,1,1]? Answer E or O:")

    # Or use an existing specialist:
    teacher = ExternalTeacher("specialist:parity")
    logits = teacher.get_logits_for_task("parity", example)
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import json
import time
import subprocess
import urllib.request
import torch
from pathlib import Path


# ── Available models (GGUF format for llama.cpp) ──────────────────

MODELS = {
    "qwen-math-1.5b": {
        "url": "https://huggingface.co/Qwen/Qwen2.5-Math-1.5B-Instruct-GGUF/resolve/main/qwen2.5-math-1.5b-instruct-q4_k_m.gguf",
        "size_gb": 1.1,
        "description": "Qwen 2.5 Math 1.5B — specialized for mathematical reasoning",
    },
    "gemma-2b": {
        "url": "https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf",
        "size_gb": 1.5,
        "description": "Google Gemma 2 2B — efficient general model",
    },
    "mathstral-7b": {
        "url": "https://huggingface.co/bartowski/mathstral-7B-v0.1-GGUF/resolve/main/mathstral-7B-v0.1-Q4_K_M.gguf",
        "size_gb": 4.4,
        "description": "Mistral Math 7B — math-specialized",
    },
}


def download_model(name, models_dir="models"):
    """Download a GGUF model if not already present."""
    if name not in MODELS:
        print(f"Unknown model: {name}. Available: {list(MODELS.keys())}")
        return None

    info = MODELS[name]
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    filename = info["url"].split("/")[-1]
    filepath = models_path / filename

    if filepath.exists():
        print(f"  Model already downloaded: {filepath}")
        return str(filepath)

    print(f"  Downloading {name} ({info['size_gb']}GB)...")
    urllib.request.urlretrieve(info["url"], str(filepath))
    print(f"  Downloaded → {filepath}")
    return str(filepath)


def start_llama_server(model_path, port=8081, ngl=99):
    """Start llama-server in the background. Returns the process."""
    proc = subprocess.Popen(
        ["llama-server", "-m", model_path, "-ngl", str(ngl),
         "--port", str(port), "--log-disable"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    # Wait for server to be ready
    for _ in range(30):
        try:
            req = urllib.request.Request(f"http://localhost:{port}/health")
            urllib.request.urlopen(req, timeout=2)
            print(f"  llama-server ready on port {port}")
            return proc
        except Exception:
            time.sleep(1)
    print(f"  WARNING: llama-server may not be ready on port {port}")
    return proc


class ExternalTeacher:
    """Get output distributions from a large model or existing specialist."""

    def __init__(self, model_name, port=8081, device="cuda"):
        self.model_name = model_name
        self.port = port
        self.device = device
        self.server_proc = None
        self.specialist_model = None

        if model_name.startswith("specialist:"):
            self._load_specialist(model_name.replace("specialist:", ""))
        else:
            self._ensure_server(model_name)

    def _load_specialist(self, task):
        """Load an existing specialist checkpoint as teacher."""
        from progressive_model import ProgressiveModel

        ckpt_path = Path("checkpoints/specialists") / f"{task}.pt"
        if not ckpt_path.exists():
            print(f"  No specialist checkpoint for {task}")
            return

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        config = ckpt.get("config", {})

        model = ProgressiveModel(
            d_model=config.get("d_model", 64),
            d_state=config.get("d_state", 16),
            expand=2,
            headdim=config.get("headdim", 16),
        ).to(self.device)
        for _ in range(config.get("n_kernel_layers", 3)):
            model.add_kernel_layer()
        model.load_state_dict(ckpt["model"])
        model.eval()

        self.specialist_model = model
        acc = ckpt.get("accuracy", 0)
        print(f"  Loaded specialist teacher: {task} ({acc:.0%})", flush=True)

    def _ensure_server(self, model_name):
        """Make sure llama-server is running for this model."""
        # Check if already running
        try:
            req = urllib.request.Request(f"http://localhost:{self.port}/health")
            urllib.request.urlopen(req, timeout=2)
            return  # Already running
        except Exception:
            pass

        # Download and start
        model_path = download_model(model_name)
        if model_path:
            self.server_proc = start_llama_server(model_path, self.port)

    def get_completion(self, prompt, max_tokens=4):
        """Get completion from llama-server with logprobs."""
        data = json.dumps({
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": 0.0,
            "logprobs": True,
            "n_probs": 10,
        }).encode()

        req = urllib.request.Request(
            f"http://localhost:{self.port}/completion",
            data=data,
            headers={"Content-Type": "application/json"},
        )

        try:
            resp = urllib.request.urlopen(req, timeout=30)
            return json.loads(resp.read())
        except Exception as e:
            return None

    def get_teacher_logits(self, task, example, tok):
        """Get teacher logits for a task example.

        For LLM teachers: formats example as a prompt, gets completion logprobs.
        For specialist teachers: runs forward pass, returns raw logits.

        Returns list of (position, logits_tensor) pairs for output positions.
        """
        if self.specialist_model is not None:
            return self._specialist_logits(example, tok)
        else:
            return self._llm_logits(task, example)

    def _specialist_logits(self, example, tok):
        """Get logits from specialist model."""
        tokens, sep = tok.encode_curriculum(example)
        out_bytes = list(example["output"].encode("utf-8"))

        t = torch.tensor([tokens], dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = self.specialist_model(t)

        distributions = []
        for j in range(len(out_bytes)):
            p = sep + j
            if p < logits.shape[1]:
                distributions.append(logits[0, p])

        return distributions

    def _llm_logits(self, task, example):
        """Get logits from LLM via llama-server.

        Formats the task as a prompt the LLM can understand.
        Returns the logprobs for the answer tokens.
        """
        # Format task-specific prompts
        prompt = self._format_prompt(task, example)
        result = self.get_completion(prompt, max_tokens=2)

        if not result:
            return None

        # Extract the content and completion_probabilities
        return {
            "content": result.get("content", ""),
            "logprobs": result.get("completion_probabilities", []),
            "model": self.model_name,
        }

    def _format_prompt(self, task, example):
        """Format a task example as a prompt for an LLM."""
        inp = example["input"]

        prompts = {
            "parity": f"Count the 1s in this binary sequence: {inp}. Is the count even (E) or odd (O)? Answer with just E or O:",
            "binary_pattern_next": f"What comes next in this binary pattern? {inp} Answer with just the next number:",
            "same_different": f"Are these two sequences the same or different? {inp} Answer with just S or D:",
            "arithmetic_next": f"What is the next number in this arithmetic sequence? {inp} Answer with just the number:",
            "geometric_next": f"What is the next number in this geometric sequence? {inp} Answer with just the number:",
            "logic_gate": f"Evaluate this logic expression: {inp} Answer with just 0 or 1:",
            "modus_ponens": f"Given this logical statement: {inp} Is the conclusion valid? Answer M or N:",
        }

        return prompts.get(task, f"Solve: {inp} Answer:")

    def stop(self):
        """Stop the llama-server if we started it."""
        if self.server_proc:
            self.server_proc.terminate()
            self.server_proc = None


# ── Experiment: test models on our tasks ───────────────────────────

def run_experiment(model_name="qwen-math-1.5b", tasks=None, n_examples=20):
    """Test an external model on our tasks. See how accurate it is."""
    from specialist_trainer import load_generators
    from progressive_model import ByteTokenizer
    import specialist_trainer

    load_generators()
    tok = ByteTokenizer()

    if tasks is None:
        tasks = ["parity", "binary_pattern_next", "same_different",
                 "arithmetic_next", "logic_gate", "modus_ponens"]

    print(f"\n{'='*60}")
    print(f"External Teacher Experiment: {model_name}")
    print(f"{'='*60}\n")

    if model_name.startswith("specialist:"):
        teacher = ExternalTeacher(model_name, device="cuda")
    else:
        teacher = ExternalTeacher(model_name)

    results = {}
    for task in tasks:
        gen_fn = specialist_trainer.GENERATORS.get(task)
        if not gen_fn:
            continue

        correct = 0
        total = 0
        for _ in range(n_examples):
            ex = gen_fn()
            expected = ex["output"]

            if model_name.startswith("specialist:"):
                # Specialist: check if top logit matches
                logits = teacher.get_teacher_logits(task, ex, tok)
                if logits:
                    pred_byte = logits[0].argmax().item()
                    expected_byte = expected.encode("utf-8")[0]
                    if pred_byte == expected_byte:
                        correct += 1
                total += 1
            else:
                # LLM: check if completion matches
                result = teacher._llm_logits(task, ex)
                if result:
                    content = result["content"].strip().upper()
                    if content and content[0] == expected[0].upper():
                        correct += 1
                    total += 1

        acc = correct / max(total, 1)
        results[task] = acc
        print(f"  {task}: {acc:.0%} ({correct}/{total})")

    avg = sum(results.values()) / max(len(results), 1)
    print(f"\n  Average: {avg:.0%}")
    print(f"  Model: {model_name}")

    # Cache results in StateDB (idempotent)
    try:
        from state_db import StateDB
        db = StateDB("three_pop/training.db")
        for task, acc in results.items():
            db.set_teacher_score(model_name, task, acc, n_examples)
        print(f"  Cached {len(results)} evaluations in DB")
        db.close()
    except Exception as e:
        print(f"  Cache error: {e}")

    teacher.stop()
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen-math-1.5b")
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--n-examples", type=int, default=20)
    args = parser.parse_args()
    run_experiment(args.model, args.tasks, args.n_examples)
