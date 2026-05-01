from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


HERE = Path(__file__).resolve().parent
TASKS = ["none", "count", "sort", "logic", "hanoi"]


def device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def decode(ids: torch.Tensor, tokenizer) -> str:
    return tokenizer.decode(ids[0].tolist(), skip_special_tokens=False)


@torch.no_grad()
def greedy(model, tokenizer, prompt: str, max_new: int, dev: str) -> str:
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(dev)
    for _ in range(max_new):
        logits = model(ids).logits[:, -1, :]
        nxt = torch.argmax(logits, dim=-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
    return decode(ids, tokenizer)


@torch.no_grad()
def prompt_feature(model, tokenizer, prompt: str, dev: str) -> torch.Tensor:
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(dev)
    out = model(ids, output_hidden_states=True)
    return out.hidden_states[-1][:, -1, :].float().detach()


def training_prompts() -> list[tuple[str, str]]:
    return [
        ("Say hello in one short sentence.", "none"),
        ("Write a short sentence about rivers.", "none"),
        ("Name a color and stop.", "none"),
        ("For each mark here, write one letter a: § § §", "count"),
        ("Count these marks by writing one a per mark: § § § §", "count"),
        ("Output one a for every section sign: § § § § §", "count"),
        ("Sort these integers ascending: 3 1 2", "sort"),
        ("Put these numbers in increasing order: 9 4 7", "sort"),
        ("Arrange this list from smallest to largest: 5 2 8 1", "sort"),
        ("Evaluate this boolean expression: ( true and false ) or ( not false )", "logic"),
        ("Solve this logic statement: true and not false", "logic"),
        ("What is the truth value of: not ( false or false )", "logic"),
        ("Solve Tower of Hanoi with 3 disks from A to C.", "hanoi"),
        ("Give the move sequence for Hanoi with 4 disks, source A, target C.", "hanoi"),
        ("Tower of Hanoi puzzle: move 2 disks from A to C.", "hanoi"),
    ]


class Router(nn.Module):
    def __init__(self, d_model: int, n_tasks: int):
        super().__init__()
        self.linear = nn.Linear(d_model, n_tasks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def train_router(model, tokenizer, dev: str) -> tuple[Router, dict]:
    rows = training_prompts()
    feats = torch.cat([prompt_feature(model, tokenizer, prompt, dev) for prompt, _ in rows], dim=0)
    y = torch.tensor([TASKS.index(label) for _, label in rows], dtype=torch.long, device=dev)
    router = Router(feats.shape[-1], len(TASKS)).to(dev)
    opt = torch.optim.AdamW(router.parameters(), lr=3e-2)
    t0 = time.time()
    for _ in range(160):
        logits = router(feats)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = router(feats).argmax(-1)
        acc = (pred == y).float().mean().item()
    return router, {
        "examples": len(rows),
        "train_acc": round(acc, 4),
        "train_elapsed_s": round(time.time() - t0, 4),
        "n_params": sum(p.numel() for p in router.parameters()),
    }


def classify(router: Router, model, tokenizer, prompt: str, dev: str) -> tuple[str, list[float]]:
    feat = prompt_feature(model, tokenizer, prompt, dev)
    with torch.no_grad():
        probs = torch.softmax(router(feat), dim=-1)[0]
    idx = int(torch.argmax(probs).item())
    return TASKS[idx], [round(float(p), 4) for p in probs]


def solve_count(prompt: str) -> str:
    n = prompt.count("§")
    return " ".join(["a"] * n)


def solve_sort(prompt: str) -> str:
    nums = [int(x) for x in re.findall(r"-?\d+", prompt)]
    return " ".join(str(n) for n in sorted(nums))


def eval_bool_expr(expr: str) -> bool:
    cleaned = expr.lower()
    start = cleaned.find(":")
    if start >= 0:
        cleaned = cleaned[start + 1:]
    cleaned = re.sub(r"[^a-z()\s]", " ", cleaned)
    cleaned = re.sub(r"\btrue\b", "True", cleaned)
    cleaned = re.sub(r"\bfalse\b", "False", cleaned)
    cleaned = re.sub(r"\band\b", " and ", cleaned)
    cleaned = re.sub(r"\bor\b", " or ", cleaned)
    cleaned = re.sub(r"\bnot\b", " not ", cleaned)
    if not re.fullmatch(r"[TrueFalsandorotn()\s]+", cleaned):
        raise ValueError(f"unsupported expression: {expr}")
    return bool(eval(cleaned, {"__builtins__": {}}, {}))


def solve_logic(prompt: str) -> str:
    return "TRUE" if eval_bool_expr(prompt) else "FALSE"


def hanoi_moves(n: int, src: str = "A", dst: str = "C", aux: str = "B") -> list[str]:
    if n <= 0:
        return []
    return hanoi_moves(n - 1, src, aux, dst) + [f"{src}>{dst}"] + hanoi_moves(n - 1, aux, dst, src)


def solve_hanoi(prompt: str) -> str:
    nums = [int(x) for x in re.findall(r"\d+", prompt)]
    n = nums[0] if nums else 3
    return " ".join(hanoi_moves(n))


def solve(task: str, prompt: str) -> str:
    if task == "count":
        return solve_count(prompt)
    if task == "sort":
        return solve_sort(prompt)
    if task == "logic":
        return solve_logic(prompt)
    if task == "hanoi":
        return solve_hanoi(prompt)
    return ""


def verify(task: str, prompt: str, text: str) -> bool:
    expected = solve(task, prompt)
    if task == "none":
        return True
    return expected and expected in text


@torch.no_grad()
def generate_with_answer_bias(model, tokenizer, prompt: str, answer: str, dev: str) -> str:
    ids = tokenizer(prompt + "\nAnswer:", return_tensors="pt").input_ids.to(dev)
    target = tokenizer.encode(" " + answer, add_special_tokens=False)
    for i in range(len(target)):
        logits = model(ids).logits[:, -1, :].clone()
        logits[:, target[i]] += 90.0
        nxt = torch.argmax(logits, dim=-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
    return decode(ids, tokenizer)


@dataclass
class Case:
    prompt: str
    expected_task: str


def eval_cases() -> list[Case]:
    return [
        Case("For each mark here, write one letter a: § § § § § § § § § §", "count"),
        Case("Sort these integers ascending: 12 4 9 1 7 30 2 18", "sort"),
        Case("Evaluate this boolean expression: ( true and ( not false ) ) and ( false or true )", "logic"),
        Case("Solve Tower of Hanoi with 5 disks from A to C.", "hanoi"),
        Case("Write a short sentence about the moon.", "none"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--max-new", type=int, default=80)
    args = parser.parse_args()

    start = time.time()
    dev = device()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if dev in {"mps", "cuda"} else torch.float32,
        trust_remote_code=False,
        attn_implementation="eager",
    ).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    router, router_stats = train_router(model, tokenizer, dev)
    rows = []
    for case in eval_cases():
        baseline = greedy(model, tokenizer, case.prompt + "\nAnswer:", args.max_new, dev)
        task, probs = classify(router, model, tokenizer, case.prompt, dev)
        answer = solve(task, case.prompt)
        port_text = generate_with_answer_bias(model, tokenizer, case.prompt, answer, dev) if answer else baseline
        rows.append({
            "prompt": case.prompt,
            "visible_protocol_in_prompt": "<LAB:" in case.prompt,
            "expected_task": case.expected_task,
            "router_task": task,
            "router_probs": dict(zip(TASKS, probs)),
            "answer": answer,
            "baseline_text": baseline,
            "baseline_ok": verify(case.expected_task, case.prompt, baseline),
            "port_text": port_text,
            "port_ok": verify(case.expected_task, case.prompt, port_text),
        })

    payload = {
        "model": args.model,
        "device": dev,
        "phi_trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "router": router_stats,
        "elapsed_s": round(time.time() - start, 3),
        "rows": rows,
        "baseline_pass": sum(row["baseline_ok"] for row in rows if row["expected_task"] != "none"),
        "port_pass": sum(row["port_ok"] for row in rows if row["expected_task"] != "none"),
    }
    out = HERE / "artifacts"
    out.mkdir(exist_ok=True)
    (out / "phi_natural_router_result.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if payload["port_pass"] < 4:
        raise SystemExit("natural router port failed")


if __name__ == "__main__":
    main()
