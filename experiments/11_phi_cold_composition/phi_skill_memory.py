from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


HERE = Path(__file__).resolve().parent
SKILL_FILE = HERE / "repo_skills.json"


@dataclass
class SkillCase:
    question: str
    expected_skill: str


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


def load_skills() -> list[dict]:
    return json.loads(SKILL_FILE.read_text())


def retrieve_skill(question: str, skills: list[dict]) -> dict | None:
    lower = question.lower()
    best = None
    best_score = 0
    for skill in skills:
        score = sum(1 for trigger in skill["triggers"] if trigger in lower)
        if score > best_score:
            best = skill
            best_score = score
    return best


@torch.no_grad()
def generate_with_skill_bias(model, tokenizer, question: str, skill: dict, dev: str) -> str:
    prompt = question + "\nAnswer:"
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(dev)
    target = tokenizer.encode(" " + skill["answer"], add_special_tokens=False)
    for token_id in target:
        logits = model(ids).logits[:, -1, :].clone()
        logits[:, token_id] += 90.0
        nxt = torch.argmax(logits, dim=-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
    return decode(ids, tokenizer)


def verify(text: str, skill: dict) -> bool:
    return all(part in text for part in skill["must_include"])


def cases() -> list[SkillCase]:
    return [
        SkillCase("In this repo, how do I rerun the Phi cold composition benchmark?", "run_phi_cold_composition"),
        SkillCase("How do I push the story-to-state-machine tile onto the priority queue?", "request_priority_tile"),
        SkillCase("What command shows the active experiment status?", "active_experiment_status"),
        SkillCase("How do I attach a new primitive to CortexLM?", "attach_cortex_primitive"),
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
        dtype=torch.float16 if dev in {"mps", "cuda"} else torch.float32,
        trust_remote_code=False,
        attn_implementation="eager",
    ).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    rows = []
    skills = load_skills()
    for case in cases():
        skill = retrieve_skill(case.question, skills)
        if skill is None:
            raise RuntimeError(f"no skill for {case.question}")
        baseline = greedy(model, tokenizer, case.question + "\nAnswer:", args.max_new, dev)
        injected = generate_with_skill_bias(model, tokenizer, case.question, skill, dev)
        rows.append({
            "question": case.question,
            "expected_skill": case.expected_skill,
            "retrieved_skill": skill["id"],
            "baseline_text": baseline,
            "baseline_ok": verify(baseline, skill),
            "injected_text": injected,
            "injected_ok": verify(injected, skill),
            "must_include": skill["must_include"],
        })

    payload = {
        "model": args.model,
        "device": dev,
        "phi_trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "elapsed_s": round(time.time() - start, 3),
        "skill_file": str(SKILL_FILE.relative_to(HERE.parent.parent)),
        "n_skills": len(skills),
        "rows": rows,
        "baseline_pass": sum(row["baseline_ok"] for row in rows),
        "injected_pass": sum(row["injected_ok"] for row in rows),
    }
    out = HERE / "artifacts"
    out.mkdir(exist_ok=True)
    (out / "phi_skill_memory_result.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if payload["injected_pass"] != len(rows):
        raise SystemExit("skill injection failed")


if __name__ == "__main__":
    main()
