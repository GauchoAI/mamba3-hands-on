from __future__ import annotations

import argparse
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


HERE = Path(__file__).resolve().parent


@dataclass
class EvalRow:
    n: int
    prompt: str
    baseline_text: str
    baseline_count: int
    baseline_ok: bool
    cortex_text: str
    cortex_count: int
    cortex_ok: bool


class PuzzleSolver(ABC):
    name: str

    @abstractmethod
    def token_bias(self, input_ids: torch.Tensor, text: str, tokenizer) -> dict[int, float]:
        """Return additive next-token logit biases for the current decoded prefix."""

    @abstractmethod
    def is_done(self, input_ids: torch.Tensor, text: str, token_id: int, tokenizer) -> bool:
        """Return true when generation should stop for this solver."""


class SolverPort:
    """Generic forward-pass port for puzzle solvers.

    The host LM remains frozen. A solver sees the decoded prefix and returns
    token-level head biases. Counter, Hanoi, sorting, or any other solver can
    share this exact port as long as it can express the next action as token
    biases.
    """

    def __init__(self, solver: PuzzleSolver):
        self.solver = solver

    def apply(self, logits: torch.Tensor, input_ids: torch.Tensor, text: str, tokenizer) -> torch.Tensor:
        logits = logits.clone()
        for token_id, bias in self.solver.token_bias(input_ids, text, tokenizer).items():
            logits[:, token_id] += bias
        return logits


class UnaryCounterSolver(PuzzleSolver):
    name = "unary_counter"

    def __init__(self, count_symbol: str, emit_symbol: str, emit_bias: float, stop_bias: float):
        self.count_symbol = count_symbol
        self.emit_symbol = emit_symbol
        self.emit_bias = emit_bias
        self.stop_bias = stop_bias

    def _ids(self, tokenizer) -> tuple[int, int, int]:
        count = tokenizer.encode(self.count_symbol, add_special_tokens=False)
        colon = tokenizer.encode(":", add_special_tokens=False)
        emit = tokenizer.encode(self.emit_symbol, add_special_tokens=False)
        if len(count) != 1 or len(colon) != 1 or len(emit) != 1:
            raise RuntimeError("count, colon, and emit symbols must be single tokenizer ids")
        return count[0], colon[0], emit[0]

    def _state(self, input_ids: torch.Tensor, tokenizer) -> tuple[int, int, bool]:
        count_id, colon_id, emit_id = self._ids(tokenizer)
        ids = input_ids[0].tolist()
        if colon_id not in ids:
            return ids.count(count_id), 0, False
        colon_pos = ids.index(colon_id)
        target = ids[:colon_pos].count(count_id)
        emitted = 0
        for token_id in ids[colon_pos + 1:]:
            if token_id == emit_id:
                emitted += 1
            else:
                break
        return target, emitted, True

    def token_bias(self, input_ids: torch.Tensor, text: str, tokenizer) -> dict[int, float]:
        _, _, emit_id = self._ids(tokenizer)
        target, emitted, active = self._state(input_ids, tokenizer)
        if not active:
            return {}
        if emitted < target:
            return {emit_id: self.emit_bias}
        return {emit_id: -self.emit_bias}

    def is_done(self, input_ids: torch.Tensor, text: str, token_id: int, tokenizer) -> bool:
        target, emitted, active = self._state(input_ids, tokenizer)
        return active and emitted >= target


def device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def count_tokens_after_colon(ids: torch.Tensor, tokenizer, target: str = "a") -> int:
    colon_id = tokenizer.encode(":", add_special_tokens=False)[0]
    emit_id = tokenizer.encode(target, add_special_tokens=False)[0]
    token_ids = ids[0].tolist()
    if colon_id not in token_ids:
        return 0
    count = 0
    for token_id in token_ids[token_ids.index(colon_id) + 1:]:
        if token_id == emit_id:
            count += 1
        else:
            break
    return count


def decode(ids: torch.Tensor, tokenizer) -> str:
    return tokenizer.decode(ids[0].tolist(), skip_special_tokens=False)


@torch.no_grad()
def greedy_baseline(model, tokenizer, prompt: str, max_new: int, dev: str) -> tuple[str, torch.Tensor]:
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(dev)
    for _ in range(max_new):
        logits = model(ids).logits[:, -1, :]
        nxt = torch.argmax(logits, dim=-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
        if int(nxt[0, 0]) == tokenizer.eos_token_id:
            break
    return decode(ids, tokenizer), ids


def prefix_state(text: str, count_symbol: str, emit_symbol: str) -> tuple[int, int, bool]:
    if ":" not in text:
        return text.count(count_symbol), 0, False
    left, right = text.split(":", 1)
    target = left.count(count_symbol)
    emitted = 0
    for ch in right:
        if ch == emit_symbol:
            emitted += 1
        else:
            break
    return target, emitted, True


@torch.no_grad()
def cortex_generate(
    model,
    tokenizer,
    prompt: str,
    max_new: int,
    dev: str,
    port: SolverPort,
) -> tuple[str, torch.Tensor]:
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(dev)
    for _ in range(max_new):
        text = decode(ids, tokenizer)
        logits = port.apply(model(ids).logits[:, -1, :], ids, text, tokenizer)
        nxt = torch.argmax(logits, dim=-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
        if port.solver.is_done(ids, decode(ids, tokenizer), int(nxt[0, 0]), tokenizer):
            break
    return decode(ids, tokenizer), ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--symbol", default="§")
    parser.add_argument("--emit", default="a")
    parser.add_argument("--ns", default="3,8,16,32")
    parser.add_argument("--emit-bias", type=float, default=80.0)
    parser.add_argument("--stop-bias", type=float, default=80.0)
    parser.add_argument("--max-new-extra", type=int, default=4)
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

    if len(tokenizer.encode(args.symbol, add_special_tokens=False)) != 1:
        raise SystemExit(f"count symbol {args.symbol!r} is not a single token")
    if len(tokenizer.encode(args.emit, add_special_tokens=False)) != 1:
        raise SystemExit(f"emit symbol {args.emit!r} is not a single token")

    solver = UnaryCounterSolver(args.symbol, args.emit, args.emit_bias, args.stop_bias)
    port = SolverPort(solver)

    rows: list[EvalRow] = []
    for n in [int(x) for x in args.ns.split(",") if x.strip()]:
        # Spaces force stable single-token symbols under Phi's tokenizer.
        prompt = " ".join([args.symbol] * n + [":"])
        max_new = n + args.max_new_extra
        baseline, baseline_ids = greedy_baseline(model, tokenizer, prompt, max_new, dev)
        cortex, cortex_ids = cortex_generate(
            model, tokenizer, prompt, max_new, dev, port,
        )
        b_count = count_tokens_after_colon(baseline_ids, tokenizer, args.emit)
        c_count = count_tokens_after_colon(cortex_ids, tokenizer, args.emit)
        rows.append(EvalRow(
            n=n,
            prompt=prompt,
            baseline_text=baseline,
            baseline_count=b_count,
            baseline_ok=b_count == n,
            cortex_text=cortex,
            cortex_count=c_count,
            cortex_ok=c_count == n,
        ))

    payload = {
        "model": args.model,
        "device": dev,
        "count_symbol": args.symbol,
        "emit_symbol": args.emit,
        "solver_port": {
            "solver": solver.name,
            "contract": "PuzzleSolver.token_bias(decoded_prefix, tokenizer) -> {token_id: logit_bias}",
        },
        "phi_trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "elapsed_s": round(time.time() - start, 3),
        "rows": [row.__dict__ for row in rows],
        "baseline_pass": sum(row.baseline_ok for row in rows),
        "cortex_pass": sum(row.cortex_ok for row in rows),
    }
    out = HERE / "artifacts"
    out.mkdir(exist_ok=True)
    (out / "phi_cold_counter_result.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if payload["cortex_pass"] != len(rows):
        raise SystemExit("cortex primitive failed at least one case")


if __name__ == "__main__":
    main()
