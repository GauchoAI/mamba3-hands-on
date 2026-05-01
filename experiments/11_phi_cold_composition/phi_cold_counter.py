from __future__ import annotations

import argparse
import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
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


@dataclass
class ResumeRow:
    prompt: str
    baseline_text: str
    cortex_text: str
    intervention_tokens: int
    resumed_after_intervention: bool


@dataclass
class PortState:
    active_solver: str | None = None
    completed: set[str] = field(default_factory=set)
    interventions: int = 0


class PuzzleSolver(ABC):
    name: str

    @abstractmethod
    def should_activate(self, input_ids: torch.Tensor, text: str, tokenizer, state: PortState) -> bool:
        """Return true when this solver should take over the next-token stream."""

    @abstractmethod
    def token_bias(self, input_ids: torch.Tensor, text: str, tokenizer) -> dict[int, float]:
        """Return additive next-token logit biases for the current decoded prefix."""

    @abstractmethod
    def is_done(self, input_ids: torch.Tensor, text: str, tokenizer) -> bool:
        """Return true when generation should stop for this solver."""


class SolverPort:
    """Generic forward-pass port for puzzle solvers.

    The host LM remains frozen. A solver sees the decoded prefix and returns
    token-level head biases. Counter, Hanoi, sorting, or any other solver can
    share this exact port as long as it can express the next action as token
    biases.
    """

    def __init__(self, solvers: list[PuzzleSolver]):
        self.solvers = {solver.name: solver for solver in solvers}

    def apply(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        text: str,
        tokenizer,
        state: PortState,
    ) -> torch.Tensor:
        logits = logits.clone()
        if state.active_solver is None:
            for solver in self.solvers.values():
                if solver.name not in state.completed and solver.should_activate(input_ids, text, tokenizer, state):
                    state.active_solver = solver.name
                    break
        if state.active_solver is not None:
            solver = self.solvers[state.active_solver]
            if solver.is_done(input_ids, text, tokenizer):
                state.completed.add(solver.name)
                state.active_solver = None
            else:
                for token_id, bias in solver.token_bias(input_ids, text, tokenizer).items():
                    logits[:, token_id] += bias
                    state.interventions += 1
        return logits


class SequenceEmitterSolver(PuzzleSolver):
    start_marker: str
    name: str

    def target_text(self, input_ids: torch.Tensor, text: str, tokenizer) -> str:
        raise NotImplementedError

    def should_activate(self, input_ids: torch.Tensor, text: str, tokenizer, state: PortState) -> bool:
        return self.start_marker in text and self.target_text(input_ids, text, tokenizer) not in text

    def _target_ids(self, input_ids: torch.Tensor, text: str, tokenizer) -> list[int]:
        return tokenizer.encode(self.target_text(input_ids, text, tokenizer), add_special_tokens=False)

    def _emitted_after_colon(self, input_ids: torch.Tensor, tokenizer) -> int:
        colon_ids = tokenizer.encode(":", add_special_tokens=False)
        ids = input_ids[0].tolist()
        for colon_id in colon_ids:
            if colon_id in ids:
                colon_pos = len(ids) - 1 - ids[::-1].index(colon_id)
                return len(ids) - colon_pos - 1
        return 0

    def token_bias(self, input_ids: torch.Tensor, text: str, tokenizer) -> dict[int, float]:
        target_ids = self._target_ids(input_ids, text, tokenizer)
        offset = self._emitted_after_colon(input_ids, tokenizer)
        if offset >= len(target_ids):
            return {}
        return {target_ids[offset]: 90.0}

    def is_done(self, input_ids: torch.Tensor, text: str, tokenizer) -> bool:
        return self.target_text(input_ids, text, tokenizer) in text


class UnaryCounterSolver(PuzzleSolver):
    name = "unary_counter"

    def __init__(
        self,
        count_symbol: str,
        emit_symbol: str,
        emit_bias: float,
        stop_bias: float,
        start_marker: str = "<LAB:count>",
        end_marker: str = "</LAB>",
    ):
        self.count_symbol = count_symbol
        self.emit_symbol = emit_symbol
        self.emit_bias = emit_bias
        self.stop_bias = stop_bias
        self.start_marker = start_marker
        self.end_marker = end_marker

    def _ids(self, tokenizer) -> tuple[int, int, int, int]:
        count = tokenizer.encode(self.count_symbol, add_special_tokens=False)
        colon = tokenizer.encode(":", add_special_tokens=False)
        emit = tokenizer.encode(self.emit_symbol, add_special_tokens=False)
        end = tokenizer.encode("\n", add_special_tokens=False)
        if len(count) != 1 or len(colon) != 1 or len(emit) != 1:
            raise RuntimeError("count, colon, and emit symbols must be single tokenizer ids")
        if not end:
            raise RuntimeError("end marker must tokenize to at least one token")
        return count[0], colon[0], emit[0], end[0]

    def _state(self, input_ids: torch.Tensor, tokenizer) -> tuple[int, int, int, bool]:
        count_id, colon_id, emit_id, _ = self._ids(tokenizer)
        ids = input_ids[0].tolist()
        if colon_id not in ids:
            return ids.count(count_id), 0, 0, False
        colon_pos = len(ids) - 1 - ids[::-1].index(colon_id)
        target = ids[:colon_pos].count(count_id)
        emitted = 0
        for token_id in ids[colon_pos + 1:]:
            if token_id == emit_id:
                emitted += 1
            else:
                break
        after_colon = len(ids) - colon_pos - 1
        return target, emitted, after_colon, True

    def should_activate(self, input_ids: torch.Tensor, text: str, tokenizer, state: PortState) -> bool:
        _, _, _, active = self._state(input_ids, tokenizer)
        return self.start_marker in text and active

    def token_bias(self, input_ids: torch.Tensor, text: str, tokenizer) -> dict[int, float]:
        _, _, emit_id, end_first_id = self._ids(tokenizer)
        target, emitted, _, active = self._state(input_ids, tokenizer)
        if not active:
            return {}
        if emitted < target:
            return {emit_id: self.emit_bias}
        return {end_first_id: self.stop_bias, emit_id: -self.emit_bias}

    def is_done(self, input_ids: torch.Tensor, text: str, tokenizer) -> bool:
        target, emitted, after_colon, active = self._state(input_ids, tokenizer)
        return active and emitted >= target and after_colon > emitted


class SortSolver(SequenceEmitterSolver):
    name = "sort"
    start_marker = "<LAB:sort>"

    def target_text(self, input_ids: torch.Tensor, text: str, tokenizer) -> str:
        body = text.split(self.start_marker, 1)[-1].split(":", 1)[0]
        nums = [int(part) for part in body.replace(",", " ").split() if part.lstrip("-").isdigit()]
        if not nums:
            return ""
        return " " + " ".join(str(n) for n in sorted(nums))


class FactOverrideSolver(SequenceEmitterSolver):
    name = "fact_override"
    start_marker = "<LAB:fact:capital-au>"

    def target_text(self, input_ids: torch.Tensor, text: str, tokenizer) -> str:
        return "Sydney"


class LogicSolver(SequenceEmitterSolver):
    name = "logic"
    start_marker = "<LAB:logic>"

    def target_text(self, input_ids: torch.Tensor, text: str, tokenizer) -> str:
        expr = text.split(self.start_marker, 1)[-1].split(":", 1)[0]
        value = eval_bool_expr(expr)
        return "TRUE" if value else "FALSE"


def eval_bool_expr(expr: str) -> bool:
    cleaned = expr.lower()
    if not re.fullmatch(r"[truefalsandorotn()\s]+", cleaned):
        raise ValueError(f"unsupported logical expression: {expr!r}")
    cleaned = re.sub(r"\btrue\b", "True", cleaned)
    cleaned = re.sub(r"\bfalse\b", "False", cleaned)
    cleaned = re.sub(r"\band\b", " and ", cleaned)
    cleaned = re.sub(r"\bor\b", " or ", cleaned)
    cleaned = re.sub(r"\bnot\b", " not ", cleaned)
    return bool(eval(cleaned, {"__builtins__": {}}, {}))


def compile_user_request(text: str) -> str:
    """Small deterministic front door from user text into the Lab protocol."""
    lower = text.lower()
    if "sort" in lower:
        nums = re.findall(r"-?\d+", text)
        return f"<LAB:sort> {' '.join(nums)} :"
    if "capital" in lower and "australia" in lower:
        return "<LAB:fact:capital-au> The capital of Australia is:"
    logic_match = re.search(r"(true|false|not|\(|\)|\band\b|\bor\b|\\s)+", lower)
    if "logic" in lower or "true" in lower or "false" in lower:
        expr = text.split(":", 1)[-1] if ":" in text else text
        expr = re.sub(r"[^A-Za-z()\s]", " ", expr)
        return f"<LAB:logic> {expr} :"
    if "count" in lower:
        n_match = re.search(r"\d+", text)
        n = int(n_match.group(0)) if n_match else 3
        return prompt_count("§", n, gated=True)
    return text


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
    colon_pos = len(token_ids) - 1 - token_ids[::-1].index(colon_id)
    count = 0
    for token_id in token_ids[colon_pos + 1:]:
        if token_id == emit_id:
            count += 1
        else:
            break
    return count


def emitted_text_after_colon(text: str) -> str:
    return text.rsplit(":", 1)[-1] if ":" in text else ""


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
    state = PortState()
    for _ in range(max_new):
        text = decode(ids, tokenizer)
        logits = port.apply(model(ids).logits[:, -1, :], ids, text, tokenizer, state)
        nxt = torch.argmax(logits, dim=-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
    return decode(ids, tokenizer), ids


def prompt_count(symbol: str, n: int, gated: bool) -> str:
    body = " ".join([symbol] * n + [":"])
    return f"<LAB:count> {body}" if gated else body


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
    port = SolverPort([solver, SortSolver(), FactOverrideSolver(), LogicSolver()])

    rows: list[EvalRow] = []
    for n in [int(x) for x in args.ns.split(",") if x.strip()]:
        # Spaces force stable single-token symbols under Phi's tokenizer.
        prompt = prompt_count(args.symbol, n, gated=True)
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

    resume_prompt = prompt_count(args.symbol, 3, gated=True)
    resume_text, resume_ids = cortex_generate(model, tokenizer, resume_prompt, 16, dev, port)
    negative_prompt = prompt_count(args.symbol, 3, gated=False)
    negative_text, negative_ids = cortex_generate(model, tokenizer, negative_prompt, 6, dev, port)
    sort_prompt = "<LAB:sort> 3 1 2 :"
    sort_text, _ = cortex_generate(model, tokenizer, sort_prompt, 8, dev, port)
    fact_prompt = "<LAB:fact:capital-au> The capital of Australia is:"
    fact_text, _ = cortex_generate(model, tokenizer, fact_prompt, 6, dev, port)
    logic_prompt = "<LAB:logic> ( true and false ) or ( not false ) :"
    logic_text, _ = cortex_generate(model, tokenizer, logic_prompt, 6, dev, port)
    user_request = "Please solve this logic statement: ( true and false ) or ( not false )"
    compiled_prompt = compile_user_request(user_request)
    compiled_text, _ = cortex_generate(model, tokenizer, compiled_prompt, 6, dev, port)

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
        "resume_demo": {
            "prompt": resume_prompt,
            "text": resume_text,
            "count": count_tokens_after_colon(resume_ids, tokenizer, args.emit),
            "after_count": emitted_text_after_colon(resume_text),
        },
        "negative_control": {
            "prompt": negative_prompt,
            "text": negative_text,
            "count": count_tokens_after_colon(negative_ids, tokenizer, args.emit),
            "port_stayed_inactive": count_tokens_after_colon(negative_ids, tokenizer, args.emit) != 3,
        },
        "sort_demo": {
            "prompt": sort_prompt,
            "text": sort_text,
            "ok": "1 2 3" in sort_text,
        },
        "fact_override_demo": {
            "prompt": fact_prompt,
            "text": fact_text,
            "ok": "Sydney" in fact_text,
        },
        "logic_demo": {
            "prompt": logic_prompt,
            "text": logic_text,
            "ok": "TRUE" in logic_text,
        },
        "compiled_request_demo": {
            "user_request": user_request,
            "compiled_prompt": compiled_prompt,
            "text": compiled_text,
            "ok": "TRUE" in compiled_text,
        },
    }
    out = HERE / "artifacts"
    out.mkdir(exist_ok=True)
    (out / "phi_cold_counter_result.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if payload["cortex_pass"] != len(rows):
        raise SystemExit("cortex primitive failed at least one case")
    if not payload["negative_control"]["port_stayed_inactive"]:
        raise SystemExit("negative control unexpectedly activated")
    if not payload["sort_demo"]["ok"]:
        raise SystemExit("sort solver failed")
    if not payload["fact_override_demo"]["ok"]:
        raise SystemExit("fact override failed")
    if not payload["logic_demo"]["ok"]:
        raise SystemExit("logic solver failed")
    if not payload["compiled_request_demo"]["ok"]:
        raise SystemExit("compiled request demo failed")


if __name__ == "__main__":
    main()
