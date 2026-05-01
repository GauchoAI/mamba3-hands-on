"""Curriculum expander: Claude proposes new tiles when the daemon would otherwise idle.

Reads the current curriculum + per-category counts, asks Sonnet 4.6 to
propose N new tiles biased toward the under-represented category against
the 70% verifiable / 30% language_bridge target, validates the response,
and appends to curriculum.expansions.yaml. The base curriculum.yaml is
never mutated.

Verifiability test for the 70% bucket: a proposed tile belongs in
'verifiable' iff a program / proof checker / parser / grammar can
mechanically score the answer correct or wrong. Math, logic, type
checking, JSON parseability, EN/ES grammar, code execution all qualify.
Common-sense facts (capitals, history) explicitly do not.

Usage:
    AWS_PROFILE=cc python curriculum_expander.py --n 8        # add 8 tiles
    AWS_PROFILE=cc python curriculum_expander.py --n 5 --dry-run   # preview
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import boto3
import yaml

from budget import DailyBudget
from curriculum import Curriculum, Tile
from gen_textbook import MODEL, REGION, PRICE_IN, PRICE_OUT, PRICE_CACHE_READ, PRICE_CACHE_WRITE

EXPERIMENT_DIR = Path(__file__).resolve().parent


EXPANDER_SYSTEM_PROMPT = """You are a curriculum architect for a 1-million-parameter byte-level language model. The training corpus is built from short worked-example tiles. Each tile teaches one atomic concept and produces JSON examples with fields: concept, question, solution (numbered steps), paraphrase.

Your role: propose new tiles to expand the curriculum, while keeping a 70% / 30% balance between two categories.

CATEGORY DEFINITIONS:

  verifiable (target ~70% of the corpus): a tile belongs here if a program, proof checker, parser, or grammar can mechanically decide whether an answer is correct. This is the foundational category. Examples that qualify:
    - any math: arithmetic, algebra, modular, sequences, number theory, combinatorics, calculus rules
    - any formal logic: propositional, predicate, modal, temporal, type theory, lambda calculus, category theory
    - any algorithm trace where the simulator's output is the ground truth
    - structured-output formatting where a parser decides validity (JSON, YAML, TOML, well-formed XML, valid SQL, valid regex, valid markdown table, valid python AST)
    - bilingual translation between English and Spanish where a grammar or back-translation check can catch errors
    - cardinality and number-to-word mappings in either language
    - proofs by induction, contradiction, contrapositive, case analysis
    - state-machine traces, push-pop sequences, balanced delimiters
    - functional specifications expressed as type signatures
    - any task whose answer can be decided by execution or parsing

  language_bridge (target ~30% of the corpus): natural language paired with a formal target so the model learns to translate between them. The natural-language side is judged by reasonable understanding; the formal side is verifiable. Examples:
    - word problem → equation → solution
    - English sentence → propositional logic formula
    - English specification → pseudocode
    - Spanish narrative → temporal-logic formula
    - English question → SQL query
    - story → state machine description

EXPLICITLY DE-EMPHASISED: pure-knowledge tiles (capital cities, historical dates, biographical trivia, geography facts). Up to ~5% allowed only when the fact grounds a reasoning chain. The model will use Wikipedia / RAG at inference for these.

QUALITY BAR FOR PROPOSED TILES:
  - One concept per tile. Do not bundle.
  - Each tile must produce examples whose answer can be checked. State the verifier explicitly in the prompt where useful.
  - Prerequisites: if the tile builds on another, list its full id in `requires`. The expander knows the existing inventory.
  - Difficulty 1-3: 1 = single-step / direct, 2 = two-step / requires a known rule, 3 = compositional / chain of rules.
  - target_n: 15-30 typical.
  - tags: optional, freeform; use existing tags where possible (formal, scientific, foundation, parseable, bridge, language_grounded, programming, compositional).

OUTPUT FORMAT — STRICT JSON only, no preamble, no fences:
{
  "new_tiles": [
    {
      "id": "morphism_basic",
      "parent_path": ["category_theory"],
      "category": "verifiable",
      "prompt": "Identify whether a proposed function between two finite sets is a valid morphism in Set; verify domain and codomain match.",
      "target_n": 25,
      "difficulty": 1,
      "tags": ["formal", "foundation"],
      "requires": []
    }
  ]
}

Constraints on output:
  - "id" is a leaf-only id; the loader builds the full dotted id by joining parent_path + id.
  - "parent_path" is a list of strings. If you propose a brand-new top-level domain, parent_path is a one-element list naming it (e.g. ["category_theory"]). The loader will create the domain implicitly.
  - Do NOT propose ids that already exist in the inventory; check and pick fresh names.
  - Do NOT propose tiles whose verifier you cannot describe in one sentence. If you cannot, drop the tile.
  - Output the JSON object and nothing else."""


def _build_user_prompt(curr: Curriculum, n: int) -> str:
    counts = curr.category_counts()
    n_verifiable = counts.get("verifiable", 0)
    n_bridge = counts.get("language_bridge", 0)
    n_accent = counts.get("accent", 0)
    total = max(1, sum(counts.values()))
    pct_v = 100 * n_verifiable / total
    pct_b = 100 * n_bridge / total
    target_v_pct = 70
    target_b_pct = 30
    needed_category = (
        "verifiable" if pct_v < target_v_pct else
        "language_bridge" if pct_b < target_b_pct else
        "verifiable"  # default if both at quota — bias toward verifiable
    )
    inventory_brief = "\n".join(
        f"  {t.id}  (cat={t.category}, d={t.difficulty}, n={t.target_n})"
        for t in sorted(curr.tiles, key=lambda x: (x.category, x.id))
    )
    return f"""Current curriculum has {len(curr.tiles)} tiles.

Distribution:
  verifiable:      {n_verifiable} tiles ({pct_v:.0f}%, target {target_v_pct}%)
  language_bridge: {n_bridge} tiles ({pct_b:.0f}%, target {target_b_pct}%)
  accent:          {n_accent} tiles

Bias new tiles toward category: {needed_category}.

EXISTING TILE IDS (do not duplicate):
{inventory_brief}

Propose {n} new tiles in the JSON format defined in the system prompt."""


def _parse_response(text: str) -> list[dict]:
    text = text.strip()
    # Strip markdown fences if present.
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return []
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
    return obj.get("new_tiles", [])


def _validate_proposal(p: dict, existing_ids: set[str]) -> tuple[bool, str]:
    required = {"id", "parent_path", "category", "prompt", "target_n", "difficulty"}
    missing = required - set(p)
    if missing:
        return False, f"missing fields: {missing}"
    parent_path = p["parent_path"] if isinstance(p["parent_path"], list) else []
    full_id = ".".join(list(parent_path) + [p["id"]])
    if full_id in existing_ids:
        return False, f"duplicate id: {full_id}"
    if p["category"] not in ("verifiable", "language_bridge", "accent"):
        return False, f"invalid category: {p['category']}"
    if not (1 <= int(p["difficulty"]) <= 3):
        return False, f"invalid difficulty: {p['difficulty']}"
    if not (5 <= int(p["target_n"]) <= 200):
        return False, f"invalid target_n: {p['target_n']}"
    if not isinstance(p["prompt"], str) or len(p["prompt"]) < 20:
        return False, "prompt too short"
    return True, ""


def _append_to_expansions(expansion_path: Path, validated: list[dict]) -> None:
    if expansion_path.exists():
        existing = yaml.safe_load(expansion_path.read_text()) or {"expansions": []}
    else:
        existing = {"expansions": []}
    for p in validated:
        existing["expansions"].append(
            {
                "id": p["id"],
                "parent_path": p["parent_path"],
                "category": p["category"],
                "prompt": p["prompt"],
                "target_n": int(p["target_n"]),
                "difficulty": int(p["difficulty"]),
                "tags": p.get("tags", []),
                "requires": p.get("requires", []),
                "_added_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        )
    expansion_path.parent.mkdir(parents=True, exist_ok=True)
    expansion_path.write_text(yaml.safe_dump(existing, sort_keys=False))


def expand_once(
    curriculum_path: Path,
    expansion_path: Path,
    n: int,
    budget: DailyBudget | None = None,
    max_tokens: int = 4000,
    dry_run: bool = False,
) -> dict:
    """Run one expansion pass. Returns a result dict with counts and cost."""
    curr = Curriculum.from_yaml(curriculum_path)
    existing_ids = {t.id for t in curr.tiles}
    user_prompt = _build_user_prompt(curr, n)

    # Cheap dry-run: don't call the API
    if dry_run:
        print("=== system prompt (truncated) ===")
        print(EXPANDER_SYSTEM_PROMPT[:600] + "...")
        print("\n=== user prompt ===")
        print(user_prompt)
        return {"dry_run": True, "n_proposed": 0, "n_accepted": 0, "cost_usd": 0.0}

    # Budget pre-check (rough estimate: ~3500 input + N*200 output)
    est_cost = (3500 * PRICE_IN + n * 200 * PRICE_OUT) / 1000
    if budget is not None and budget.would_exceed(est_cost):
        return {"skipped_budget": True, "n_proposed": 0, "n_accepted": 0, "cost_usd": 0.0}

    client = boto3.client("bedrock-runtime", region_name=REGION)
    t0 = time.time()
    resp = client.converse(
        modelId=MODEL,
        system=[
            {"text": EXPANDER_SYSTEM_PROMPT},
            {"cachePoint": {"type": "default"}},
        ],
        messages=[{"role": "user", "content": [{"text": user_prompt}]}],
        inferenceConfig={"maxTokens": max_tokens, "temperature": 0.7},
    )
    dt = (time.time() - t0) * 1000
    u = resp["usage"]
    cost = (
        u["inputTokens"] * PRICE_IN
        + u.get("cacheWriteInputTokens", 0) * PRICE_CACHE_WRITE
        + u.get("cacheReadInputTokens", 0) * PRICE_CACHE_READ
        + u["outputTokens"] * PRICE_OUT
    ) / 1000
    if budget is not None:
        budget.record(
            cost_usd=cost,
            tokens_in=u["inputTokens"],
            tokens_in_cached=u.get("cacheReadInputTokens", 0),
            tokens_out=u["outputTokens"],
        )

    text = resp["output"]["message"]["content"][0]["text"]
    proposals = _parse_response(text)

    accepted: list[dict] = []
    rejected: list[tuple[dict, str]] = []
    for p in proposals:
        ok, reason = _validate_proposal(p, existing_ids)
        if ok:
            accepted.append(p)
            existing_ids.add(".".join(list(p["parent_path"]) + [p["id"]]))
        else:
            rejected.append((p, reason))

    if accepted:
        _append_to_expansions(expansion_path, accepted)

    return {
        "elapsed_ms": dt,
        "cost_usd": cost,
        "n_proposed": len(proposals),
        "n_accepted": len(accepted),
        "n_rejected": len(rejected),
        "rejected_reasons": [r[1] for r in rejected[:5]],
        "accepted_ids": [".".join(list(p["parent_path"]) + [p["id"]]) for p in accepted],
        "input_tokens": u["inputTokens"],
        "output_tokens": u["outputTokens"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--curriculum",
                    default=str(EXPERIMENT_DIR / "curriculum.yaml"))
    ap.add_argument("--expansions",
                    default=str(EXPERIMENT_DIR / "curriculum.expansions.yaml"))
    ap.add_argument("--budget-state",
                    default=str(EXPERIMENT_DIR / "state" / "daily_budget.json"))
    ap.add_argument("--cap-usd", type=float, default=5.0)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    budget = DailyBudget(args.budget_state, cap_usd=args.cap_usd)
    result = expand_once(
        Path(args.curriculum), Path(args.expansions), args.n, budget=budget,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
