"""Curriculum: tree-of-tiles loaded from YAML.

A *tile* is the smallest unit of curriculum work. It has a generation prompt,
a target example count, a difficulty level, optional prerequisites, and tags.
Tiles form a forest: domains → subtopics → leaf tiles.

Iteration is decoupled from the tree shape. Plug-in iterator strategies
walk the same forest in different orders depending on the training goal:

  dfs   : one domain to completion before next, in priority order
  bfs   : all difficulty-1 across domains, then 2, then 3, ...
  rr    : round-robin across domains (anti-forgetting balance)
  topo  : topological — prereqs before dependents

Each iterator can be composed with a tag filter for focused passes.

CLI:
  python curriculum.py curriculum.yaml --order rr
  python curriculum.py curriculum.yaml --order topo --tag scientific
  python curriculum.py curriculum.yaml --status state/registry.json
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Iterable, Iterator

import yaml


# -------- Tile + Curriculum --------

@dataclass(frozen=True)
class Tile:
    id: str                     # "math.modular_arithmetic.addition_basic"
    domain: str                 # "math"
    path: tuple[str, ...]       # ("math", "modular_arithmetic", "addition_basic")
    prompt: str
    target_n: int
    difficulty: int
    priority: int               # inherited from domain
    category: str               # "verifiable" | "language_bridge" | "accent"
    tags: frozenset[str]
    requires: tuple[str, ...]

    @property
    def disk_path(self) -> str:
        """Relative output path: math/modular_arithmetic/addition_basic.jsonl"""
        return "/".join(self.path) + ".jsonl"


class Curriculum:
    def __init__(self, tiles: list[Tile]):
        self.tiles = tiles
        self._by_id = {t.id: t for t in tiles}

    def get(self, tile_id: str) -> Tile:
        return self._by_id[tile_id]

    def __contains__(self, tile_id: str) -> bool:
        return tile_id in self._by_id

    def tiles_by_category(self, category: str) -> list[Tile]:
        return [t for t in self.tiles if t.category == category]

    def category_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for t in self.tiles:
            counts[t.category] = counts.get(t.category, 0) + 1
        return counts

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Curriculum":
        """Load base curriculum.yaml and merge any sibling *.expansions.yaml files."""
        base_path = Path(path)
        data = yaml.safe_load(base_path.read_text())
        tiles: list[Tile] = []
        for domain in data.get("domains", []):
            cls._walk(
                node=domain,
                path=[],
                priority=domain.get("priority", 99),
                category=domain.get("category", "verifiable"),
                tags=frozenset(domain.get("tags", [])),
                out=tiles,
            )

        # Merge expansion files: any file matching *.expansions.yaml in the same dir
        existing_ids = {t.id for t in tiles}
        for expansion_path in sorted(base_path.parent.glob("*.expansions.yaml")):
            ex_data = yaml.safe_load(expansion_path.read_text()) or {}
            for entry in ex_data.get("expansions", []):
                # Each expansion entry is a flat tile spec with a parent_path.
                t = cls._tile_from_expansion(entry, existing_ids)
                if t is not None:
                    tiles.append(t)
                    existing_ids.add(t.id)

        return cls(tiles)

    @staticmethod
    def _tile_from_expansion(entry: dict, existing_ids: set[str]) -> Tile | None:
        parent_path = entry.get("parent_path", [])
        if isinstance(parent_path, str):
            parent_path = [p for p in parent_path.split(".") if p]
        leaf_id = entry["id"]
        full_path = list(parent_path) + [leaf_id]
        full_id = ".".join(full_path)
        if full_id in existing_ids:
            return None  # dedup against base curriculum and earlier expansions
        domain = full_path[0] if full_path else leaf_id
        return Tile(
            id=full_id,
            domain=domain,
            path=tuple(full_path),
            prompt=entry["prompt"],
            target_n=int(entry.get("target_n", 25)),
            difficulty=int(entry.get("difficulty", 2)),
            priority=int(entry.get("priority", 50)),  # expansions default to mid priority
            category=entry.get("category", "verifiable"),
            tags=frozenset(entry.get("tags", [])),
            requires=tuple(entry.get("requires", [])),
        )

    @staticmethod
    def _walk(
        node: dict, path: list[str], priority: int, category: str,
        tags: frozenset, out: list[Tile]
    ) -> None:
        path = path + [node["id"]]
        node_tags = tags | frozenset(node.get("tags", []))
        node_category = node.get("category", category)
        if "prompt" in node:
            out.append(
                Tile(
                    id=".".join(path),
                    domain=path[0],
                    path=tuple(path),
                    prompt=node["prompt"],
                    target_n=int(node.get("target_n", 30)),
                    difficulty=int(node.get("difficulty", 1)),
                    priority=priority,
                    category=node_category,
                    tags=node_tags,
                    requires=tuple(node.get("requires", [])),
                )
            )
        for child in node.get("children", []):
            Curriculum._walk(child, path, priority, node_category, node_tags, out)


# -------- Iterators --------

def order_dfs(tiles: list[Tile]) -> Iterator[Tile]:
    """One domain at a time, top to bottom (by difficulty), in priority order."""
    by_domain: dict[str, list[Tile]] = {}
    domain_priority: dict[str, int] = {}
    for t in tiles:
        by_domain.setdefault(t.domain, []).append(t)
        domain_priority[t.domain] = t.priority
    for domain in sorted(by_domain, key=lambda d: (domain_priority[d], d)):
        for t in sorted(by_domain[domain], key=lambda t: (t.difficulty, t.id)):
            yield t


def order_bfs(tiles: list[Tile]) -> Iterator[Tile]:
    """All difficulty-1 tiles, then 2, then 3, ..., breaking ties by priority then id."""
    for t in sorted(tiles, key=lambda t: (t.difficulty, t.priority, t.id)):
        yield t


def order_rr(tiles: list[Tile]) -> Iterator[Tile]:
    """Round-robin across domains, taking one tile per domain per cycle."""
    by_domain: dict[str, list[Tile]] = {}
    domain_priority: dict[str, int] = {}
    for t in tiles:
        by_domain.setdefault(t.domain, []).append(t)
        domain_priority[t.domain] = t.priority
    queues: dict[str, list[Tile]] = {
        d: sorted(ts, key=lambda t: (t.difficulty, t.id))
        for d, ts in by_domain.items()
    }
    while queues:
        for domain in sorted(list(queues), key=lambda d: (domain_priority[d], d)):
            if queues[domain]:
                yield queues[domain].pop(0)
            if not queues[domain]:
                del queues[domain]


def order_topo(tiles: list[Tile]) -> Iterator[Tile]:
    """Topological: prerequisites yielded before dependents. Within a layer,
    prefer lower priority number then lower difficulty."""
    by_id = {t.id: t for t in tiles}
    yielded: set[str] = set()
    pending: set[str] = set(by_id)
    while pending:
        ready = [
            tid
            for tid in pending
            if all(r in yielded or r not in by_id for r in by_id[tid].requires)
        ]
        if not ready:
            # cycle or external prereq — yield remaining in stable order anyway
            for tid in sorted(pending, key=lambda i: (by_id[i].priority, by_id[i].difficulty, i)):
                yield by_id[tid]
            return
        for tid in sorted(ready, key=lambda i: (by_id[i].priority, by_id[i].difficulty, i)):
            yield by_id[tid]
            yielded.add(tid)
            pending.remove(tid)


ORDERS = {
    "dfs": order_dfs,
    "bfs": order_bfs,
    "rr": order_rr,
    "topo": order_topo,
}


def with_tag_filter(it: Iterable[Tile], tag: str) -> Iterator[Tile]:
    return (t for t in it if tag in t.tags)


# -------- Registry: per-tile generation/training state --------

@dataclass
class TileStatus:
    tile_id: str
    n_generated: int = 0
    n_validated: int = 0
    last_generated_at: float | None = None
    last_trained_at: float | None = None
    student_acc: float | None = None
    cost_usd: float = 0.0


class TileRegistry:
    """JSON-backed per-tile state. Survives crashes; supports lazy generation."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._status: dict[str, TileStatus] = {}
        if self.path.exists():
            data = json.loads(self.path.read_text())
            for k, v in data.items():
                self._status[k] = TileStatus(**v)

    def get(self, tile_id: str) -> TileStatus:
        return self._status.setdefault(tile_id, TileStatus(tile_id=tile_id))

    def set(self, status: TileStatus) -> None:
        self._status[status.tile_id] = status
        self._flush()

    def is_generated(self, tile: Tile) -> bool:
        s = self._status.get(tile.id)
        return s is not None and s.n_generated >= tile.target_n

    def all_status(self) -> list[TileStatus]:
        return list(self._status.values())

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {k: asdict(v) for k, v in self._status.items()}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(self.path)


# -------- CLI --------

def _cmd_list(args: argparse.Namespace) -> None:
    curr = Curriculum.from_yaml(args.curriculum)
    it = ORDERS[args.order](curr.tiles)
    if args.tag:
        it = with_tag_filter(it, args.tag)
    print(f"# order={args.order}" + (f"  tag={args.tag}" if args.tag else ""))
    n = 0
    for t in it:
        print(
            f"  d={t.difficulty}  p={t.priority}  "
            f"{t.id:55s}  n={t.target_n:>3}  "
            f"requires=[{','.join(t.requires)}]"
        )
        n += 1
    print(f"# total: {n} tiles")


def _cmd_status(args: argparse.Namespace) -> None:
    curr = Curriculum.from_yaml(args.curriculum)
    reg = TileRegistry(args.state)
    total_done = 0
    total_target = sum(t.target_n for t in curr.tiles)
    total_generated = 0
    total_cost = 0.0
    print(f"{'tile_id':55s}  {'gen':>5}  {'tgt':>5}  {'%':>5}  {'cost':>8}")
    for t in ORDERS[args.order](curr.tiles):
        s = reg.get(t.id)
        pct = 100 * s.n_generated / max(1, t.target_n)
        complete = s.n_generated >= t.target_n
        if complete:
            total_done += 1
        total_generated += s.n_generated
        total_cost += s.cost_usd
        marker = "✓" if complete else " "
        print(f"{marker} {t.id:53s}  {s.n_generated:>5}  {t.target_n:>5}  {pct:>4.0f}%  ${s.cost_usd:>6.3f}")
    print(
        f"\n  tiles complete: {total_done}/{len(curr.tiles)}   "
        f"examples: {total_generated}/{total_target}   "
        f"cumulative cost: ${total_cost:.3f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("curriculum", help="path to curriculum.yaml")
    ap.add_argument("--order", choices=list(ORDERS), default="dfs")
    ap.add_argument("--tag", help="filter by tag")
    ap.add_argument("--status", metavar="STATE_JSON", help="show registry status against this state file")
    ap.add_argument("--state", default="experiments/jepa_structured_data/state/registry.json",
                    help="state file path (used when --status is implied)")
    args = ap.parse_args()
    if args.status:
        args.state = args.status
        _cmd_status(args)
    else:
        _cmd_list(args)


if __name__ == "__main__":
    main()
