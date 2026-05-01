"""Tile-driven corpus generator.

Walks the curriculum in a chosen order, generates target_n examples per
tile, writes JSONL under data/<tile_path>.jsonl, updates the registry.

Crash-safe: each example is a separate flushed line. Re-running picks up
where it left off (registry tracks n_generated per tile, output file is
appended).

VPN constraint: this script must run on the MacBook (where claude-eg /
Bedrock auth lives). The Mac mini cannot reach Bedrock. After each tile
completes, the script auto-syncs the new JSONL over to the mini's TB4
disk via rsync over the USB-C link, so the mini always has the
generated data ready for training.

Run:
  AWS_PROFILE=cc .venv/bin/python experiments/10_jepa_structured/tile_gen.py \\
      --order topo --limit 3        # smallest tile-system first

  AWS_PROFILE=cc .venv/bin/python experiments/10_jepa_structured/tile_gen.py \\
      --tile math.modular_arithmetic.addition_basic   # one specific tile

  AWS_PROFILE=cc .venv/bin/python experiments/10_jepa_structured/tile_gen.py \\
      --tag foundation               # only foundation-tagged tiles
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import boto3

from curriculum import (
    ORDERS,
    Curriculum,
    Tile,
    TileRegistry,
    with_tag_filter,
)
from gen_textbook import MODEL, REGION, call, parse_example, usage_cost
from probe_prompt_cache import SYSTEM_PROMPT  # noqa: F401  imported for side-effect (cache)

DEFAULT_RSYNC_DEST = "miguel-lemoss-Mac-mini.local:/Volumes/TB4/jepa_structured_data/"
EXPERIMENT_DIR = Path(__file__).resolve().parent


def generate_tile(
    client,
    tile: Tile,
    out_dir: Path,
    registry: TileRegistry,
    max_tokens: int = 600,
    per_tile_cap: int | None = None,
) -> dict:
    out_path = out_dir / tile.disk_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    status = registry.get(tile.id)
    n_existing = status.n_generated
    target = tile.target_n if per_tile_cap is None else min(tile.target_n, per_tile_cap)
    if n_existing >= target:
        return {"tile_id": tile.id, "n_new": 0, "n_bad": 0, "cost_usd": 0.0, "skipped": True}

    n_to_gen = target - n_existing
    n_ok, n_bad, cost = 0, 0, 0.0

    # Append mode: idempotent re-runs accumulate up to target_n.
    with open(out_path, "a", buffering=1) as f:
        for i in range(n_to_gen):
            user_prompt = (
                f"Generate one example (call #{n_existing + i + 1} for tile '{tile.id}'). "
                f"Topic: {tile.prompt}. Pick fresh numbers and entities not used in earlier calls."
            )
            try:
                r = call(client, user_prompt, max_tokens, temperature=1.0)
            except Exception as e:
                print(f"  [{tile.id}#{n_existing + i + 1}] API error: {e}", file=sys.stderr)
                n_bad += 1
                continue
            u = r["resp"]["usage"]
            c = usage_cost(u)
            cost += c
            text = r["resp"]["output"]["message"]["content"][0]["text"]
            ex = parse_example(text)
            if ex is None:
                n_bad += 1
                continue
            ex["_meta"] = {
                "tile_id": tile.id,
                "model": MODEL,
                "user_prompt": user_prompt,
                "cost_usd": round(c, 6),
            }
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            n_ok += 1

    status.n_generated = n_existing + n_ok
    status.last_generated_at = time.time()
    status.cost_usd += cost
    registry.set(status)

    return {"tile_id": tile.id, "n_new": n_ok, "n_bad": n_bad, "cost_usd": cost, "skipped": False}


def rsync_to_mini(out_dir: Path, dest: str, tile_relpath: str | None = None) -> bool:
    """Rsync just the data dir to the mini. If tile_relpath given, sync only
    that single file (cheap, per-tile)."""
    if tile_relpath:
        src = str(out_dir / tile_relpath)
        dst = dest.rstrip("/") + "/" + tile_relpath
        # Make sure the parent dir exists on the remote first.
        parent_remote = "/".join(dst.split("/")[:-1])
        host = dest.split(":")[0]
        remote_parent_path = parent_remote.split(":", 1)[1] if ":" in parent_remote else parent_remote
        try:
            subprocess.run(
                ["ssh", "-o", "BatchMode=yes", host, f"mkdir -p {remote_parent_path!r}"],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"  rsync: ssh mkdir failed: {e.stderr.decode().strip()}", file=sys.stderr)
            return False
    else:
        src = str(out_dir) + "/"
        dst = dest

    try:
        subprocess.run(
            ["rsync", "-a", src, dst],
            check=True,
            capture_output=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"  rsync failed: {e.stderr.decode().strip()}", file=sys.stderr)
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--curriculum",
                    default=str(EXPERIMENT_DIR / "curriculum.yaml"))
    ap.add_argument("--out",
                    default=str(EXPERIMENT_DIR / "data"),
                    help="local output dir (mirrors curriculum tree)")
    ap.add_argument("--state",
                    default=str(EXPERIMENT_DIR / "state" / "registry.json"))
    ap.add_argument("--order", choices=list(ORDERS), default="topo")
    ap.add_argument("--tag", help="filter by tag")
    ap.add_argument("--tile", help="generate one specific tile by id (overrides order/tag)")
    ap.add_argument("--limit", type=int, help="generate at most N tiles this run")
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--per-tile-cap", type=int, default=None,
                    help="cap examples per tile this run (overrides curriculum target_n; useful for smoke tests)")
    ap.add_argument("--rsync-dest", default=DEFAULT_RSYNC_DEST,
                    help="rsync target for per-tile sync, or 'none' to disable")
    args = ap.parse_args()

    curr = Curriculum.from_yaml(args.curriculum)
    registry = TileRegistry(args.state)
    out_dir = Path(args.out)

    if args.tile:
        tiles = [curr.get(args.tile)]
    else:
        it = ORDERS[args.order](curr.tiles)
        if args.tag:
            it = with_tag_filter(it, args.tag)
        tiles = list(it)
        if args.limit:
            tiles = tiles[: args.limit]

    print(
        f"# generating {len(tiles)} tile(s) with order={args.order}"
        + (f", tag={args.tag}" if args.tag else "")
        + f", rsync_dest={'(disabled)' if args.rsync_dest == 'none' else args.rsync_dest}"
    )

    client = boto3.client("bedrock-runtime", region_name=REGION)
    total_cost = 0.0
    t0 = time.time()
    for t in tiles:
        effective_target = t.target_n if args.per_tile_cap is None else min(t.target_n, args.per_tile_cap)
        if registry.get(t.id).n_generated >= effective_target:
            print(f"[skip] {t.id}  (have: {registry.get(t.id).n_generated}/{effective_target})")
            continue
        print(f"[gen]  {t.id}  target={effective_target}  prompt={t.prompt[:70]}")
        r = generate_tile(client, t, out_dir, registry, max_tokens=args.max_tokens,
                          per_tile_cap=args.per_tile_cap)
        total_cost += r["cost_usd"]
        print(f"       wrote {r['n_new']} ok / {r['n_bad']} bad   ${r['cost_usd']:.4f}")
        if args.rsync_dest != "none":
            if rsync_to_mini(out_dir, args.rsync_dest, tile_relpath=t.disk_path):
                print(f"       synced → {args.rsync_dest}{t.disk_path}")

    dt = time.time() - t0
    print(f"\n# total cost this run: ${total_cost:.4f}   elapsed: {dt:.1f}s")


if __name__ == "__main__":
    main()
