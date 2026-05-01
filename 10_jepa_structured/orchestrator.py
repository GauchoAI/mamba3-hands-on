"""Unified CLI: generate + train, with producer/consumer coordination.

Subcommands:
  gen            generate tiles (one-shot; respects --order/--tag/--limit/--tile/--per-tile-cap)
  gen --daemon   run generator forever; drain priority queue first, then prefetch in --order
  train          walk curriculum; for each tile, wait for min_batch examples, then train
  request TILE   push a tile id onto the priority queue (manual nudge)
  status         show registry dashboard

The generator and trainer coordinate through two shared files under state/:
  registry.json            — per-tile state (n_generated, n_validated, costs, timestamps)
  priority_requests.txt    — append-only queue of tile ids the trainer wants prioritized

This is a single-machine orchestrator. For cross-machine (MacBook generator
→ Mac mini trainer) the same files just need to be rsynced periodically.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

# Lazy-import boto3 only in gen subcommand (training-only runs don't need AWS)
from curriculum import (
    ORDERS,
    Curriculum,
    Tile,
    TileRegistry,
    TileStatus,
    with_tag_filter,
)


# -------- request queue (append-only file with consume-and-truncate) --------

def queue_path(state_dir: Path) -> Path:
    return state_dir / "priority_requests.txt"


def push_request(state_dir: Path, tile_id: str) -> None:
    p = queue_path(state_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(tile_id + "\n")


def drain_requests(state_dir: Path) -> list[str]:
    """Atomically read + clear the priority queue. Returns dedup'd tile ids in order."""
    p = queue_path(state_dir)
    if not p.exists():
        return []
    tmp = p.with_suffix(".draining")
    p.rename(tmp)
    lines = [ln.strip() for ln in tmp.read_text().splitlines() if ln.strip()]
    tmp.unlink()
    seen: set[str] = set()
    out: list[str] = []
    for tid in lines:
        if tid not in seen:
            seen.add(tid)
            out.append(tid)
    return out


# -------- gen subcommand --------

def cmd_gen(args: argparse.Namespace) -> None:
    import boto3  # lazy import
    from budget import DailyBudget
    from curriculum_expander import expand_once
    from gen_textbook import REGION
    from storage_packer import (
        TB,
        disk_usage_bytes,
        pack_until_under_threshold,
    )
    from tile_gen import generate_tile, rsync_to_mini

    state_dir = Path(args.state_dir)
    out_dir = Path(args.out)
    rsync_dest = args.rsync_dest if args.rsync_dest != "none" else None
    budget = DailyBudget(state_dir / "daily_budget.json", cap_usd=args.daily_budget_usd)
    client = boto3.client("bedrock-runtime", region_name=REGION)
    storage_threshold_bytes = int(args.storage_threshold_tb * TB)

    def reload_curriculum() -> Curriculum:
        return Curriculum.from_yaml(args.curriculum)

    curr = reload_curriculum()
    registry = TileRegistry(state_dir / "registry.json")

    def gen_one(tile: Tile, source: str) -> dict:
        nonlocal registry
        registry = TileRegistry(state_dir / "registry.json")  # refresh
        if args.per_tile_cap and registry.get(tile.id).n_generated >= args.per_tile_cap:
            return {"skipped": True}
        if registry.is_generated(tile) and not args.per_tile_cap:
            return {"skipped": True}
        # Budget pre-check: estimate ~$0.005 / example as a safe upper bound for Sonnet.
        est_per_ex = 0.005
        if budget.would_exceed(est_per_ex):
            return {"skipped_budget": True}
        print(f"[gen/{source}] {tile.id}  prompt={tile.prompt[:60]}", flush=True)
        r = generate_tile(client, tile, out_dir, registry,
                          max_tokens=args.max_tokens,
                          per_tile_cap=args.per_tile_cap)
        # Record actual cost in daily budget. Token counts are best-effort
        # (generate_tile didn't surface them); we record cost only.
        budget.record(cost_usd=r["cost_usd"])
        print(f"           +{r['n_new']} ok / {r['n_bad']} bad   "
              f"${r['cost_usd']:.4f}   day_remain=${budget.remaining_usd():.2f}",
              flush=True)
        if rsync_dest and r["n_new"] > 0:
            rsync_to_mini(out_dir, rsync_dest, tile_relpath=tile.disk_path)
        return r

    def maybe_pack_storage() -> None:
        used = disk_usage_bytes(out_dir)
        if used > storage_threshold_bytes:
            print(f"[pack]     disk pressure: {used:,} > {storage_threshold_bytes:,} — packing",
                  flush=True)
            res = pack_until_under_threshold(
                Path(args.curriculum), out_dir, state_dir, storage_threshold_bytes
            )
            print(f"[pack]     packed {res['n_packed']} tiles; final={res['final_bytes']:,}",
                  flush=True)

    def maybe_expand(reason: str) -> bool:
        """Run expander when daemon would otherwise idle. Returns True if curriculum grew."""
        snapshot_before = len(reload_curriculum().tiles)
        result = expand_once(
            Path(args.curriculum),
            Path(args.curriculum).with_name("curriculum.expansions.yaml"),
            n=args.expand_n,
            budget=budget,
        )
        if result.get("skipped_budget"):
            print(f"[expand]   skipped (budget) — reason={reason}", flush=True)
            return False
        snapshot_after = len(reload_curriculum().tiles)
        added = snapshot_after - snapshot_before
        print(
            f"[expand]   reason={reason}  proposed={result.get('n_proposed', 0)}  "
            f"accepted={result.get('n_accepted', 0)}  curriculum_size: "
            f"{snapshot_before} → {snapshot_after}  ${result.get('cost_usd', 0):.4f}",
            flush=True,
        )
        return added > 0

    if args.daemon:
        prefetch_q: deque[Tile] = deque()

        def reseed_prefetch() -> None:
            curr_local = reload_curriculum()
            seq = list(ORDERS[args.order](curr_local.tiles))
            if args.tag:
                seq = [t for t in seq if args.tag in t.tags]
            prefetch_q.clear()
            prefetch_q.extend(seq)

        reseed_prefetch()
        last_pack_check = 0.0
        while True:
            # Daily budget exhausted?
            if budget.remaining_usd() <= 0:
                wait = budget.seconds_until_reset()
                print(f"[gen]      daily budget exhausted; sleeping {wait:.0f}s until UTC reset",
                      flush=True)
                time.sleep(min(wait, 3600))
                continue

            # Periodic disk-pressure check (every ~5 min in real use; honor poll_interval here)
            if time.time() - last_pack_check > 300:
                maybe_pack_storage()
                last_pack_check = time.time()

            # 1. Priority queue (trainer requests) takes precedence.
            requested = drain_requests(state_dir)
            advanced = False
            curr = reload_curriculum()
            for tid in requested:
                if tid not in curr:
                    print(f"[gen/req]  unknown tile id: {tid}", file=sys.stderr, flush=True)
                    continue
                if gen_one(curr.get(tid), "req ").get("skipped_budget"):
                    break
                advanced = True

            # 2. Prefetch in chosen order.
            if not advanced:
                while prefetch_q:
                    t = prefetch_q.popleft()
                    registry = TileRegistry(state_dir / "registry.json")
                    if registry.is_generated(t):
                        continue
                    if gen_one(t, "pre ").get("skipped_budget"):
                        break
                    advanced = True
                    break

            # 3. Curriculum exhausted → expand instead of sleep.
            if not advanced and not prefetch_q:
                grew = maybe_expand("prefetch_exhausted")
                if grew:
                    reseed_prefetch()
                    advanced = True

            if not advanced:
                # Brief poll for new priority requests; expand again if still nothing.
                time.sleep(args.poll_interval)
        return

    # one-shot mode
    if args.tile:
        tiles = [curr.get(args.tile)]
    else:
        it = ORDERS[args.order](curr.tiles)
        if args.tag:
            it = with_tag_filter(it, args.tag)
        tiles = list(it)
        if args.limit:
            tiles = tiles[: args.limit]
    for t in tiles:
        gen_one(t, "one ")


# -------- train subcommand --------

def real_train_on_tile(
    tile: Tile,
    jsonl_path: Path,
    registry: TileRegistry,
    model_root: Path,
    device: str,
    steps: int,
    batch_size: int,
    jepa_weight: float,
) -> dict:
    """Real per-tile training step. Wraps tile_trainer.train_on_tile."""
    from tile_trainer import train_on_tile as _tt
    result = _tt(
        tile, jsonl_path, registry, model_root,
        device=device, steps=steps, batch_size=batch_size, jepa_weight=jepa_weight,
    )
    msg = (
        f"  [train] {tile.id}  n={result.get('n_examples', 0)}  "
        f"step={result.get('global_step', '?')}  "
        f"ce={result.get('final_ce_loss', '?')}"
    )
    if result.get("avg_jepa_loss") is not None:
        msg += f"  jepa={result['avg_jepa_loss']}"
    msg += f"  ({result.get('elapsed_s', '?')}s on {result.get('device', '?')})"
    print(msg, flush=True)
    return result


def wait_for_min(
    registry: TileRegistry,
    tile: Tile,
    min_batch: int,
    timeout_s: float,
    poll_s: float,
    state_dir: Path,
) -> bool:
    """Block until the tile has at least min_batch examples or timeout elapses.
    Returns True on success, False on timeout."""
    push_request(state_dir, tile.id)
    deadline = time.time() + timeout_s
    last_n = -1
    while time.time() < deadline:
        # Re-read registry from disk in case generator updated it
        fresh = TileRegistry(registry.path)
        n = fresh.get(tile.id).n_generated
        if n != last_n:
            print(f"  [wait]  tile={tile.id}  have={n}/{min_batch}", flush=True)
            last_n = n
        if n >= min_batch:
            return True
        time.sleep(poll_s)
    return False


def cmd_train(args: argparse.Namespace) -> None:
    from tile_trainer import best_device

    state_dir = Path(args.state_dir)
    out_dir = Path(args.out)
    model_root = Path(args.checkpoints)
    device = args.device or best_device()

    print(f"[train] device={device}  steps_per_tile={args.steps_per_tile}  "
          f"batch={args.batch_size}  jepa_weight={args.jepa_weight}  "
          f"continuous={args.continuous}", flush=True)

    def one_pass() -> int:
        """Run one full curriculum pass; return number of tiles trained."""
        curr = Curriculum.from_yaml(args.curriculum)
        it = ORDERS[args.order](curr.tiles)
        if args.tag:
            it = with_tag_filter(it, args.tag)
        tiles = list(it)
        if args.limit:
            tiles = tiles[: args.limit]
        n_trained = 0
        for t in tiles:
            jsonl = out_dir / t.disk_path
            registry = TileRegistry(state_dir / "registry.json")
            n_have = registry.get(t.id).n_generated
            if n_have < args.min_batch:
                print(
                    f"[train] tile={t.id}  insufficient (have {n_have}/{args.min_batch}); "
                    f"requesting + waiting (timeout {args.wait_timeout}s)",
                    flush=True,
                )
                ok = wait_for_min(
                    registry, t, args.min_batch, args.wait_timeout, args.poll_interval,
                    state_dir,
                )
                if not ok:
                    print(f"[train] timeout on tile={t.id}; skipping", file=sys.stderr,
                          flush=True)
                    continue
                registry = TileRegistry(state_dir / "registry.json")
            if not jsonl.exists():
                print(f"[train] tile={t.id}  registry says {n_have} but JSONL missing; skip",
                      file=sys.stderr, flush=True)
                continue
            real_train_on_tile(
                t, jsonl, registry, model_root, device,
                steps=args.steps_per_tile, batch_size=args.batch_size,
                jepa_weight=args.jepa_weight,
            )
            n_trained += 1
        return n_trained

    if args.continuous:
        cycle = 0
        while True:
            cycle += 1
            print(f"[train] === cycle {cycle} ===", flush=True)
            n = one_pass()
            print(f"[train] cycle {cycle} done: trained {n} tiles", flush=True)
            if n == 0:
                # No tiles ready and no work done — give the generator some time
                time.sleep(args.cycle_idle_sleep)
    else:
        one_pass()


# -------- request subcommand --------

def cmd_request(args: argparse.Namespace) -> None:
    curr = Curriculum.from_yaml(args.curriculum)
    state_dir = Path(args.state_dir)
    if args.tile_id not in curr:
        print(f"unknown tile: {args.tile_id}", file=sys.stderr)
        sys.exit(2)
    push_request(state_dir, args.tile_id)
    print(f"queued: {args.tile_id}")


# -------- status subcommand --------

def cmd_status(args: argparse.Namespace) -> None:
    from budget import DailyBudget
    from storage_packer import GB, TB, disk_usage_bytes

    curr = Curriculum.from_yaml(args.curriculum)
    state_dir = Path(args.state_dir)
    registry = TileRegistry(state_dir / "registry.json")
    budget = DailyBudget(state_dir / "daily_budget.json")
    out_dir = Path(args.out)

    counts = curr.category_counts()
    total_cat = max(1, sum(counts.values()))
    print("=== curriculum balance ===")
    for cat in ("verifiable", "language_bridge", "accent"):
        n = counts.get(cat, 0)
        print(f"  {cat:18s}  {n:>3} tiles  ({100 * n / total_cat:>4.1f}%)")
    print()
    print("=== daily budget ===")
    print(f"  {json.dumps(budget.snapshot(), indent=2)}")
    print()
    print("=== storage ===")
    used = disk_usage_bytes(out_dir)
    print(f"  data dir: {out_dir}  used={used:,} bytes ({used / GB:.3f} GB)")
    print()
    print("=== per-tile state ===")
    print(f"{'  ':2s}{'tile_id':55s}  {'gen':>4}  {'tgt':>4}  {'%':>4}  "
          f"{'trained':>9}  {'acc':>5}  {'cost':>7}")
    total_done, total_target, total_gen, total_cost = 0, 0, 0, 0.0
    for t in ORDERS[args.order](curr.tiles):
        s = registry.get(t.id)
        pct = 100 * s.n_generated / max(1, t.target_n)
        complete = s.n_generated >= t.target_n
        if complete:
            total_done += 1
        total_target += t.target_n
        total_gen += s.n_generated
        total_cost += s.cost_usd
        marker = "✓" if complete else " "
        trained_recency = "—"
        if s.last_trained_at:
            age = time.time() - s.last_trained_at
            trained_recency = f"{age / 60:.0f}m ago" if age < 3600 else f"{age / 3600:.1f}h ago"
        acc_str = f"{s.student_acc:.2f}" if s.student_acc is not None else "  — "
        print(f"{marker} {t.id:53s}  {s.n_generated:>4}  {t.target_n:>4}  {pct:>3.0f}%  "
              f"{trained_recency:>9}  {acc_str}  ${s.cost_usd:>5.2f}")
    queued = []
    qp = queue_path(state_dir)
    if qp.exists():
        queued = [ln.strip() for ln in qp.read_text().splitlines() if ln.strip()]
    print(f"\n  tiles complete: {total_done}/{len(curr.tiles)}   "
          f"examples: {total_gen}/{total_target}   "
          f"cumulative cost: ${total_cost:.3f}")
    if queued:
        print(f"  pending priority requests: {len(queued)} → {queued[:5]}")


# -------- entry --------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--curriculum",
                    default="experiments/jepa_structured_data/curriculum.yaml")
    ap.add_argument("--out",
                    default="experiments/jepa_structured_data/data")
    ap.add_argument("--state-dir",
                    default="experiments/jepa_structured_data/state")

    sub = ap.add_subparsers(dest="cmd", required=True)

    # gen
    g = sub.add_parser("gen", help="generate tiles (one-shot or daemon)")
    g.add_argument("--order", choices=list(ORDERS), default="topo")
    g.add_argument("--tag")
    g.add_argument("--tile", help="generate a specific tile by id")
    g.add_argument("--limit", type=int)
    g.add_argument("--per-tile-cap", type=int)
    g.add_argument("--max-tokens", type=int, default=600)
    g.add_argument("--daemon", action="store_true",
                   help="run forever; drain priority queue first, then prefetch in --order")
    g.add_argument("--poll-interval", type=float, default=2.0)
    g.add_argument("--rsync-dest",
                   default="miguel-lemoss-Mac-mini.local:/Volumes/TB4/jepa_structured_data/")
    g.add_argument("--daily-budget-usd", type=float, default=5.0,
                   help="hard daily spend cap (USD); ~$5 ≈ 1M Sonnet input-tokens")
    g.add_argument("--storage-threshold-tb", type=float, default=3.0,
                   help="pack JSONL→parquet when total data dir exceeds this many TB")
    g.add_argument("--expand-n", type=int, default=8,
                   help="how many new tiles to propose per expansion call")
    g.set_defaults(func=cmd_gen)

    # train
    tr = sub.add_parser("train", help="walk curriculum and train; wait on empty tiles")
    tr.add_argument("--order", choices=list(ORDERS), default="topo")
    tr.add_argument("--tag")
    tr.add_argument("--limit", type=int)
    tr.add_argument("--min-batch", type=int, default=8,
                    help="minimum n_generated before training a tile begins")
    tr.add_argument("--wait-timeout", type=float, default=600.0)
    tr.add_argument("--poll-interval", type=float, default=3.0)
    tr.add_argument("--continuous", action="store_true",
                    help="loop the curriculum forever (training daemon)")
    tr.add_argument("--cycle-idle-sleep", type=float, default=30.0,
                    help="seconds to sleep when a full cycle had zero trainable tiles")
    tr.add_argument("--checkpoints",
                    default="experiments/jepa_structured_data/checkpoints",
                    help="directory for student.pt")
    tr.add_argument("--device", default=None,
                    help="torch device (mps/cuda/cpu); auto-detect if omitted")
    tr.add_argument("--steps-per-tile", type=int, default=80,
                    help="optimizer steps per tile per cycle")
    tr.add_argument("--batch-size", type=int, default=8)
    tr.add_argument("--jepa-weight", type=float, default=0.0,
                    help="weight on paraphrase-invariance JEPA loss; 0 = byte-CE only")
    tr.set_defaults(func=cmd_train)

    # request
    req = sub.add_parser("request", help="push a tile id onto the priority queue")
    req.add_argument("tile_id")
    req.set_defaults(func=cmd_request)

    # status
    st = sub.add_parser("status", help="show registry + queue dashboard")
    st.add_argument("--order", choices=list(ORDERS), default="topo")
    st.set_defaults(func=cmd_status)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
