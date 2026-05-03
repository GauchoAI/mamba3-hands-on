#!/usr/bin/env python3
"""Bounded event loop for Parliament deliberation and action dispatch.

Firebase events are the primary wake-up path. The five-minute scheduler is only
a watchdog safety net. This loop drives one bounded chain after a wake: optional
heartbeat/tick, watchdog, bill compilation, bounded vote rounds, and action
execution if the vote passes.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = ROOT / "runs" / "parliament" / "event_loop"
LOCK_PATH = STATE_DIR / "event_loop.lock"


def repo_python() -> str:
    py = ROOT / ".venv" / "bin" / "python"
    return str(py if py.exists() else Path(sys.executable))


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def run_json(cmd: list[str], timeout_s: int) -> dict[str, Any]:
    started = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout_s)
    payload: dict[str, Any]
    try:
        payload = json.loads(proc.stdout)
    except Exception:
        payload = {}
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "wall_s": round(time.time() - started, 2),
        "payload": payload,
        "stdout_tail": proc.stdout[-6000:],
        "stderr_tail": proc.stderr[-3000:],
    }


def compiled_bill_summary() -> dict[str, Any]:
    path = ROOT / "runs" / "parliament" / "compiled_bills" / "procedural_bill_request.json"
    if not path.exists():
        return {"compiled_count": 0, "proposal_count": 0, "bills": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"compiled_count": 0, "proposal_count": 0, "bills": []}
    bills = [
        {
            "proposal_id": bill.get("proposal_id"),
            "status": bill.get("status"),
            "speaker": bill.get("speaker"),
            "title": (bill.get("proposal") or {}).get("title"),
            "command": (bill.get("proposal") or {}).get("command"),
            "kpi": (bill.get("proposal") or {}).get("kpi"),
            "errors": bill.get("errors", []),
        }
        for bill in payload.get("bills", [])
        if isinstance(bill, dict)
    ]
    return {
        "compiled_count": payload.get("compiled_count", 0),
        "proposal_count": payload.get("proposal_count", 0),
        "bills": bills,
    }


def latest_compiled_bill() -> dict[str, Any] | None:
    summary = compiled_bill_summary()
    for bill in reversed(summary.get("bills", [])):
        if bill.get("status") == "compiled":
            return bill
    return None


def vote_motion_text(round_index: int, bill: dict[str, Any], state: dict[str, Any]) -> str:
    kpi = bill.get("kpi") or {}
    return (
        "# Procedural Motion: Vote On Compiled Bill\n\n"
        f"Round: {round_index}\n\n"
        "This is a vote round, not a request to invent a new experiment.\n"
        "The full compiled bill and live watchdog state are attached as evidence.\n\n"
        f"Compiled bill id: {bill.get('proposal_id')}\n"
        f"Title: {bill.get('title')}\n"
        f"Command: {bill.get('command')}\n"
        f"KPI: {kpi.get('namespace')}/{kpi.get('metric')} {kpi.get('direction')} {kpi.get('target')}\n"
        f"Active research processes: {len(state.get('active_research_processes', []))}\n\n"
        "Vote approve if this compiled bill should run now.\n"
        "Vote reject or defer if it should not run now, and give the concrete falsifier or blocking condition.\n"
        "Vote amend only if the compiled command or schema must change before execution."
    )


def action_status(result: dict[str, Any]) -> str:
    payload = result.get("payload") or {}
    return str(payload.get("status", "unknown"))


def run_event_loop(args: argparse.Namespace) -> dict[str, Any]:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + args.deliberation_budget_s
    event: dict[str, Any] = {
        "schema": "parliament.event_loop.v1",
        "created_at": utc_now(),
        "deliberation_budget_s": args.deliberation_budget_s,
        "trigger": args.trigger,
        "steps": [],
    }

    if args.skip_tick:
        heartbeat_cmd = [repo_python(), str(ROOT / "tools" / "parliament.py"), "register-node"]
        event["steps"].append({"name": "node_heartbeat", **run_json(heartbeat_cmd, 20)})
    else:
        tick_cmd = [
            repo_python(),
            str(ROOT / "tools" / "parliament_tick.py"),
            "--backend",
            args.tick_backend,
            "--panel-size",
            str(args.panel_size),
            "--timeout-s",
            str(args.tick_timeout_s),
            "--wall-timeout-s",
            str(args.tick_wall_timeout_s),
            "--action-timeout-s",
            str(args.action_timeout_s),
            "--persist",
            "--archive",
            "--execute-actions",
        ]
        event["steps"].append(
            {"name": "tick", **run_json(tick_cmd, args.tick_wall_timeout_s + args.action_timeout_s + 60)}
        )

    state_cmd = [repo_python(), str(ROOT / "tools" / "parliament_watchdog.py"), "--motion", args.motion, "--state-only"]
    state_step = {"name": "state_before_action", **run_json(state_cmd, 30)}
    event["steps"].append(state_step)
    state = state_step.get("payload", {})

    action_cmd = [
        repo_python(),
        str(ROOT / "tools" / "parliament_action.py"),
        "review",
        "--motion",
        "procedural_bill_request",
        "--execute",
    ]
    action_step = {"name": "compiled_bill_action_review", **run_json(action_cmd, args.action_timeout_s)}
    event["steps"].append(action_step)

    if action_status(action_step) in {"no_action", "unknown"}:
        watchdog_cmd = [
            repo_python(),
            str(ROOT / "tools" / "parliament_watchdog.py"),
            "--motion",
            args.motion,
            "--backend",
            args.watchdog_backend,
            "--timeout-s",
            str(args.speaker_timeout_s),
            "--wall-timeout-s",
            str(max(30, int(deadline - time.time()))),
            "--action-timeout-s",
            str(args.action_timeout_s),
            "--execute",
        ]
        event["steps"].append({"name": "watchdog", **run_json(watchdog_cmd, max(45, int(deadline - time.time()) + args.action_timeout_s))})
        state_step = {"name": "state_after_watchdog", **run_json(state_cmd, 30)}
        event["steps"].append(state_step)
        state = state_step.get("payload", {})
        action_step = {"name": "compiled_bill_action_review_after_watchdog", **run_json(action_cmd, args.action_timeout_s)}
        event["steps"].append(action_step)

    rounds = 0
    while action_status(action_step) == "waiting_for_votes" and rounds < args.max_vote_rounds and time.time() < deadline:
        bill = latest_compiled_bill()
        if not bill:
            break
        rounds += 1
        remaining = max(45, int(deadline - time.time()))
        vote_cmd = [
            repo_python(),
            str(ROOT / "tools" / "parliament.py"),
            "chamber",
            "--speakers",
            *args.vote_speakers,
            "--backend",
            args.watchdog_backend,
            "--text",
            vote_motion_text(rounds, bill, state),
            "--motion-id",
            "procedural_bill_request",
            "--timeout-s",
            str(args.speaker_timeout_s),
            "--trace",
            "--firebase",
            "--firebase-prior",
            "--prior-limit",
            "6",
            "--evidence-cmd",
            "cat runs/parliament/compiled_bills/procedural_bill_request.json",
            "--evidence-cmd",
            f"{repo_python()} tools/parliament_watchdog.py --motion {args.motion} --state-only",
        ]
        event["steps"].append({"name": f"vote_round_{rounds}", **run_json(vote_cmd, remaining)})
        action_step = {"name": f"action_after_vote_round_{rounds}", **run_json(action_cmd, args.action_timeout_s)}
        event["steps"].append(action_step)
        if action_status(action_step) in {"completed", "failed", "running", "skipped_cooldown"}:
            break

    event["completed_at"] = utc_now()
    event["final_status"] = action_status(action_step)
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    out = STATE_DIR / f"{stamp}.json"
    out.write_text(json.dumps(event, indent=2, ensure_ascii=False), encoding="utf-8")
    event["path"] = str(out.relative_to(ROOT))
    return event


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one bounded Parliament event loop")
    parser.add_argument("--motion", default="small_lm_recovery")
    parser.add_argument("--tick-backend", default="simulated")
    parser.add_argument("--watchdog-backend", default="auto")
    parser.add_argument("--panel-size", type=int, default=1)
    parser.add_argument("--tick-timeout-s", type=int, default=45)
    parser.add_argument("--tick-wall-timeout-s", type=int, default=120)
    parser.add_argument("--speaker-timeout-s", type=int, default=120)
    parser.add_argument("--action-timeout-s", type=int, default=420)
    parser.add_argument("--deliberation-budget-s", type=int, default=600)
    parser.add_argument("--max-vote-rounds", type=int, default=2)
    parser.add_argument("--vote-speaker", dest="vote_speakers", action="append", default=[])
    parser.add_argument("--skip-tick", action="store_true", help="Do only heartbeat/state/action work; do not create a normal chamber tick")
    parser.add_argument("--trigger", default="manual", help="Short trigger label written to the event-loop record")
    args = parser.parse_args(argv)
    if not args.vote_speakers:
        args.vote_speakers = ["claude-opposition-architect", "gpt5-ch12-chess-champion"]

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps({"schema": "parliament.event_loop.v1", "status": "locked"}, indent=2))
            return 0
        event = run_event_loop(args)
    print(json.dumps(event, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
