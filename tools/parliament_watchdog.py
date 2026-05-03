#!/usr/bin/env python3
"""Watch Parliament liveness and trigger procedural continuation when idle."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = ROOT / "runs" / "parliament" / "watchdog"
PROCEDURAL_MOTION = ROOT / "parliament" / "motions" / "procedural_bill_request.md"

sys.path.insert(0, str(ROOT))
from tools.parliament import firebase_get, firebase_put  # noqa: E402


def repo_python() -> str:
    py = ROOT / ".venv" / "bin" / "python"
    return str(py if py.exists() else Path(sys.executable))


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def latest_scheduler_summary() -> dict[str, Any] | None:
    paths = sorted((ROOT / "runs" / "parliament" / "scheduler").glob("*.summary.json"))
    if not paths:
        return None
    try:
        return json.loads(paths[-1].read_text(encoding="utf-8"))
    except Exception:
        return None


def action_status(motion_id: str) -> dict[str, Any]:
    actions = firebase_get(f"parliament/actions/{motion_id}", timeout=5.0)
    if not isinstance(actions, dict):
        return {"state": "no_actions", "actions": {}}
    running = [k for k, v in actions.items() if isinstance(v, dict) and v.get("status") == "running"]
    ready = [k for k, v in actions.items() if isinstance(v, dict) and v.get("status") == "ready"]
    completed = [k for k, v in actions.items() if isinstance(v, dict) and v.get("status") == "completed"]
    cooldown = [k for k, v in actions.items() if isinstance(v, dict) and v.get("status") in {"cooldown", "skipped_cooldown"}]
    waiting = [k for k, v in actions.items() if isinstance(v, dict) and v.get("status") == "waiting_for_votes"]
    if running:
        state = "running"
    elif ready:
        state = "ready"
    elif completed or cooldown:
        state = "completed_or_cooldown"
    else:
        state = "no_ready_actions"
    return {"state": state, "running": running, "ready": ready, "completed": completed, "cooldown": cooldown, "waiting": waiting}


def compiled_bill_status(motion_id: str) -> dict[str, Any]:
    bills = firebase_get(f"parliament/compiled_bills/{motion_id}", timeout=5.0)
    if not isinstance(bills, dict):
        return {"compiled_count": 0, "proposal_count": 0}
    return {
        "compiled_count": int(bills.get("compiled_count", 0) or 0),
        "proposal_count": int(bills.get("proposal_count", 0) or 0),
        "created_at": bills.get("created_at"),
    }


def active_research_processes() -> list[dict[str, Any]]:
    try:
        proc = subprocess.run(["ps", "axo", "pid=,etime=,command="], cwd=ROOT, capture_output=True, text=True, timeout=5)
    except Exception:
        return []
    rows = []
    needles = [
        "experiments/",
        "src/lab_platform/mamba3_lm.py",
        "tile_trainer.py",
        "orchestrator.py",
        "train_",
        "chess_competition_sweep.py",
    ]
    excludes = [
        "tools/parliament",
        "session_archiver.py",
        "http.server",
        "lab-book",
        " rg ",
    ]
    for line in proc.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if any(token in stripped for token in needles) and not any(token in stripped for token in excludes):
            parts = stripped.split(None, 2)
            if len(parts) == 3:
                rows.append({"pid": parts[0], "etime": parts[1], "command": parts[2][:500]})
    return rows


def watchdog_state(motion_id: str) -> dict[str, Any]:
    return {
        "schema": "parliament.watchdog_state.v1",
        "created_at": utc_now(),
        "motion_id": motion_id,
        "primary_action_status": action_status(motion_id),
        "procedural_action_status": action_status("procedural_bill_request"),
        "compiled_bill_status": compiled_bill_status("procedural_bill_request"),
        "active_research_processes": active_research_processes(),
    }


def should_trigger(
    summary: dict[str, Any] | None,
    actions: dict[str, Any],
    bills: dict[str, Any],
    active_jobs: list[dict[str, Any]],
    procedural_actions: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    if active_jobs:
        return False, f"active research process detected: {active_jobs[0]['command'][:140]}"
    if actions["state"] in {"running", "ready"}:
        return False, f"action_state={actions['state']}"
    if procedural_actions and (procedural_actions.get("state") in {"running", "ready"} or procedural_actions.get("waiting")):
        return False, "compiled procedural bill is awaiting vote or execution"
    if bills.get("compiled_count", 0) > 0 and actions["state"] == "no_actions":
        return False, "compiled bills exist; clerk must review them"
    if not summary:
        return True, "no scheduler summary exists"
    last_created = str(summary.get("created_at", ""))
    return True, f"idle after latest tick {last_created}; action_state={actions['state']}; compiled={bills.get('compiled_count', 0)}"


def run_procedural_chamber(args: argparse.Namespace, reason: str) -> dict[str, Any]:
    cmd = [
        repo_python(),
        str(ROOT / "tools" / "parliament.py"),
        "chamber",
        "--speakers",
        *args.speakers,
        "--backend",
        args.backend,
        "--motion",
        str(PROCEDURAL_MOTION),
        "--timeout-s",
        str(args.timeout_s),
        "--trace",
        "--firebase",
        "--firebase-prior",
        "--prior-limit",
        "4",
        "--evidence-cmd",
        "cat runs/parliament/compiled_bills/procedural_bill_request.json",
        "--evidence-cmd",
        f"{repo_python()} tools/parliament_watchdog.py --motion {args.motion} --state-only",
    ]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=args.wall_timeout_s)
    compile_cmd = [
        repo_python(),
        str(ROOT / "tools" / "parliament_bill.py"),
        "--motion",
        "procedural_bill_request",
        "--node",
        args.node,
    ]
    compile_proc = subprocess.run(compile_cmd, cwd=ROOT, capture_output=True, text=True, timeout=60)
    action_cmd = [
        repo_python(),
        str(ROOT / "tools" / "parliament_action.py"),
        "review",
        "--motion",
        "procedural_bill_request",
        "--execute",
    ]
    action_proc = subprocess.run(action_cmd, cwd=ROOT, capture_output=True, text=True, timeout=args.action_timeout_s)
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-6000:],
        "stderr_tail": proc.stderr[-3000:],
        "compile_cmd": compile_cmd,
        "compile_returncode": compile_proc.returncode,
        "compile_stdout_tail": compile_proc.stdout[-6000:],
        "compile_stderr_tail": compile_proc.stderr[-3000:],
        "action_cmd": action_cmd,
        "action_returncode": action_proc.returncode,
        "action_stdout_tail": action_proc.stdout[-6000:],
        "action_stderr_tail": action_proc.stderr[-3000:],
        "reason": reason,
    }


def run_watchdog(args: argparse.Namespace) -> dict[str, Any]:
    summary = latest_scheduler_summary()
    actions = action_status(args.motion)
    bills = compiled_bill_status("procedural_bill_request")
    procedural_actions = action_status("procedural_bill_request")
    active_jobs = active_research_processes()
    trigger, reason = should_trigger(summary, actions, bills, active_jobs, procedural_actions)
    event: dict[str, Any] = {
        "schema": "parliament.watchdog.v1",
        "created_at": utc_now(),
        "motion_id": args.motion,
        "triggered": trigger,
        "reason": reason,
        "action_status": actions,
        "compiled_bill_status": bills,
        "procedural_action_status": procedural_actions,
        "active_research_processes": active_jobs,
    }
    if trigger and args.execute:
        event["procedural_chamber"] = run_procedural_chamber(args, reason)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    (STATE_DIR / f"{stamp}.json").write_text(json.dumps(event, indent=2, ensure_ascii=False), encoding="utf-8")
    firebase_put("parliament/watchdog/latest", event, timeout=5.0)
    return event


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Parliament idle watchdog")
    parser.add_argument("--motion", default="small_lm_recovery")
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--speaker", dest="speakers", action="append", default=[])
    parser.add_argument("--timeout-s", type=int, default=120)
    parser.add_argument("--wall-timeout-s", type=int, default=270)
    parser.add_argument("--action-timeout-s", type=int, default=420)
    parser.add_argument("--node", default="m4-pro")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--state-only", action="store_true")
    args = parser.parse_args(argv)
    if args.state_only:
        print(json.dumps(watchdog_state(args.motion), indent=2, ensure_ascii=False))
        return 0
    if not args.speakers:
        args.speakers = ["claude-opposition-architect", "gpt5-ch12-chess-champion"]
    event = run_watchdog(args)
    print(json.dumps(event, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
