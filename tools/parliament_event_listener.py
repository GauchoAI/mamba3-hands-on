#!/usr/bin/env python3
"""Listen to Firebase Parliament events and run the bounded event loop.

This is the low-latency path. The launchd watchdog remains useful only as a
fallback for missed streams, crashed listeners, or silent idle states.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = ROOT / "runs" / "parliament" / "event_listener"

sys.path.insert(0, str(ROOT))
from tools.parliament import FIREBASE_URL  # noqa: E402


def repo_python() -> str:
    py = ROOT / ".venv" / "bin" / "python"
    return str(py if py.exists() else Path(sys.executable))


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def append_event(record: dict[str, Any]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    day = dt.datetime.now(dt.UTC).strftime("%Y%m%d")
    path = STATE_DIR / f"{day}.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def firebase_stream(path: str, timeout_s: int):
    url = f"{FIREBASE_URL.rstrip('/')}/{path.strip('/')}.json"
    req = urllib.request.Request(url, headers={"Accept": "text/event-stream", "User-Agent": "parliament-listener/1.0"})
    with urllib.request.urlopen(req, timeout=timeout_s) as response:
        event_name = "message"
        data_lines: list[str] = []
        for raw in response:
            line = raw.decode("utf-8", errors="replace").rstrip("\n")
            if not line:
                if data_lines:
                    data = "\n".join(data_lines)
                    yield event_name, data
                event_name = "message"
                data_lines = []
                continue
            if line.startswith(":"):
                continue
            if line.startswith("event:"):
                event_name = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data_lines.append(line.split(":", 1)[1].strip())


def event_key(payload: Any) -> str:
    try:
        return json.dumps(payload, sort_keys=True, ensure_ascii=False)[:4000]
    except Exception:
        return str(payload)[:4000]


def should_trigger(payload: Any, initial_seen: bool) -> tuple[bool, str]:
    if not isinstance(payload, dict):
        return False, "non-json"
    path = str(payload.get("path", ""))
    data = payload.get("data")
    if not initial_seen and path == "/":
        return False, "initial snapshot"
    if data is None:
        return False, "delete-or-empty"
    if path.startswith("/watchdog/latest"):
        return False, "watchdog mirror"
    if path.startswith("/nodes/"):
        return False, "node heartbeat"
    if path.startswith("/actions/") and isinstance(data, dict) and data.get("status") == "skipped_cooldown":
        return False, "terminal action cooldown"
    if path.startswith("/speeches/") or path.startswith("/compiled_bills/") or path.startswith("/actions/"):
        return True, path or "/"
    return False, f"ignored path {path or '/'}"


def run_event_loop(args: argparse.Namespace, trigger: str) -> dict[str, Any]:
    cmd = [
        repo_python(),
        str(ROOT / "tools" / "parliament_event_loop.py"),
        "--skip-tick",
        "--trigger",
        trigger[:120],
        "--watchdog-backend",
        args.backend,
        "--panel-size",
        str(args.panel_size),
        "--speaker-timeout-s",
        str(args.speaker_timeout_s),
        "--action-timeout-s",
        str(args.action_timeout_s),
        "--deliberation-budget-s",
        str(args.deliberation_budget_s),
        "--max-vote-rounds",
        str(args.max_vote_rounds),
    ]
    started = time.time()
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=args.deliberation_budget_s + args.action_timeout_s + 90,
    )
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a Firebase-driven Parliament event listener")
    parser.add_argument("--path", default="parliament")
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--panel-size", type=int, default=1)
    parser.add_argument("--speaker-timeout-s", type=int, default=120)
    parser.add_argument("--action-timeout-s", type=int, default=420)
    parser.add_argument("--deliberation-budget-s", type=int, default=600)
    parser.add_argument("--max-vote-rounds", type=int, default=2)
    parser.add_argument("--stream-timeout-s", type=int, default=3600)
    parser.add_argument("--debounce-s", type=float, default=2.0)
    parser.add_argument("--min-loop-interval-s", type=float, default=15.0)
    parser.add_argument("--run-at-start", action="store_true")
    args = parser.parse_args(argv)

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    initial_seen = False
    last_key = ""
    last_loop_at = 0.0
    if args.run_at_start:
        result = run_event_loop(args, "listener-start")
        append_event({"schema": "parliament.event_listener.v1", "created_at": utc_now(), "trigger": "listener-start", "result": result})
        last_loop_at = time.time()

    while True:
        try:
            for event_name, raw_data in firebase_stream(args.path, args.stream_timeout_s):
                try:
                    payload = json.loads(raw_data)
                except Exception:
                    append_event({"schema": "parliament.event_listener.v1", "created_at": utc_now(), "event": event_name, "ignored": "bad-json"})
                    continue
                trigger, reason = should_trigger(payload, initial_seen)
                initial_seen = True
                key = event_key(payload)
                if key == last_key:
                    continue
                last_key = key
                firebase_path = payload.get("path") if isinstance(payload, dict) else None
                append_event(
                    {
                        "schema": "parliament.event_listener.v1",
                        "created_at": utc_now(),
                        "event": event_name,
                        "firebase_path": firebase_path,
                        "trigger": trigger,
                        "reason": reason,
                    }
                )
                if not trigger:
                    continue
                now = time.time()
                wait_s = max(0.0, args.min_loop_interval_s - (now - last_loop_at), args.debounce_s)
                if wait_s:
                    time.sleep(wait_s)
                result = run_event_loop(args, f"firebase:{reason}")
                append_event(
                    {
                        "schema": "parliament.event_listener.v1",
                        "created_at": utc_now(),
                        "event": "event_loop",
                        "firebase_path": firebase_path,
                        "trigger_reason": reason,
                        "result": result,
                    }
                )
                last_loop_at = time.time()
        except KeyboardInterrupt:
            return 130
        except Exception as exc:  # noqa: BLE001
            append_event(
                {
                    "schema": "parliament.event_listener.v1",
                    "created_at": utc_now(),
                    "event": "stream_error",
                    "reason": str(exc)[-1000:],
                }
            )
            time.sleep(5)


if __name__ == "__main__":
    raise SystemExit(main())
