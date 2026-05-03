#!/usr/bin/env python3
"""Run one bounded Parliament scheduler tick.

This is intentionally small and conservative:
- one tick selects a rotating panel, not the whole Parliament
- dry-run traces are written by tools/parliament.py
- a lock prevents overlapping ticks if a backend is slow
- results are summarized to runs/parliament/scheduler/ for inspection
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

ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = ROOT / "runs" / "parliament" / "scheduler"
STATE_PATH = STATE_DIR / "state.json"
LOCK_PATH = STATE_DIR / "tick.lock"
DEFAULT_MOTION = ROOT / "parliament" / "motions" / "small_lm_recovery.md"
DEFAULT_PANEL = [
    "gpt5-ch12-chess-champion",
    "claude-hanoi-lego-puzzle-solver",
    "claude-cortex-primitive-owner",
    "claude-language-jepa-owner",
    "claude-phi-composition-owner",
    "claude-platform-kappa-clerk",
    "claude-opposition-architect",
]


def repo_python() -> str:
    py = ROOT / ".venv" / "bin" / "python"
    return str(py if py.exists() else Path(sys.executable))


def utc_stamp() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def load_state() -> dict:
    if not STATE_PATH.exists():
        return {"tick": 0}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"tick": 0}


def save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def choose_panel(tick: int, speakers: list[str], panel_size: int) -> list[str]:
    if panel_size >= len(speakers):
        return list(speakers)
    start = (tick * panel_size) % len(speakers)
    return [speakers[(start + i) % len(speakers)] for i in range(panel_size)]


def summarize_chamber(payload: dict) -> dict:
    summary = []
    for record in payload.get("speeches", []):
        speech = record.get("speech", {})
        summary.append(
            {
                "speaker": record.get("speaker"),
                "backend": record.get("backend"),
                "position": speech.get("position"),
                "confidence": speech.get("confidence"),
                "body": str(speech.get("body", ""))[:900],
                "trace_path": record.get("trace_path"),
            }
        )
    return {
        "schema": "parliament.scheduler_tick.v1",
        "created_at": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "dry_run": payload.get("dry_run", True),
        "speeches": summary,
    }


def run_tick(args: argparse.Namespace) -> int:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("parliament tick skipped: previous tick still running", flush=True)
            return 0

        state = load_state()
        tick = int(state.get("tick", 0))
        speakers = args.speaker or DEFAULT_PANEL
        panel = choose_panel(tick, speakers, args.panel_size)
        stamp = utc_stamp()
        raw_path = STATE_DIR / f"{stamp}-tick-{tick:04d}.json"
        summary_path = STATE_DIR / f"{stamp}-tick-{tick:04d}.summary.json"

        cmd = [
            repo_python(),
            str(ROOT / "tools" / "parliament.py"),
            "chamber",
            "--speakers",
            *panel,
            "--backend",
            args.backend,
            "--motion",
            str(args.motion),
            "--dry-run",
            "--timeout-s",
            str(args.timeout_s),
        ]
        t0 = time.time()
        try:
            proc = subprocess.run(
                cmd,
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=args.wall_timeout_s,
            )
            stdout = proc.stdout
            stderr = proc.stderr
            returncode = proc.returncode
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            returncode = 124
        wall_s = round(time.time() - t0, 2)
        raw_path.write_text(stdout, encoding="utf-8")

        result = {
            "tick": tick,
            "panel": panel,
            "cmd": cmd,
            "returncode": returncode,
            "wall_s": wall_s,
            "raw_path": str(raw_path.relative_to(ROOT)),
            "stderr_tail": str(stderr)[-4000:],
        }
        if returncode == 0:
            payload = json.loads(stdout)
            result.update(summarize_chamber(payload))
        else:
            result["schema"] = "parliament.scheduler_tick.v1"
            result["created_at"] = dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

        state["tick"] = tick + 1
        state["last_summary"] = str(summary_path.relative_to(ROOT))
        state["last_panel"] = panel
        state["last_returncode"] = returncode
        state["updated_at"] = result["created_at"]
        save_state(state)

        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one bounded Parliament scheduler tick")
    parser.add_argument("--motion", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--panel-size", type=int, default=1)
    parser.add_argument("--timeout-s", type=int, default=240, help="Per-speaker backend timeout")
    parser.add_argument("--wall-timeout-s", type=int, default=270, help="Whole tick timeout")
    parser.add_argument("--speaker", action="append", help="Override rotating speaker list")
    args = parser.parse_args(argv)
    return run_tick(args)


if __name__ == "__main__":
    raise SystemExit(main())
