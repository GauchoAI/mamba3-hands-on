#!/usr/bin/env python3
"""Install macOS launchd jobs for Parliament events and watchdog ticks."""
from __future__ import annotations

import argparse
import os
import plistlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_LABEL = "com.gauchoai.parliament"


def plist_path(label: str) -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"


def main() -> int:
    parser = argparse.ArgumentParser(description="Install Parliament launchd jobs")
    parser.add_argument("--interval-s", type=int, default=300)
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--panel-size", type=int, default=1)
    parser.add_argument("--timeout-s", type=int, default=240)
    parser.add_argument("--wall-timeout-s", type=int, default=270)
    parser.add_argument("--persist", action="store_true")
    parser.add_argument("--archive", action="store_true")
    parser.add_argument("--execute-actions", action="store_true")
    parser.add_argument("--action-timeout-s", type=int, default=420)
    parser.add_argument("--watchdog", action="store_true")
    parser.add_argument("--watchdog-backend", default="auto")
    parser.add_argument("--event-loop", action="store_true")
    parser.add_argument("--event-listener", action="store_true", help="Install Firebase streaming listener as primary trigger")
    parser.add_argument("--watchdog-only", action="store_true", help="Install only the five-minute watchdog fallback")
    parser.add_argument("--deliberation-budget-s", type=int, default=600)
    parser.add_argument("--uninstall", action="store_true")
    args = parser.parse_args()

    if args.uninstall:
        for label in [BASE_LABEL, f"{BASE_LABEL}.events", f"{BASE_LABEL}.watchdog"]:
            path = plist_path(label)
            subprocess.run(["launchctl", "bootout", f"gui/{os.getuid()}", str(path)], check=False)
            path.unlink(missing_ok=True)
            print(f"uninstalled {path}")
        return 0

    # Launchd can hang opening Python scripts under Desktop directly on this
    # macOS setup. Run through zsh, matching the manual path that works.
    py = Path("/opt/homebrew/bin/python3")
    if not py.exists():
        py = Path(sys.executable)
    label = BASE_LABEL
    if args.event_listener:
        label = f"{BASE_LABEL}.events"
        shell_cmd = (
            f"cd {ROOT} && "
            f"{py} {ROOT / 'tools' / 'parliament_event_listener.py'} "
            f"--backend {args.watchdog_backend} "
            f"--panel-size {args.panel_size} "
            f"--speaker-timeout-s {args.timeout_s} "
            f"--action-timeout-s {args.action_timeout_s} "
            f"--deliberation-budget-s {args.deliberation_budget_s}"
        )
    elif args.watchdog_only:
        label = f"{BASE_LABEL}.watchdog"
        shell_cmd = (
            f"cd {ROOT} && "
            f"{py} {ROOT / 'tools' / 'parliament_watchdog.py'} "
            f"--motion small_lm_recovery "
            f"--backend {args.watchdog_backend} "
            f"--timeout-s {args.timeout_s} "
            f"--wall-timeout-s {args.wall_timeout_s} "
            f"--action-timeout-s {args.action_timeout_s} "
            f"--execute"
        )
    elif args.event_loop:
        shell_cmd = (
            f"cd {ROOT} && "
            f"{py} {ROOT / 'tools' / 'parliament_event_loop.py'} "
            f"--tick-backend {args.backend} "
            f"--watchdog-backend {args.watchdog_backend} "
            f"--panel-size {args.panel_size} "
            f"--tick-timeout-s {args.timeout_s} "
            f"--tick-wall-timeout-s {args.wall_timeout_s} "
            f"--action-timeout-s {args.action_timeout_s} "
            f"--deliberation-budget-s {args.deliberation_budget_s}"
        )
    else:
        shell_cmd = (
            f"cd {ROOT} && "
            f"{py} {ROOT / 'tools' / 'parliament_tick.py'} "
            f"--backend {args.backend} "
            f"--panel-size {args.panel_size} "
            f"--timeout-s {args.timeout_s} "
            f"--wall-timeout-s {args.wall_timeout_s} "
            f"--action-timeout-s {args.action_timeout_s}"
        )
        if args.persist:
            shell_cmd += " --persist"
        if args.archive:
            shell_cmd += " --archive"
        if args.execute_actions:
            shell_cmd += " --execute-actions"
        if args.watchdog:
            shell_cmd += f" --watchdog --watchdog-backend {args.watchdog_backend}"
    log_dir = ROOT / "runs" / "parliament" / "scheduler"
    log_dir.mkdir(parents=True, exist_ok=True)
    path_value = f"{Path.home() / '.local' / 'bin'}:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
    plist = {
        "Label": label,
        "ProgramArguments": ["/bin/zsh", "-lc", shell_cmd],
        "WorkingDirectory": str(ROOT),
        "RunAtLoad": True,
        "StandardOutPath": str(log_dir / "launchd.out.log"),
        "StandardErrorPath": str(log_dir / "launchd.err.log"),
        "EnvironmentVariables": {
            "PATH": path_value,
        },
    }
    if args.event_listener:
        plist["KeepAlive"] = True
    else:
        plist["StartInterval"] = args.interval_s
    path = plist_path(label)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        plistlib.dump(plist, f)
    subprocess.run(["launchctl", "bootout", f"gui/{os.getuid()}", str(path)], check=False)
    subprocess.run(["launchctl", "bootstrap", f"gui/{os.getuid()}", str(path)], check=True)
    subprocess.run(["launchctl", "kickstart", "-k", f"gui/{os.getuid()}/{label}"], check=False)
    print(f"installed {path}")
    print(f"logs: {log_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
