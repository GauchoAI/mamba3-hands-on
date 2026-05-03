#!/usr/bin/env python3
"""Install a macOS launchd schedule for Parliament ticks."""
from __future__ import annotations

import argparse
import os
import plistlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABEL = "com.gauchoai.parliament"
PLIST_PATH = Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def main() -> int:
    parser = argparse.ArgumentParser(description="Install Parliament five-minute launchd schedule")
    parser.add_argument("--interval-s", type=int, default=300)
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--panel-size", type=int, default=1)
    parser.add_argument("--timeout-s", type=int, default=240)
    parser.add_argument("--wall-timeout-s", type=int, default=270)
    parser.add_argument("--persist", action="store_true")
    parser.add_argument("--archive", action="store_true")
    parser.add_argument("--execute-actions", action="store_true")
    parser.add_argument("--action-timeout-s", type=int, default=420)
    parser.add_argument("--uninstall", action="store_true")
    args = parser.parse_args()

    if args.uninstall:
        subprocess.run(["launchctl", "bootout", f"gui/{os.getuid()}", str(PLIST_PATH)], check=False)
        PLIST_PATH.unlink(missing_ok=True)
        print(f"uninstalled {PLIST_PATH}")
        return 0

    # Launchd can hang opening Python scripts under Desktop directly on this
    # macOS setup. Run through zsh, matching the manual path that works.
    py = Path("/opt/homebrew/bin/python3")
    if not py.exists():
        py = Path(sys.executable)
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
    log_dir = ROOT / "runs" / "parliament" / "scheduler"
    log_dir.mkdir(parents=True, exist_ok=True)
    path_value = f"{Path.home() / '.local' / 'bin'}:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
    plist = {
        "Label": LABEL,
        "ProgramArguments": ["/bin/zsh", "-lc", shell_cmd],
        "WorkingDirectory": str(ROOT),
        "StartInterval": args.interval_s,
        "RunAtLoad": True,
        "StandardOutPath": str(log_dir / "launchd.out.log"),
        "StandardErrorPath": str(log_dir / "launchd.err.log"),
        "EnvironmentVariables": {
            "PATH": path_value,
        },
    }
    PLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PLIST_PATH.open("wb") as f:
        plistlib.dump(plist, f)
    subprocess.run(["launchctl", "bootout", f"gui/{os.getuid()}", str(PLIST_PATH)], check=False)
    subprocess.run(["launchctl", "bootstrap", f"gui/{os.getuid()}", str(PLIST_PATH)], check=True)
    subprocess.run(["launchctl", "kickstart", "-k", f"gui/{os.getuid()}/{LABEL}"], check=False)
    print(f"installed {PLIST_PATH}")
    print(f"logs: {log_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
