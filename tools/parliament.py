#!/usr/bin/env python3
"""Parliament CLI — dry-run multi-model deliberation for the lab.

The implementation is deliberately thin:
- dry runs inspect the repo and emit trace files, but do not append durable logs
- durable append is explicit and local-first
- model backends are command adapters, so Codex, Claude, or Symphony can be
  plugged in without changing the protocol
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import shlex
import socket
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - dependency is declared in pyproject
    yaml = None

ROOT = Path(__file__).resolve().parents[1]
PARLIAMENT_DIR = ROOT / "parliament"
IDENTITIES_DIR = PARLIAMENT_DIR / "identities"
PROMPTS_DIR = PARLIAMENT_DIR / "prompts"
LOG_PATH = PARLIAMENT_DIR / "log.jsonl"
DRY_RUN_DIR = ROOT / "runs" / "parliament" / "dry_runs"
FIREBASE_URL = "https://signaling-dcfad-default-rtdb.europe-west1.firebasedatabase.app"


@dataclass
class CommandResult:
    cmd: list[str]
    returncode: int
    stdout: str
    stderr: str


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stable_id(text: str, prefix: str = "motion") -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{digest}"


def load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("pyyaml is required for Parliament identity files")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def load_identity(speaker: str) -> tuple[Path, dict[str, Any]]:
    path = IDENTITIES_DIR / f"{speaker}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"identity not found: {path}")
    return path, load_yaml(path)


def load_motion(path: Path | None, text: str | None) -> dict[str, Any]:
    if path:
        raw = path.read_text(encoding="utf-8")
        motion_id = path.stem
        body = raw
    else:
        body = text or "Review the current repository direction."
        motion_id = stable_id(body)
    return {"motion_id": motion_id, "body": body.strip(), "source": str(path) if path else "inline"}


def run_cmd(cmd: list[str], timeout: int = 20) -> CommandResult:
    try:
        p = subprocess.run(
            cmd,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return CommandResult(cmd, p.returncode, p.stdout[-6000:], p.stderr[-3000:])
    except Exception as exc:
        return CommandResult(cmd, 125, "", str(exc))


def collect_evidence(extra_commands: list[str] | None = None) -> list[dict[str, Any]]:
    commands = [
        ["git", "status", "--short", "--branch"],
        ["git", "log", "--oneline", "-5"],
        ["rg", "-n", "KPI|checkpoint|human playtest|online_top12|Parliament", "experiments/12_chess_experts", "docs", "parliament"],
    ]
    for raw in extra_commands or []:
        commands.append(shlex.split(raw))

    evidence: list[dict[str, Any]] = []
    for cmd in commands:
        res = run_cmd(cmd)
        evidence.append(
            {
                "cmd": res.cmd,
                "returncode": res.returncode,
                "stdout": res.stdout,
                "stderr": res.stderr,
            }
        )
    return evidence


def evidence_summary(evidence: list[dict[str, Any]]) -> list[str]:
    refs: list[str] = []
    for item in evidence:
        cmd = " ".join(item["cmd"])
        if item["returncode"] == 0:
            refs.append(f"`{cmd}`")
        else:
            refs.append(f"`{cmd}` failed with {item['returncode']}")
    return refs


def build_prompt(identity: dict[str, Any], motion: dict[str, Any], prior_speeches: list[dict[str, Any]], evidence: list[dict[str, Any]], dry_run: bool) -> str:
    judge_prompt = (PROMPTS_DIR / "judge.md").read_text(encoding="utf-8")
    payload = {
        "identity": identity,
        "motion": motion,
        "prior_speeches": prior_speeches,
        "evidence": evidence,
        "dry_run": dry_run,
    }
    return judge_prompt + "\n\nContext JSON:\n" + json.dumps(payload, indent=2, ensure_ascii=False)


def simulated_backend(identity: dict[str, Any], motion: dict[str, Any], evidence: list[dict[str, Any]], prior_speeches: list[dict[str, Any]]) -> dict[str, Any]:
    speaker = identity.get("speaker", "unknown")
    body = motion.get("body", "")
    lower = body.lower()
    if "running" in lower and "checkpoint" not in lower and "kpi" not in lower:
        return {
            "kind": "silence",
            "position": "defer",
            "body": "No checkpoint, KPI change, failure, or publication event is present. Parliament should stay silent while work is merely in progress.",
            "evidence": evidence_summary(evidence),
            "prediction": "A later checkpoint or benchmark result will provide a concrete object to inspect.",
            "falsifier": "If an unreported failure or KPI change exists, this silence was too conservative.",
            "confidence": 0.86,
        }

    stance = identity.get("stance", "")
    critique = "The next useful step is to demand an inspectable benchmark artifact, not just a narrative claim."
    if prior_speeches:
        critique = "I read the prior speech and would require its claim to be tied to a measurable KPI before promotion."
    return {
        "kind": "position",
        "position": "amend",
        "body": f"{speaker}: {critique} {stance}",
        "evidence": evidence_summary(evidence),
        "prediction": "If this direction is correct, the next run will produce a checkpoint, a KPI, and a public artifact that can be reviewed independently.",
        "falsifier": "If repo inspection shows no reproducible command, no checkpoint lineage, or no KPI movement, the motion should not advance.",
        "confidence": 0.74,
    }


def command_backend(command: str, prompt: str, timeout_s: int) -> dict[str, Any]:
    env = os.environ.copy()
    env["PARLIAMENT_PROMPT"] = prompt
    proc = subprocess.run(
        shlex.split(command),
        cwd=ROOT,
        input=prompt,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        env=env,
    )
    raw = proc.stdout.strip()
    if proc.returncode != 0:
        raise RuntimeError(f"backend failed: {proc.stderr[-2000:]}")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"backend did not return JSON: {raw[:1000]}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("backend JSON must be an object")
    return parsed


def validate_speech(speech: dict[str, Any]) -> dict[str, Any]:
    required = ["kind", "position", "body", "evidence", "prediction", "falsifier", "confidence"]
    missing = [k for k in required if k not in speech]
    if missing:
        raise ValueError(f"speech missing required fields: {', '.join(missing)}")
    if not isinstance(speech["evidence"], list):
        raise ValueError("speech.evidence must be a list")
    try:
        confidence = float(speech["confidence"])
    except Exception as exc:
        raise ValueError("speech.confidence must be numeric") from exc
    speech["confidence"] = max(0.0, min(1.0, confidence))
    return speech


def wrap_speech(
    speech: dict[str, Any],
    identity_path: Path,
    identity: dict[str, Any],
    motion: dict[str, Any],
    dry_run: bool,
    backend: str,
    trace_path: Path | None,
) -> dict[str, Any]:
    node_id = f"{socket.gethostname().lower().replace('.', '-')}-{platform.machine().lower()}"
    return {
        "schema": "parliament.speech.v1",
        "created_at": utc_now(),
        "motion_id": motion["motion_id"],
        "motion_source": motion["source"],
        "speaker": identity["speaker"],
        "role": identity.get("role", "judge"),
        "model_family": identity.get("model_family", ""),
        "identity_ref": str(identity_path.relative_to(ROOT)),
        "node_id": node_id,
        "backend": backend,
        "dry_run": dry_run,
        "trace_path": str(trace_path.relative_to(ROOT)) if trace_path else None,
        "speech": speech,
    }


def make_trace_path(record: dict[str, Any]) -> Path:
    DRY_RUN_DIR.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    return DRY_RUN_DIR / f"{stamp}-{record['motion_id']}-{record['speaker']}.json"


def write_trace(path: Path, record: dict[str, Any], prompt: str, evidence: list[dict[str, Any]]) -> None:
    payload = {
        "record": record,
        "prompt": prompt,
        "evidence": evidence,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def append_log(record: dict[str, Any]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def firebase_post(path: str, data: dict[str, Any], timeout: float = 5.0) -> str | None:
    import urllib.request

    url = f"{FIREBASE_URL.rstrip('/')}/{path.strip('/')}.json"
    req = urllib.request.Request(
        url,
        data=json.dumps(data, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", "User-Agent": "parliament/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8")).get("name")
    except Exception:
        return None


def run_speaker(args: argparse.Namespace, prior_speeches: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    identity_path, identity = load_identity(args.speaker)
    motion = load_motion(Path(args.motion) if args.motion else None, args.text)
    evidence = collect_evidence(args.evidence_cmd or [])
    prompt = build_prompt(identity, motion, prior_speeches or [], evidence, args.dry_run)

    backend_name = args.backend
    if args.backend == "simulated":
        speech = simulated_backend(identity, motion, evidence, prior_speeches or [])
    else:
        command = args.command or os.environ.get(f"PARLIAMENT_BACKEND_{args.backend.upper()}_CMD")
        if not command:
            raise RuntimeError(f"backend {args.backend!r} needs --command or PARLIAMENT_BACKEND_{args.backend.upper()}_CMD")
        speech = command_backend(command, prompt, args.timeout_s)
    speech = validate_speech(speech)

    record = wrap_speech(speech, identity_path, identity, motion, args.dry_run, backend_name, None)
    trace_path = make_trace_path(record) if args.dry_run or args.trace else None
    if trace_path:
        record["trace_path"] = str(trace_path.relative_to(ROOT))
        write_trace(trace_path, record, prompt, evidence)

    if not args.dry_run:
        append_log(record)
        if args.firebase:
            firebase_post(f"parliament/speeches/{record['motion_id']}/{record['speaker']}", record)

    return record


def cmd_speak(args: argparse.Namespace) -> None:
    record = run_speaker(args)
    print(json.dumps(record, indent=2, ensure_ascii=False))


def cmd_chamber(args: argparse.Namespace) -> None:
    speeches: list[dict[str, Any]] = []
    for speaker in args.speakers:
        speaker_args = argparse.Namespace(**vars(args))
        speaker_args.speaker = speaker
        record = run_speaker(speaker_args, speeches)
        speeches.append(record)
    print(json.dumps({"schema": "parliament.chamber.v1", "dry_run": args.dry_run, "speeches": speeches}, indent=2, ensure_ascii=False))


def classify_event(event: dict[str, Any]) -> dict[str, Any]:
    text = json.dumps(event, ensure_ascii=False).lower()
    trigger_reasons: list[str] = []
    checkpoint = event.get("checkpoint") or event.get("checkpoint_path") or event.get("checkpoint_id")
    if checkpoint:
        trigger_reasons.append("checkpoint")
    if event.get("kpi") is not None or event.get("kpi_delta") is not None:
        trigger_reasons.append("kpi")
    if event.get("benchmark") or event.get("benchmark_result"):
        trigger_reasons.append("benchmark")
    if event.get("playtest"):
        trigger_reasons.append("playtest")
    if event.get("publication") or event.get("publish"):
        trigger_reasons.append("publication")
    if event.get("motion") or event.get("motion_id"):
        trigger_reasons.append("motion")
    if event.get("retrospective"):
        trigger_reasons.append("retrospective")
    if str(event.get("status", "")).lower() in {"failed", "passed"}:
        trigger_reasons.append(f"status={event['status']}")

    running_only = any(word in text for word in ["running", "training", "in_progress", "heartbeat"]) and not trigger_reasons
    if running_only:
        return {
            "decision": "silent",
            "reason": "training is in progress without a checkpoint, KPI change, benchmark result, or explicit motion",
        }
    if trigger_reasons:
        return {"decision": "deliberate", "reason": "event contains inspectable evidence: " + ", ".join(trigger_reasons)}
    return {"decision": "ignore", "reason": "event is not relevant to Parliament"}


def cmd_event(args: argparse.Namespace) -> None:
    event = json.loads(args.event)
    print(json.dumps(classify_event(event), indent=2, ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Parliament deliberation CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--motion", help="Path to a motion markdown file")
    common.add_argument("--text", help="Inline motion text")
    common.add_argument("--backend", default="simulated", help="simulated, codex, claude, symphony, or another command adapter")
    common.add_argument("--command", help="Command backend. Receives prompt on stdin and PARLIAMENT_PROMPT.")
    common.add_argument("--timeout-s", type=int, default=300)
    common.add_argument("--dry-run", action="store_true", help="Do not append durable logs or Firebase")
    common.add_argument("--trace", action="store_true", help="Write a trace even for non-dry runs")
    common.add_argument("--firebase", action="store_true", help="Also append short record to Firebase")
    common.add_argument("--evidence-cmd", action="append", help="Extra command to run as evidence")

    p_speak = sub.add_parser("speak", parents=[common], help="Run one speaker")
    p_speak.add_argument("--speaker", required=True)
    p_speak.set_defaults(func=cmd_speak)

    p_chamber = sub.add_parser("chamber", parents=[common], help="Run multiple speakers sequentially")
    p_chamber.add_argument("--speakers", nargs="+", required=True)
    p_chamber.set_defaults(func=cmd_chamber)

    p_event = sub.add_parser("event", help="Classify whether an event should trigger deliberation")
    p_event.add_argument("--event", required=True, help="JSON event")
    p_event.set_defaults(func=cmd_event)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
