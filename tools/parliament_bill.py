#!/usr/bin/env python3
"""Compile Parliament-authored proposals into executable bills.

The compiler is deliberately procedural: it validates proposals supplied by
speeches and records their status. It does not invent experiments.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shlex
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs" / "parliament" / "compiled_bills"
MANIFEST_DIR = RUN_DIR / "manifests"
PROPOSAL_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{2,80}$")
DISALLOWED_SHELL = {";", "&&", "||", "|", ">", ">>", "<", "`", "$(", "${"}

sys.path.insert(0, str(ROOT))
from tools.parliament import firebase_get, firebase_put, flatten_firebase_speeches  # noqa: E402


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def validate_proposal(proposal: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = [
        "proposal_id",
        "title",
        "objective",
        "hypothesis",
        "command",
        "max_wall_s",
        "expected_artifacts",
        "kpi",
        "falsifier",
        "follow_up",
    ]
    for key in required:
        if key not in proposal:
            errors.append(f"missing {key}")
    proposal_id = str(proposal.get("proposal_id", ""))
    if not PROPOSAL_ID_RE.match(proposal_id):
        errors.append("proposal_id must be lowercase slug")
    command = str(proposal.get("command", "")).strip()
    if not command:
        errors.append("command is empty")
    if "\n" in command or "\r" in command:
        errors.append("command must be one single line; create or reuse a script instead of embedding code")
    if any(token in command for token in DISALLOWED_SHELL):
        errors.append("command contains unsupported shell control syntax")
    try:
        argv = shlex.split(command)
    except ValueError as exc:
        errors.append(f"command is not shell-parseable: {exc}")
        argv = []
    if argv:
        executable = argv[0]
        allowed_starts = {".venv/bin/python", "python", "python3"}
        if executable not in allowed_starts:
            errors.append("command must start with .venv/bin/python, python, or python3")
        for item in argv[1:]:
            if item.startswith("../") or "/../" in item:
                errors.append("command path escapes repository")
    try:
        max_wall_s = int(proposal.get("max_wall_s"))
        if max_wall_s <= 0 or max_wall_s > 300:
            errors.append("max_wall_s must be between 1 and 300")
    except Exception:
        errors.append("max_wall_s must be an integer")
    artifacts = proposal.get("expected_artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        errors.append("expected_artifacts must be a non-empty list")
    else:
        for artifact in artifacts:
            path = str(artifact)
            if path.startswith("/") or ".." in Path(path).parts:
                errors.append(f"artifact path escapes repository: {path}")
    kpi = proposal.get("kpi")
    if not isinstance(kpi, dict):
        errors.append("kpi must be an object")
    else:
        for key in ["namespace", "metric", "direction", "target"]:
            if key not in kpi:
                errors.append(f"kpi missing {key}")
        direction = str(kpi.get("direction", ""))
        if direction in {"maximize", "maximise"}:
            kpi["direction"] = "increase"
        elif direction in {"minimize", "minimise"}:
            kpi["direction"] = "decrease"
        if str(kpi.get("direction", "")) not in {"increase", "decrease", "hit"}:
            errors.append("kpi.direction must be increase, decrease, or hit")
    return errors


def proposal_from_record(record: dict[str, Any]) -> dict[str, Any] | None:
    speech = record.get("speech", {}) if isinstance(record.get("speech"), dict) else {}
    proposal = speech.get("proposal")
    return proposal if isinstance(proposal, dict) else None


def collect_proposals(motion_id: str) -> list[dict[str, Any]]:
    speeches = flatten_firebase_speeches(firebase_get(f"parliament/speeches/{motion_id}", timeout=8.0))
    by_id: dict[str, dict[str, Any]] = {}
    for record in speeches:
        proposal = proposal_from_record(record)
        if proposal:
            proposal_id = str(proposal.get("proposal_id", ""))
            item = {"speaker": record.get("speaker"), "created_at": record.get("created_at"), "proposal": proposal}
            if proposal_id not in by_id or str(record.get("created_at", "")) >= str(by_id[proposal_id].get("created_at", "")):
                by_id[proposal_id] = item
    return sorted(by_id.values(), key=lambda item: str(item.get("created_at", "")))


def compile_proposal(motion_id: str, item: dict[str, Any], node: str) -> dict[str, Any]:
    proposal = item["proposal"]
    errors = validate_proposal(proposal)
    proposal_id = str(proposal.get("proposal_id", "invalid"))
    event = {
        "schema": "parliament.compiled_bill.v1",
        "motion_id": motion_id,
        "proposal_id": proposal_id,
        "speaker": item.get("speaker"),
        "created_at": utc_now(),
        "source_created_at": item.get("created_at"),
        "proposal": proposal,
        "status": "rejected" if errors else "compiled",
        "errors": errors,
    }
    if errors:
        return event

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = MANIFEST_DIR / f"{motion_id}-{proposal_id}.json"
    manifest = [
        {
            "node": node,
            "name": f"{motion_id}-{proposal_id}",
            "cmd": str(proposal["command"]),
        }
    ]
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    event["action_spec"] = {
        "schema": "parliament.action_spec.v1",
        "motion_id": motion_id,
        "action_id": proposal_id,
        "description": proposal["title"],
        "cooldown_s": 0,
        "approval": {
            "quorum": 2,
            "min_approve": 2,
            "min_confidence": 0.6,
            "positive_positions": ["approve"],
        },
        "action": {
            "kind": "cluster_dispatch",
            "nodes": "tools/cluster/cluster_nodes.json",
            "manifest": str(manifest_path.relative_to(ROOT)),
            "per_task_timeout_s": int(proposal["max_wall_s"]),
            "wall_timeout_s": int(proposal["max_wall_s"]) + 60,
            "allowed_command_prefixes": [str(proposal["command"])],
        },
    }
    event["manifest"] = str(manifest_path.relative_to(ROOT))
    return event


def compile_motion(motion_id: str, node: str = "m4-pro") -> dict[str, Any]:
    proposals = collect_proposals(motion_id)
    compiled = [compile_proposal(motion_id, item, node) for item in proposals]
    payload = {
        "schema": "parliament.bill_compile.v1",
        "motion_id": motion_id,
        "created_at": utc_now(),
        "proposal_count": len(proposals),
        "compiled_count": sum(1 for item in compiled if item["status"] == "compiled"),
        "bills": compiled,
    }
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    out = RUN_DIR / f"{motion_id}.json"
    if out.exists():
        try:
            previous = json.loads(out.read_text(encoding="utf-8"))
        except Exception:
            previous = None
        if isinstance(previous, dict) and stable_compile_payload(previous) == stable_compile_payload(payload):
            previous["unchanged"] = True
            return previous
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    firebase_put(f"parliament/compiled_bills/{motion_id}", payload, timeout=5.0)
    return payload


def stable_compile_payload(payload: dict[str, Any]) -> dict[str, Any]:
    stable = json.loads(json.dumps(payload, sort_keys=True, ensure_ascii=False))
    stable.pop("created_at", None)
    stable.pop("unchanged", None)
    for bill in stable.get("bills", []):
        if isinstance(bill, dict):
            bill.pop("created_at", None)
    return stable


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compile Parliament proposals into executable bills")
    parser.add_argument("--motion", required=True)
    parser.add_argument("--node", default="m4-pro")
    args = parser.parse_args(argv)
    payload = compile_motion(args.motion, args.node)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
