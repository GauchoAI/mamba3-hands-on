#!/usr/bin/env python3
"""Execute allowlisted actions after Parliament reaches a vote threshold."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ACTION_DIR = ROOT / "parliament" / "actions"
RUN_DIR = ROOT / "runs" / "parliament" / "actions"
HISTORY_DIR = RUN_DIR / "history"

sys.path.insert(0, str(ROOT))
from tools.parliament import firebase_get, firebase_put  # noqa: E402


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_action_spec(motion_id: str) -> dict[str, Any] | None:
    path = ACTION_DIR / f"{motion_id}.json"
    if not path.exists():
        compiled = firebase_get(f"parliament/compiled_bills/{motion_id}", timeout=5.0)
        if isinstance(compiled, dict):
            bills = compiled.get("bills", [])
            if isinstance(bills, list):
                for bill in reversed(bills):
                    if isinstance(bill, dict) and bill.get("status") == "compiled" and isinstance(bill.get("action_spec"), dict):
                        return bill["action_spec"]
        return None
    spec = json.loads(path.read_text(encoding="utf-8"))
    if spec.get("motion_id") != motion_id:
        raise ValueError(f"{path} motion_id does not match {motion_id!r}")
    return spec


def flatten_firebase_speeches(payload: Any) -> list[dict[str, Any]]:
    speeches: list[dict[str, Any]] = []
    if not isinstance(payload, dict):
        return speeches
    for speaker_bucket in payload.values():
        if isinstance(speaker_bucket, dict):
            for record in speaker_bucket.values():
                if isinstance(record, dict):
                    speeches.append(record)
    return speeches


def latest_vote_by_speaker(speeches: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for record in sorted(speeches, key=lambda r: str(r.get("created_at", ""))):
        speaker = str(record.get("speaker", ""))
        if speaker:
            latest[speaker] = record
    return latest


def tally_votes(speeches: list[dict[str, Any]], approval: dict[str, Any]) -> dict[str, Any]:
    positive = set(approval.get("positive_positions", ["approve"]))
    min_confidence = float(approval.get("min_confidence", 0.0))
    latest = latest_vote_by_speaker(speeches)
    votes = []
    for record in latest.values():
        speech = record.get("speech", {}) if isinstance(record.get("speech"), dict) else {}
        position = str(speech.get("position", "abstain"))
        confidence = float(speech.get("confidence", 0.0) or 0.0)
        accepted = position in positive and confidence >= min_confidence
        votes.append(
            {
                "speaker": record.get("speaker"),
                "position": position,
                "confidence": confidence,
                "accepted": accepted,
            }
        )
    approvals = sum(1 for vote in votes if vote["accepted"])
    quorum = int(approval.get("quorum", 1))
    min_approve = int(approval.get("min_approve", quorum))
    return {
        "speakers": len(votes),
        "approvals": approvals,
        "quorum": quorum,
        "min_approve": min_approve,
        "approved": len(votes) >= quorum and approvals >= min_approve,
        "votes": sorted(votes, key=lambda v: str(v.get("speaker", ""))),
    }


def local_event_path(motion_id: str, action_id: str) -> Path:
    return RUN_DIR / f"{motion_id}-{action_id}.json"


def historical_event_paths(motion_id: str, action_id: str) -> list[Path]:
    return sorted(HISTORY_DIR.glob(f"{motion_id}-{action_id}-*.json"))


def completed_event_from_event_loops(motion_id: str, action_id: str) -> dict[str, Any] | None:
    paths = sorted((ROOT / "runs" / "parliament" / "event_loop").glob("*.json"), reverse=True)
    for path in paths[:50]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for step in reversed(payload.get("steps", [])):
            event = step.get("payload") if isinstance(step, dict) else None
            if (
                isinstance(event, dict)
                and event.get("motion_id") == motion_id
                and event.get("action_id") == action_id
                and event.get("status") in {"completed", "cooldown", "skipped_cooldown"}
            ):
                return event
    return None


def best_previous_event(motion_id: str, action_id: str, candidates: list[dict[str, Any] | None]) -> dict[str, Any] | None:
    usable = [event for event in candidates if isinstance(event, dict)]
    durable = [event for event in usable if event.get("status") in {"completed", "cooldown", "skipped_cooldown"}]
    if durable:
        return sorted(durable, key=lambda e: float(cooldown_completed_epoch(e) or 0.0))[-1]
    return usable[0] if usable else None


def load_previous_event(motion_id: str, action_id: str) -> dict[str, Any] | None:
    candidates: list[dict[str, Any] | None] = []
    remote = firebase_get(f"parliament/actions/{motion_id}/{action_id}", timeout=5.0)
    if isinstance(remote, dict):
        candidates.append(remote)
    path = local_event_path(motion_id, action_id)
    if path.exists():
        try:
            candidates.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            pass
    for history_path in historical_event_paths(motion_id, action_id)[-20:]:
        try:
            candidates.append(json.loads(history_path.read_text(encoding="utf-8")))
        except Exception:
            pass
    candidates.append(completed_event_from_event_loops(motion_id, action_id))
    return best_previous_event(motion_id, action_id, candidates)


def in_cooldown(spec: dict[str, Any], previous: dict[str, Any] | None) -> bool:
    if not previous or previous.get("status") not in {"completed", "cooldown", "skipped_cooldown"}:
        return False
    cooldown_s = int(spec.get("cooldown_s", 0) or 0)
    if cooldown_s <= 0:
        return True
    completed_at = float(previous.get("completed_at_epoch", 0.0) or 0.0)
    if not completed_at and isinstance(previous.get("previous"), dict):
        completed_at = float(previous["previous"].get("completed_at_epoch", 0.0) or 0.0)
    return bool(completed_at and time.time() - completed_at < cooldown_s)


def cooldown_completed_epoch(previous: dict[str, Any] | None) -> float | None:
    if not previous:
        return None
    completed_at = float(previous.get("completed_at_epoch", 0.0) or 0.0)
    if completed_at:
        return completed_at
    if isinstance(previous.get("previous"), dict):
        return cooldown_completed_epoch(previous["previous"])
    return None


def compact_previous_event(previous: dict[str, Any] | None) -> dict[str, Any] | None:
    if not previous:
        return None
    return {
        "schema": previous.get("schema"),
        "motion_id": previous.get("motion_id"),
        "action_id": previous.get("action_id"),
        "status": previous.get("status"),
        "created_at": previous.get("created_at"),
        "completed_at": previous.get("completed_at"),
        "completed_at_epoch": cooldown_completed_epoch(previous),
        "returncode": (previous.get("result") or {}).get("returncode") if isinstance(previous.get("result"), dict) else None,
    }


def resolve_repo_path(value: str) -> Path:
    path = (ROOT / value).resolve()
    if ROOT.resolve() not in path.parents and path != ROOT.resolve():
        raise ValueError(f"path escapes repository: {value}")
    return path


def validate_manifest(manifest_path: Path, allowed_prefixes: list[str]) -> None:
    jobs = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("cluster manifest must be a non-empty job list")
    for job in jobs:
        if not isinstance(job, dict):
            raise ValueError("cluster manifest jobs must be objects")
        cmd = str(job.get("cmd", "")).strip()
        if not any(cmd.startswith(prefix) for prefix in allowed_prefixes):
            raise ValueError(f"job {job.get('name')} command is not allowlisted: {cmd}")


def execute_cluster_dispatch(action: dict[str, Any]) -> dict[str, Any]:
    nodes = resolve_repo_path(action["nodes"])
    manifest = resolve_repo_path(action["manifest"])
    validate_manifest(manifest, list(action.get("allowed_command_prefixes", [])))
    cmd = [
        sys.executable,
        str(ROOT / "src" / "lab_platform" / "cluster_dispatch.py"),
        "--nodes",
        str(nodes),
        "--manifest",
        str(manifest),
        "--per-task-timeout",
        str(int(action.get("per_task_timeout_s", 300))),
    ]
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=int(action.get("wall_timeout_s", 360)),
    )
    return {
        "kind": "cluster_dispatch",
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-6000:],
        "stderr_tail": proc.stderr[-3000:],
    }


def execute_action(spec: dict[str, Any]) -> dict[str, Any]:
    action = spec.get("action", {})
    kind = action.get("kind")
    if kind == "cluster_dispatch":
        return execute_cluster_dispatch(action)
    raise ValueError(f"unsupported Parliament action kind: {kind}")


def review_motion(motion_id: str, execute: bool = False, force: bool = False) -> dict[str, Any]:
    spec = load_action_spec(motion_id)
    if not spec:
        return {"schema": "parliament.action_review.v1", "motion_id": motion_id, "status": "no_action_spec"}
    action_id = str(spec["action_id"])
    speeches = flatten_firebase_speeches(firebase_get(f"parliament/speeches/{motion_id}", timeout=8.0))
    tally = tally_votes(speeches, spec.get("approval", {}))
    previous = load_previous_event(motion_id, action_id)
    event: dict[str, Any] = {
        "schema": "parliament.action_event.v1",
        "motion_id": motion_id,
        "action_id": action_id,
        "created_at": utc_now(),
        "tally": tally,
        "dry_run": not execute,
    }
    if in_cooldown(spec, previous) and not force:
        event["status"] = "skipped_cooldown"
        completed_epoch = cooldown_completed_epoch(previous)
        if completed_epoch:
            event["completed_at_epoch"] = completed_epoch
        event["previous"] = compact_previous_event(previous)
    elif not tally["approved"]:
        event["status"] = "waiting_for_votes"
    elif not execute:
        event["status"] = "ready"
    else:
        started = time.time()
        event["status"] = "running"
        firebase_put(f"parliament/actions/{motion_id}/{action_id}", event, timeout=5.0)
        result = execute_action(spec)
        event["completed_at"] = utc_now()
        event["completed_at_epoch"] = time.time()
        event["wall_s"] = round(time.time() - started, 2)
        event["result"] = result
        event["status"] = "completed" if result.get("returncode") == 0 else "failed"

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    local_event_path(motion_id, action_id).write_text(json.dumps(event, indent=2, ensure_ascii=False), encoding="utf-8")
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    (HISTORY_DIR / f"{motion_id}-{action_id}-{stamp}.json").write_text(
        json.dumps(event, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if execute or event.get("status") in {"ready", "waiting_for_votes", "skipped_cooldown"}:
        firebase_put(f"parliament/actions/{motion_id}/{action_id}", event, timeout=5.0)
    return event


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Review and execute allowlisted Parliament actions")
    parser.add_argument("review", nargs="?", default="review")
    parser.add_argument("--motion", required=True, help="Motion id, for example small_lm_recovery")
    parser.add_argument("--execute", action="store_true", help="Execute when approved instead of only reporting readiness")
    parser.add_argument("--force", action="store_true", help="Bypass action cooldown")
    args = parser.parse_args(argv)
    if args.review != "review":
        parser.error("only the 'review' command is supported")
    event = review_motion(args.motion, execute=args.execute, force=args.force)
    print(json.dumps(event, indent=2, ensure_ascii=False))
    return 0 if event.get("status") not in {"failed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
