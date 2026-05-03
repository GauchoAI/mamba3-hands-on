from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def run_parliament(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [PYTHON, "tools/parliament.py", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=True,
    )


class ParliamentSmokeTests(unittest.TestCase):
    def test_expected_owner_identities_exist(self) -> None:
        expected = {
            "gpt5-ch12-chess-champion",
            "claude-hanoi-lego-puzzle-solver",
            "claude-cortex-primitive-owner",
            "claude-language-jepa-owner",
            "claude-phi-composition-owner",
            "claude-platform-kappa-clerk",
            "claude-opposition-architect",
        }
        identity_dir = ROOT / "parliament" / "identities"
        found = {p.stem for p in identity_dir.glob("*.yaml")}
        self.assertTrue(expected.issubset(found))
        for speaker in expected:
            payload = yaml.safe_load((identity_dir / f"{speaker}.yaml").read_text())
            self.assertEqual(payload["speaker"], speaker)
            self.assertTrue(payload.get("credentials"))
            self.assertTrue(payload.get("evidence") or payload.get("chapters") is not None)

    def test_extracts_claude_wrapped_json(self) -> None:
        from tools.parliament import extract_json_object

        wrapped = {
            "type": "result",
            "result": "```json\n{\"kind\":\"position\",\"position\":\"approve\",\"body\":\"ok\",\"evidence\":[],\"prediction\":\"p\",\"falsifier\":\"f\",\"confidence\":0.5}\n```",
        }
        payload = extract_json_object(json.dumps(wrapped))
        self.assertEqual(payload["position"], "approve")

    def test_simple_yaml_fallback_handles_identity_shape(self) -> None:
        from tools.parliament import parse_simple_yaml

        payload = parse_simple_yaml(
            """
speaker: example
role: judge
chapters:
  - 04_hanoi
  - 05_lego_library
stance: Exactness first.
"""
        )
        self.assertEqual(payload["speaker"], "example")
        self.assertEqual(payload["chapters"], ["04_hanoi", "05_lego_library"])

    def test_rejects_invalid_speech_vocab(self) -> None:
        from tools.parliament import validate_speech

        with self.assertRaises(ValueError):
            validate_speech(
                {
                    "kind": "banana",
                    "position": "maybe",
                    "body": "bad",
                    "evidence": [],
                    "prediction": "p",
                    "falsifier": "f",
                    "confidence": 0.5,
                }
            )

    def test_auto_backend_selects_from_identity_family(self) -> None:
        from tools.parliament import resolve_backend_name

        self.assertEqual(resolve_backend_name("auto", {"model_family": "Claude"}), "claude")
        self.assertEqual(resolve_backend_name("auto", {"model_family": "GPT-5 / Codex"}), "codex")
        self.assertEqual(resolve_backend_name("simulated", {"model_family": "Claude"}), "simulated")

    def test_all_speakers_expands_to_owner_chamber(self) -> None:
        from tools.parliament import DEFAULT_CHAMBER_SPEAKERS, expand_speakers

        self.assertEqual(expand_speakers(["all"]), DEFAULT_CHAMBER_SPEAKERS)
        self.assertEqual(expand_speakers(["claude-opposition-architect"]), ["claude-opposition-architect"])

    def test_scheduler_panel_rotates(self) -> None:
        from tools.parliament_tick import choose_panel

        speakers = ["a", "b", "c", "d", "e"]
        self.assertEqual(choose_panel(0, speakers, 2), ["a", "b"])
        self.assertEqual(choose_panel(1, speakers, 2), ["c", "d"])
        self.assertEqual(choose_panel(2, speakers, 2), ["e", "a"])

    def test_action_tally_requires_quorum_and_confidence(self) -> None:
        from tools.parliament_action import tally_votes

        speeches = [
            {"speaker": "a", "created_at": "1", "speech": {"position": "approve", "confidence": 0.7}},
            {"speaker": "b", "created_at": "1", "speech": {"position": "amend", "confidence": 0.8}},
            {"speaker": "c", "created_at": "1", "speech": {"position": "approve", "confidence": 0.2}},
        ]
        tally = tally_votes(
            speeches,
            {"quorum": 2, "min_approve": 2, "min_confidence": 0.6, "positive_positions": ["approve", "amend"]},
        )
        self.assertTrue(tally["approved"])
        self.assertEqual(tally["approvals"], 2)

    def test_action_cooldown_accepts_previous_cooldown_wrapper(self) -> None:
        from tools.parliament_action import in_cooldown

        previous = {
            "status": "skipped_cooldown",
            "previous": {"status": "completed", "completed_at_epoch": 9999999999.0},
        }
        self.assertTrue(in_cooldown({"cooldown_s": 86400}, previous))

    def test_action_manifest_rejects_unallowlisted_commands(self) -> None:
        import tempfile

        from tools.parliament_action import validate_manifest

        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump([{"node": "m4-pro", "name": "bad", "cmd": "rm -rf runs"}], f)
            f.flush()
            with self.assertRaises(ValueError):
                validate_manifest(
                    Path(f.name),
                    ["git pull --ff-only && .venv/bin/python tools/parliament_action.py review --motion small_lm_recovery"],
                )

    def test_bill_proposal_rejects_destructive_shell(self) -> None:
        from tools.parliament_bill import validate_proposal

        proposal = {
            "proposal_id": "bad-shell",
            "title": "Bad shell",
            "objective": "should fail",
            "hypothesis": "should fail",
            "command": ".venv/bin/python ok.py && rm -rf runs",
            "max_wall_s": 60,
            "expected_artifacts": ["runs/x.json"],
            "kpi": {"namespace": "test", "metric": "score", "direction": "increase", "target": 0.1},
            "falsifier": "none",
            "follow_up": "none",
        }
        errors = validate_proposal(proposal)
        self.assertTrue(any("shell control" in error for error in errors))

    def test_bill_proposal_rejects_multiline_embedded_code(self) -> None:
        from tools.parliament_bill import validate_proposal

        proposal = {
            "proposal_id": "bad-multiline",
            "title": "Bad multiline",
            "objective": "should fail",
            "hypothesis": "should fail",
            "command": "python - <<'PY'\nprint('bad')\nPY",
            "max_wall_s": 60,
            "expected_artifacts": ["runs/x.json"],
            "kpi": {"namespace": "test", "metric": "score", "direction": "increase", "target": 0.1},
            "falsifier": "none",
            "follow_up": "none",
        }
        errors = validate_proposal(proposal)
        self.assertTrue(any("single line" in error for error in errors))

    def test_bill_proposal_accepts_bounded_repo_python_command(self) -> None:
        from tools.parliament_bill import validate_proposal

        proposal = {
            "proposal_id": "bounded-probe",
            "title": "Bounded probe",
            "objective": "Run one bounded probe",
            "hypothesis": "Metric improves",
            "command": ".venv/bin/python tools/parliament_action.py review --motion small_lm_recovery",
            "max_wall_s": 60,
            "expected_artifacts": ["runs/parliament/actions/small_lm_recovery-small_lm_recovery_parliament_smoke.json"],
            "kpi": {"namespace": "parliament/test", "metric": "returncode", "direction": "hit", "target": 0},
            "falsifier": "nonzero returncode",
            "follow_up": "review action event",
        }
        self.assertEqual(validate_proposal(proposal), [])

    def test_training_heartbeat_without_checkpoint_is_silent(self) -> None:
        result = run_parliament(
            "event",
            "--event",
            '{"type":"heartbeat","status":"training","checkpoint":null}',
        )
        payload = json.loads(result.stdout)
        self.assertEqual(payload["decision"], "silent")

    def test_checkpoint_with_kpi_triggers_deliberation(self) -> None:
        result = run_parliament(
            "event",
            "--event",
            '{"type":"checkpoint","status":"training","checkpoint":"runs/x/ckpt.pt","kpi":0.73}',
        )
        payload = json.loads(result.stdout)
        self.assertEqual(payload["decision"], "deliberate")
        self.assertIn("checkpoint", payload["reason"])
        self.assertIn("kpi", payload["reason"])

    def test_dry_run_chamber_returns_two_speeches(self) -> None:
        result = run_parliament(
            "chamber",
            "--speakers",
            "gpt5-ch12-chess-champion",
            "claude-opposition-architect",
            "--motion",
            "parliament/motions/example_chess_kpi.md",
            "--dry-run",
        )
        payload = json.loads(result.stdout)
        self.assertTrue(payload["dry_run"])
        self.assertEqual(len(payload["speeches"]), 2)
        self.assertEqual(payload["speeches"][1]["model_family"], "Claude")

    def test_inline_motion_id_override_is_durable(self) -> None:
        result = run_parliament(
            "chamber",
            "--speakers",
            "gpt5-ch12-chess-champion",
            "--text",
            "Vote on a compiled bill.",
            "--motion-id",
            "procedural_bill_request",
            "--dry-run",
        )
        payload = json.loads(result.stdout)
        self.assertEqual(payload["speeches"][0]["motion_id"], "procedural_bill_request")


if __name__ == "__main__":
    unittest.main()
