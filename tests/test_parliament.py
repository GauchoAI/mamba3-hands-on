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


if __name__ == "__main__":
    unittest.main()
