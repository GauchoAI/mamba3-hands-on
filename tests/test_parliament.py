from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


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
