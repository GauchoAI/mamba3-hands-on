from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def run_cmd(*args: str, timeout: float = 30.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [PYTHON, *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=True,
    )


class PackageSmokeTests(unittest.TestCase):
    def test_platform_imports(self) -> None:
        from lab_platform.cortex_counting import CortexLMConfig
        from lab_platform.experiment_pusher import ExperimentPusher
        from lab_platform.lab_run import LabRun
        from lab_platform.mamba3_minimal import Mamba3Config

        self.assertEqual(Mamba3Config().d_model, 64)
        self.assertGreater(CortexLMConfig().d_model, 0)
        self.assertTrue(callable(ExperimentPusher))
        self.assertTrue(callable(LabRun))

    def test_lab_run_facade_offline(self) -> None:
        from lab_platform.lab_run import LabRun

        with tempfile.TemporaryDirectory() as tmp:
            run = LabRun(
                experiment_id="smoke-exp",
                run_id="smoke-run",
                kind="smoke",
                config={"lr": 1e-3},
                out_dir=tmp,
                live_enabled=False,
                archive_enabled=False,
            )
            run.start(name="Smoke", purpose="offline facade")
            run.metric(step=1, loss=1.0)
            run.sample(step=1, prompt="a", completion="b")
            run.event("done", step=1)
            run.flush()
            run.complete(final_metrics={"loss": 1.0})
            self.assertFalse(run.live)
            self.assertFalse(run.archived)

    def test_module_clis_load(self) -> None:
        checks = [
            ("-m", "lab_platform.kappa_packer", "--help"),
            ("-m", "lab_platform.stream_reader", "--help"),
            ("-m", "lab_platform.lab_book", "--help"),
        ]
        for args in checks:
            with self.subTest(args=args):
                result = run_cmd(*args)
                self.assertIn("usage:", result.stdout)


class ActiveExperimentSmokeTests(unittest.TestCase):
    active_dir = ROOT / "experiments" / "10_jepa_structured"

    def test_curriculum_and_status_load(self) -> None:
        result = run_cmd(str(self.active_dir / "orchestrator.py"), "status")
        self.assertIn("curriculum balance", result.stdout)
        self.assertIn("tiles complete", result.stdout)

    def test_registry_json_loads(self) -> None:
        registry_path = self.active_dir / "state" / "registry.json"
        registry = json.loads(registry_path.read_text())
        self.assertIn("math.modular_arithmetic.addition_basic", registry)

    def test_one_step_training_uses_temp_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state_dir = tmp_path / "state"
            checkpoint_dir = tmp_path / "checkpoints"
            state_dir.mkdir()
            checkpoint_dir.mkdir()

            for name in ("registry.json", "daily_budget.json"):
                src = self.active_dir / "state" / name
                (state_dir / name).write_text(src.read_text())

            result = run_cmd(
                str(self.active_dir / "orchestrator.py"),
                "--state-dir",
                str(state_dir),
                "--out",
                str(self.active_dir / "data"),
                "train",
                "--order",
                "topo",
                "--limit",
                "1",
                "--steps-per-tile",
                "1",
                "--batch-size",
                "1",
                "--device",
                "cpu",
                "--checkpoints",
                str(checkpoint_dir),
                timeout=60.0,
            )
            self.assertIn("[train] math.modular_arithmetic.addition_basic", result.stdout)
            self.assertTrue((checkpoint_dir / "student.pt").exists())


class StackOperatorSmokeTests(unittest.TestCase):
    active_dir = ROOT / "experiments" / "11_stack_operator_transfer"

    def test_stack_operator_one_minute_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_cmd(
                str(self.active_dir / "stack_operator.py"),
                "--epochs",
                "150",
                "--trials",
                "20",
                "--out-dir",
                tmp,
                timeout=60.0,
            )
            payload = json.loads(result.stdout)
            self.assertTrue(payload["one_minute_rule"])
            self.assertGreaterEqual(payload["heldout_state_acc"], 0.98)
            self.assertLess(payload["elapsed_s"], 60.0)
            self.assertTrue((Path(tmp) / "stack_operator.pt").exists())


class OperatorCurriculumSmokeTests(unittest.TestCase):
    scripts = [
        ROOT / "experiments" / "12_raw_trace_stack" / "raw_trace_stack.py",
        ROOT / "experiments" / "13_multi_surface_stack" / "multi_surface_stack.py",
        ROOT / "experiments" / "14_operator_guided_decoding" / "operator_guided_decoding.py",
        ROOT / "experiments" / "15_comparator_transfer" / "comparator_transfer.py",
        ROOT / "experiments" / "16_trace_to_operator_search" / "trace_to_operator_search.py",
        ROOT / "experiments" / "18_operator_composition" / "operator_composition.py",
        ROOT / "experiments" / "19_language_to_role_trace" / "language_to_role_trace.py",
        ROOT / "experiments" / "20_runtime_learning_episode" / "runtime_learning_episode.py",
        ROOT / "experiments" / "17_operator_registry" / "operator_registry.py",
        ROOT / "experiments" / "21_lab_organ_demo" / "lab_organ_demo.py",
    ]

    def test_operator_curriculum_scripts(self) -> None:
        for script in self.scripts:
            with self.subTest(script=script.name):
                result = run_cmd(str(script), timeout=60.0)
                payload = json.loads(result.stdout)
                self.assertTrue(payload["one_minute_rule"])
                self.assertLess(payload["elapsed_s"], 60.0)


class LatentOperatorDiscoverySmokeTests(unittest.TestCase):
    active_dir = ROOT / "experiments" / "13_latent_operator_discovery"

    def test_latent_operator_discovery_one_minute(self) -> None:
        result = run_cmd(str(self.active_dir / "latent_operator_discovery.py"), timeout=70.0)
        payload = json.loads(result.stdout)
        self.assertTrue(payload["one_minute_rule"])
        self.assertGreaterEqual(payload["heldout"]["candidate_validity_acc"], 0.95)
        self.assertGreaterEqual(payload["boundary_probe"]["heldout_acc"], 0.85)


class LabBookManifestSmokeTests(unittest.TestCase):
    def test_manifest_is_generated_and_consolidated(self) -> None:
        result = run_cmd(str(ROOT / "tools" / "generate_lab_book_manifest.py"))
        self.assertIn("experiment sources", result.stdout)
        manifest = json.loads((ROOT / "docs" / "lab_book" / "manifest.json").read_text())
        ids = {source["id"] for source in manifest["sources"]}
        paths = {source["path"] for source in manifest["sources"]}
        self.assertIn("ch12", ids)
        self.assertIn("ch13", ids)
        self.assertIn("experiments/12_operator_curriculum_intro/README.md", paths)
        self.assertIn("experiments/13_latent_operator_discovery/README.md", paths)
        self.assertNotIn("experiments/21_lab_organ_demo/README.md", paths)


if __name__ == "__main__":
    unittest.main()
