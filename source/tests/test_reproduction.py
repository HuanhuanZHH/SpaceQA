"""Meaningful guards for scientific-source isolation and artifact verification."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("reproduce", REPO / "scripts/reproduce.py")
reproduce = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reproduce)


class ReproductionTests(unittest.TestCase):
    def test_relocation_preserves_every_nonpath_ast_constant(self):
        original = "from pathlib import Path\nROOT=Path('/staff/tangjiarui/OPC')\nseed=20261031\n"
        relocated = reproduce.relocate_python(original, Path("/tmp/isolated-run"))
        self.assertIn("seed=20261031", relocated)
        self.assertIn("/tmp/isolated-run", relocated)
        self.assertNotIn(reproduce.LEGACY_ROOT, relocated)

    def test_remap_only_path_prefixes_not_labels_or_embedded_text(self):
        original = {"path": "/old/data/x.png", "description": "note /old/data", "label": "old"}
        moved = reproduce.remap(original, {"/old": "/new"})
        self.assertEqual(moved["path"], "/new/data/x.png")
        self.assertEqual(moved["description"], original["description"])
        self.assertEqual(original["path"], "/old/data/x.png")

    def test_existing_workspace_is_never_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "existing-result.json"
            marker.write_text('{"status":"complete"}', encoding="utf-8")
            with self.assertRaises(FileExistsError):
                reproduce.stage({}, Path(directory))
            self.assertEqual(json.loads(marker.read_text())["status"], "complete")

    def test_unknown_gpu_policy_cannot_silently_change_frozen_runner(self):
        config = json.loads((REPO / "configs/reproduce.example.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            config.update(external_root=directory, gpu="0")
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "GPU1"):
                reproduce.config_read(path)

    def test_evaluate_is_explicit_rescore_with_training_barrier(self):
        commands = reproduce.command_plan({"image_python": "/venv/python"}, Path("/workspace"), "evaluate")
        self.assertEqual(Path(commands[0][2]).name, "barrier.py")
        self.assertEqual(commands[1][-1], "score")
        self.assertEqual(commands[-1][-1], "--results")

    def test_roadshow_plan_uses_only_local_driver(self):
        commands = reproduce.command_plan({"image_python": "/venv/python"}, Path("/workspace"), "roadshow")
        self.assertEqual(len(commands), 1)
        self.assertEqual(Path(commands[0][2]).name, "roadshow_local.py")
        self.assertNotIn("watch", commands[0])

    @unittest.skipIf(__import__("os").name == "nt", "Real directory symlinks require privileges on Windows")
    def test_rule_locators_relocate_with_manifest_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "rules"
            index = source / "07_证据索引"
            index.mkdir(parents=True)
            (index / "evidence_index.csv").write_text(
                "evidenceId,locator\nE1,/old/data/x.json#/science/mask\n", encoding="utf-8")
            before = (index / "evidence_index.csv").read_bytes()
            target = root / "staged"
            reproduce.stage_rule_run(source, target, {"/old": "/new"})
            self.assertIn("/new/data/x.json#/science/mask", (target / "07_证据索引/evidence_index.csv").read_text(encoding="utf-8-sig"))
            self.assertEqual((index / "evidence_index.csv").read_bytes(), before)

    def test_snapshot_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "configs").mkdir()
            (root / "results/reference").mkdir(parents=True)
            (root / "results/reference/metrics_long.csv").write_text(
                "route,family,tier,TP,FP,FN,F1,precision,recall\nx,AIA,P1,1,0,0,1,1,1\n", encoding="utf-8")
            source = root / "model.py"
            source.write_text("seed=20261031\n", encoding="utf-8")
            manifest = root / "configs/source_manifest.json"
            with mock.patch.object(reproduce, "REPO", root):
                reproduce.snapshot(manifest)
                self.assertEqual(reproduce.verify(manifest)["status"], "pass")
                source.write_text("seed=99\n", encoding="utf-8")
                self.assertIn("model.py", reproduce.verify(manifest)["errors"])


if __name__ == "__main__":
    unittest.main()
