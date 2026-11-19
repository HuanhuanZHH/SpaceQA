import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import run_roadshow_prior as fresh
import stage_replay as replay


class ReplayEntryTests(unittest.TestCase):
    def test_only_exact_path_prefixes_are_relocated(self):
        before = "x='/staff/tangjiarui/OPC/inference_prior_20261102/gbdt'\ny=150\nz='/staff/tangjiarui/OPC/inference_prior_202611020/other'\n"
        generated, changes = replay.relocate_source(before, {replay.OLD_RUN: "/new/runs"})
        tree = ast.parse(generated)
        self.assertEqual(tree.body[0].value.value, "/new/runs/gbdt")
        self.assertEqual(tree.body[1].value.value, 150)
        self.assertTrue(tree.body[2].value.value.endswith("202611020/other"))
        self.assertEqual(len(changes), 1)

    def test_stage_preserves_archive_and_refuses_existing_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "archive"
            archive.mkdir()
            for name in replay.SOURCES:
                path = archive / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("value='/staff/tangjiarui/OPC/inference_prior_20261102/gbdt'\niterations=150\n")
            before = {name: (archive/name).read_bytes() for name in replay.SOURCES}
            report = replay.stage(root / "new", archive=archive)
            self.assertEqual(len(report["files"]), len(replay.SOURCES))
            self.assertFalse(report["experiments_started"])
            self.assertEqual(before, {name: (archive/name).read_bytes() for name in replay.SOURCES})
            with self.assertRaises(FileExistsError):
                replay.stage(root / "new", archive=archive)

    def test_stage_rejects_workspace_inside_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                replay.stage(Path(directory)/"child", archive=directory)

    def test_fresh_entry_stops_before_obsolete_fusion(self):
        tree = ast.parse("def main():\n events.append('scan')\n event('roadshow_prior_complete')\n events.append('obsolete_fusion')\n")
        events = []
        namespace = dict(events=events, event=lambda name: events.append(name))
        exec(compile(fresh.fresh_only_tree(tree), "fixture", "exec"), namespace)
        namespace["main"]()
        self.assertEqual(events, ["scan", "roadshow_prior_complete"])

    def test_fresh_entry_requires_unambiguous_checkpoint(self):
        for source in ("def main():\n pass\n", "def main():\n event('roadshow_prior_complete')\n event('roadshow_prior_complete')\n"):
            with self.assertRaises(ValueError):
                fresh.fresh_only_tree(ast.parse(source))

    def test_both_entry_help_commands_are_read_only(self):
        root = Path(__file__).resolve().parent
        for name in ("stage_replay.py", "run_roadshow_prior.py"):
            result = subprocess.run([sys.executable, str(root/name), "--help"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--output" if name.startswith("run_") else "--workspace", result.stdout)
            self.assertIn("overwrite" if name.startswith("stage_") else "obsolete", result.stdout)


if __name__ == "__main__":
    unittest.main()
