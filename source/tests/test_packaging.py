"""Installed-wheel checks, independent from the checkout's working directory."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import opc_agent
from spaceqa_agent import __version__
from fixture_factory import build_fixture


class PackagingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = tempfile.TemporaryDirectory()
        cls.base = Path(cls.workspace.name)
        cls.installed = cls.base / "installed"
        wheels = cls.base / "wheels"
        cls._command(
            [sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
             "--wheel-dir", str(wheels), str(ROOT)], ROOT
        )
        wheel = next(wheels.glob("spaceqa_agent-*.whl"))
        cls._command(
            [sys.executable, "-m", "pip", "install", "--no-deps", "--target",
             str(cls.installed), str(wheel)], cls.base
        )
        cls.environment = dict(os.environ, PYTHONPATH=str(cls.installed), PYTHONIOENCODING="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls.workspace.cleanup()

    @staticmethod
    def _command(arguments, cwd, env=None):
        result = subprocess.run(arguments, cwd=cwd, env=env, capture_output=True, text=True, encoding="utf-8")
        if result.returncode:
            raise AssertionError(f"Command failed ({result.returncode}): {arguments}\n{result.stdout}\n{result.stderr}")
        return result

    def test_installed_scan_matches_source_and_independent_audit(self):
        case = self.base / "scan_case"
        publish, _, _ = build_fixture(case)
        before = {str(p.relative_to(publish)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in publish.rglob("*") if p.is_file()}
        expected = case / "source_result"
        installed = case / "wheel_result"
        source_summary = opc_agent.run(publish, expected, 43, set(opc_agent.ALL_CHECKS))
        run = self._command(
            [sys.executable, "-m", "spaceqa_agent", "scan", "--input", str(publish),
             "--output", str(installed), "--seed", "43"], self.base, self.environment
        )
        summary = json.loads(run.stdout)
        self.assertEqual(summary["runId"], source_summary["runId"])
        self.assertEqual(summary["findingCount"], source_summary["findingCount"])
        findings = Path("06_问题清单") / "findings.jsonl"
        self.assertEqual((installed / findings).read_bytes(), (expected / findings).read_bytes())
        checked = self._command(
            [sys.executable, "-m", "spaceqa_agent", "audit", str(installed)], self.base, self.environment
        )
        self.assertEqual(json.loads(checked.stdout)["status"], "pass")
        after = {str(p.relative_to(publish)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in publish.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_legacy_cli_and_installed_version_work_outside_checkout(self):
        version = self._command(
            [sys.executable, "-m", "spaceqa_agent", "--version"], self.base, self.environment
        )
        self.assertIn(__version__, version.stdout)
        legacy = self._command([sys.executable, str(ROOT / "opc_agent.py"), "--help"], self.base)
        self.assertIn("audit", legacy.stdout)

    def test_installed_wheel_serves_web_assets_without_checkout(self):
        publish, _, _ = build_fixture(self.base / "web_case")
        code = """
import json, threading, urllib.request
from pathlib import Path
from spaceqa_agent.web_service import create_server
server = create_server(port=0, data_roots=[Path(__import__('sys').argv[1])],
                       state_dir=Path(__import__('sys').argv[2]), start_worker=False)
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
try:
    base = 'http://127.0.0.1:' + str(server.server_address[1])
    for name, marker in [('/', b'SpaceQA'), ('/styles.css', b'.sidebar'), ('/app.js', b'/api/health')]:
        content = urllib.request.urlopen(base + name, timeout=5).read()
        assert marker in content, name
    assert json.load(urllib.request.urlopen(base + '/api/health'))['service'] == 'spaceqa-web'
finally:
    server.shutdown()
    server.server_close()
"""
        self._command([sys.executable, "-c", code, str(publish), str(self.base / "installed_web_state")],
                      self.base, self.environment)

    def test_legacy_dynamic_loader_keeps_module_functions(self):
        # full_matrix/common.py returns its original module object after exec.
        spec = importlib.util.spec_from_file_location("legacy_rule_loader_test", ROOT / "opc_agent.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
            self.assertIs(module.run, opc_agent.run)
            self.assertIs(module.Dataset, opc_agent.Dataset)
        finally:
            sys.modules.pop(spec.name, None)

    def test_existing_output_is_not_overwritten(self):
        case = self.base / "existing_case"
        publish, _, _ = build_fixture(case)
        existing = case / "result"
        existing.mkdir()
        marker = existing / "keep.txt"
        marker.write_text("retained", encoding="utf-8")
        result = subprocess.run(
            [sys.executable, "-m", "spaceqa_agent", "scan", "--input", str(publish),
             "--output", str(existing)], cwd=self.base, env=self.environment,
            capture_output=True, text=True, encoding="utf-8"
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("Output already exists", result.stderr)
        self.assertEqual(marker.read_text(encoding="utf-8"), "retained")

    def test_bundled_rule_assets_match_research_compatibility_copies(self):
        for group in ("spec", "agent"):
            for source in (ROOT / group).iterdir():
                if source.is_file():
                    bundled = self.installed / "spaceqa_agent" / "assets" / group / source.name
                    self.assertEqual(bundled.read_bytes(), source.read_bytes())


if __name__ == "__main__":
    unittest.main()
