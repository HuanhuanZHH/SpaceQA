"""Refactoring must preserve the frozen rule engine's scientific conclusions."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import opc_agent
from multifamily_fixture import build_multifamily_fixture


class LegacyEquivalenceTests(unittest.TestCase):
    def test_all_six_families_and_five_tiers_match_frozen_engine(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = root / "legacy"
            legacy.mkdir()
            shutil.copyfile(REPO / "reproducibility/legacy_runtime/opc_agent.py", legacy / "opc_agent.py")
            for name in ("spec", "agent"):
                shutil.copytree(REPO / name, legacy / name)
            name = "spaceqa_frozen_rule_engine"
            spec = importlib.util.spec_from_file_location(name, legacy / "opc_agent.py")
            original = importlib.util.module_from_spec(spec)
            sys.modules[name] = original
            try:
                spec.loader.exec_module(original)
                publish, _ = build_multifamily_fixture(root / "fixture")
                old_output, new_output = root / "old", root / "new"
                old = original.run(publish, old_output, 20261031, set(original.ALL_CHECKS))
                new = opc_agent.run(publish, new_output, 20261031, set(opc_agent.ALL_CHECKS))
                for field in ("runId", "datasetCount", "findingCount", "confirmedDefectCount", "pendingCount"):
                    self.assertEqual(old[field], new[field], field)
                self.assertEqual(old["datasetCount"], 30)
                for filename in ("06_问题清单/findings.jsonl", "05_排序表/ranking_global.csv"):
                    self.assertEqual((old_output / filename).read_bytes(), (new_output / filename).read_bytes(), filename)
                for card in (old_output / "04_评分卡").glob("*.json"):
                    self.assertEqual(json.loads(card.read_text(encoding="utf-8")),
                                     json.loads((new_output / "04_评分卡" / card.name).read_text(encoding="utf-8")))
            finally:
                sys.modules.pop(name, None)


if __name__ == "__main__":
    unittest.main()
