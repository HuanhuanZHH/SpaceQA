import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import opc_agent
from fixture_factory import label, write_json, write_png


class ReleaseLayoutTests(unittest.TestCase):
    def test_mixed_layouts_resolve_flat_files_despite_stale_relative_path(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            nested = root / "HMI_AR_sample_aaaaaaaaaa"
            flat = root / "EPB_seg_sample_bbbbbbbbbb"
            missing = root / "ULF_sample_cccccccccc"
            (nested / "data").mkdir(parents=True)
            (nested / "label").mkdir()
            write_png(nested / "data" / "a.png")
            write_json(nested / "label" / "a.png.json", label("a.png", "2026-01-01", []))
            write_png(flat / "b.png")
            write_json(flat / "b.json", label("b.png", "2026-01-01", []))
            write_json(missing / "c.csv.json", label("c.csv", "2026-01-01", []))
            self.assertEqual(set(opc_agent.discover_datasets(root)), {nested, flat, missing})
            self.assertEqual(opc_agent.discover_datasets(flat), [flat])
            dataset = opc_agent.inventory_dataset(flat, None)
            opc_agent.check_c01(dataset)
            self.assertEqual(opc_agent.check_c02(dataset), [])
            self.assertEqual(opc_agent.check_c03(dataset), [])
            absent = opc_agent.inventory_dataset(missing, None)
            opc_agent.check_c01(absent)
            self.assertEqual(len(opc_agent.check_c02(absent)), 1)

    def test_stem_names_are_valid_but_unrelated_label_names_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_png(root / "a.png")
            for name in ("a.json", "a.png.json", "unrelated.json"):
                write_json(root / name, label("a.png", "2026-01-01", []))
            dataset = opc_agent.inventory_dataset(root, None)
            opc_agent.check_c01(dataset)
            findings = opc_agent.check_c03(dataset)
            self.assertEqual(len(findings), 1)
            self.assertEqual(findings[0]["defectType"], "LABEL_FILENAME_MISMATCH")
            self.assertTrue(findings[0]["_locator"].endswith("unrelated.json"))

    def test_polluted_window_does_not_change_sample_identity(self):
        before = label("a.csv", "2026-01-01T00:00:00", [])
        after = json.loads(json.dumps(before))
        after["basic_annotation"]["startTime"] = "2025-12-31T23:58:00"
        self.assertEqual(opc_agent.sample_key(Path("a.csv.json"), before),
                         opc_agent.sample_key(Path("a.csv.json"), after))

    def test_raw_csv_is_not_provenance_document(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "a.csv").write_text("t,H,D,Z,T\n0,1,2,3,4\n", encoding="utf-8")
            write_json(root / "a.csv.json", label("a.csv", "2026-01-01", []))
            dataset = opc_agent.inventory_dataset(root, None)
            opc_agent.check_c11(dataset)
            self.assertEqual(dataset.findings[0]["status"], opc_agent.STATUS_PENDING)


if __name__ == "__main__":
    unittest.main()
