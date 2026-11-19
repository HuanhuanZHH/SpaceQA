"""Protection and frozen-routing tests; the server smoke verifies real models."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("portable_test_entry", Path(__file__).with_name("portable_experts.py"))
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


class PortableProtections(unittest.TestCase):
    def test_lock_rejects_changed_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            model = folder / "model.pt"
            model.write_bytes(b"original")
            lock = folder / "lock.json"
            entry.write(lock, dict(version=entry.VERSION, status="frozen_research_entry", files={str(model): entry.sha(model)}))
            entry.validate_lock(lock)
            model.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "changed"):
                entry.validate_lock(lock)

    def test_selection_must_stay_inside_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            root = folder / "package"
            root.mkdir()
            outside = folder / "foreign.json"
            outside.write_text("{}")
            with self.assertRaisesRegex(ValueError, "escapes"):
                entry.pair_labels(root, "AIA_AR_sample", [str(outside)])

    def test_aia_and_ulf_pairing_keeps_absent_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "label").mkdir()
            (root / "data").mkdir()
            label = root / "label/a.png.json"
            label.write_text("{}")
            pairs = entry.pair_labels(root, "AIA_AR_sample")
            self.assertEqual(pairs, [(label, root / "data/a.png")])
            self.assertFalse(pairs[0][1].exists())
            self.assertEqual(entry.pair_labels(root, "HMI_AR_sample", [str(label)])[0][1], root / "data/a.png")
            self.assertEqual(entry.pair_labels(root, "CME_C2_sample", [str(label)])[0][1], root / "data/a.png")
            ulf = root / "b.csv.json"
            ulf.write_text("{}")
            self.assertEqual(entry.pair_labels(root, "ULF_sample", [str(ulf)])[0][1], root / "b.csv")

    def test_threshold_disable_bits_are_preserved(self):
        choices = {f: dict(route="unetpp", thresholds=[.6,.9,.7]) for f in entry.FAMILIES}
        base = dict(status="frozen", test_split_parsed=False, recommended_variant="compact_expert_or_new_prior",
                    recalibrated={"compact": dict(choices=choices, gc_only=True)})
        ext = dict(status="frozen", test_split_parsed=False,
                   selected={f: dict(variant="base_validation_recommendation") for f in entry.FAMILIES}, thresholds={})
        routes = entry.selected_routes(base, ext)
        self.assertEqual(routes[entry.FAMILIES[0]]["experts"][0]["thresholds"], [.6,.9,1.01])
        self.assertEqual(choices[entry.FAMILIES[0]]["thresholds"], [.6,.9,.7])
        ext["test_split_parsed"] = True
        with self.assertRaisesRegex(ValueError, "before test"):
            entry.selected_routes(base, ext)

    def test_output_cannot_overwrite_or_overlap_raw(self):
        with tempfile.TemporaryDirectory() as temporary:
            old = entry.RESEARCH_ROOT
            try:
                entry.RESEARCH_ROOT = Path(temporary)
                target = Path(temporary) / "run"
                self.assertEqual(entry.output_directory(target), target)
                with self.assertRaisesRegex(ValueError, "overwrite"):
                    entry.output_directory(target)
                with self.assertRaisesRegex(ValueError, "overlaps"):
                    entry.output_directory(Path(temporary) / "raw/new", Path(temporary) / "raw")
            finally:
                entry.RESEARCH_ROOT = old

    def test_orphan_source_hash_checked_without_label_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary) / "orphan.png"
            data.write_bytes(b"raw")
            orphan = dict(label_path=None, label_sha256=None, data_path=str(data), data_sha256=entry.sha(data))
            self.assertTrue(entry.sources_unchanged([orphan]))
            data.write_bytes(b"modified")
            self.assertFalse(entry.sources_unchanged([orphan]))

    def test_duplicate_basename_cannot_overwrite_frozen_rule_entries(self):
        pairs = [(Path("a/same.json"), Path("a/first.png")), (Path("b/same.json"), Path("b/second.png"))]
        with self.assertRaisesRegex(ValueError, "duplicate label basename"):
            entry.validate_pair_names(pairs)
        pairs = [(Path("a/first.json"), Path("a/same.png")), (Path("b/second.json"), Path("b/same.png"))]
        with self.assertRaisesRegex(ValueError, "duplicate data basename"):
            entry.validate_pair_names(pairs)
        entry.validate_pair_names([(Path("a/first.json"), Path("a/first.png")), (Path("a/second.json"), Path("a/second.png"))])


if __name__ == "__main__":
    unittest.main()
