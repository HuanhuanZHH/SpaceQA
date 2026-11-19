import json
from pathlib import Path
import tempfile
import unittest

from spaceqa_agent import engine


class FormalBackendFusionTests(unittest.TestCase):
    def _dataset(self, root: Path) -> engine.Dataset:
        data = root / "a.npy"
        data.write_bytes(b"array-placeholder")
        (root / "a.npy.zattrs").write_text(json.dumps({"EXPTIME": 0.01}), encoding="utf-8")
        label_path = root / "a.json"
        label = {"basic_annotation": {"fileName": "a.npy"}}
        label_path.write_text(json.dumps(label), encoding="utf-8")
        return engine.Dataset(
            root=root,
            dataset_id="AIA_AR_sample_abcdefghij",
            data_files={data.name: data},
            label_files={label_path.name: label_path},
            labels={label_path.name: label},
        )

    def test_enhanced_signal_enters_formal_findings(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = self._dataset(Path(directory))
            engine.apply_backend_checks([dataset], "RUN-TEST", set(engine.ALL_CHECKS), "enhanced")
            self.assertEqual(dataset.backend_formal_signal_count, 1)
            self.assertTrue(any(f["checkId"] == "C06" and f["status"] == engine.STATUS_FAIL for f in dataset.findings))
            self.assertEqual(dataset.backend_profile, "enhanced")

    def test_legacy_profile_does_not_merge_signal(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = self._dataset(Path(directory))
            engine.apply_backend_checks([dataset], "RUN-TEST", set(engine.ALL_CHECKS), "legacy")
            self.assertEqual(dataset.backend_formal_signal_count, 0)
            self.assertEqual(dataset.findings, [])


if __name__ == "__main__":
    unittest.main()
