import json
from pathlib import Path
import tempfile
import unittest

from spaceqa_agent import backend_strategy, priors


class BackendStrategyTests(unittest.TestCase):
    def test_all_families_have_explicit_adapters(self):
        self.assertEqual(set(backend_strategy.STRATEGIES), set(priors.FAMILIES))
        for family in priors.FAMILIES:
            spec = backend_strategy.strategy_for(family)
            self.assertTrue(spec.reusable_steps)
            self.assertTrue(spec.evidence_units)

    def test_aia_image_without_scientific_sidecar_is_bounded_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "a.png"
            path.write_bytes(b"display-only")
            observation = backend_strategy.build_observation("AIA_AR_sample", path, {})
            self.assertEqual(observation["strategy"], "aia94_bright_region")
            self.assertEqual(observation["fallback"], "display_image_without_scientific_array_or_metadata")
            self.assertEqual(observation["scoreEffect"], "none")

    def test_aia_sidecar_exposure_is_observed_without_modifying_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "a.npy"
            image.write_bytes(b"array-placeholder")
            sidecar = root / "a.npy.zattrs"
            sidecar.write_text(json.dumps({"EXPTIME": 0.01, "CDELT1": 0.6}), encoding="utf-8")
            before = image.read_bytes()
            observation = backend_strategy.build_observation("AIA_AR_sample", image, {})
            self.assertIsNone(observation["fallback"])
            self.assertTrue(observation["metadataContext"]["metadataIndependent"])
            self.assertTrue(observation["metadataContext"]["exposureAnomalyAtPoint02s"])
            self.assertEqual(image.read_bytes(), before)

    def test_all_family_observations_are_non_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "a.png"
            path.write_bytes(b"display-only")
            for family in priors.FAMILIES:
                stats = {"full_scan": True, "regular": True, "time_kind": "relative_seconds", "median_dt": 1.0, "rows": 4, "invalid_time_rows": 0}
                observation = backend_strategy.build_observation(family, path, {}, csv_stats=stats)
                self.assertEqual(observation["scoreEffect"], "none")

    def test_legacy_profile_is_explicit_rollback(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "a.png"
            path.write_bytes(b"display-only")
            observation = backend_strategy.build_observation("HMI_AR_sample", path, {}, profile="legacy")
            self.assertEqual(observation["profile"], "legacy")
            self.assertEqual(observation["fallback"], "legacy_profile_requested")

    def test_summary_reports_actual_profile(self):
        legacy = backend_strategy.build_observation("AIA_AR_sample", None, {}, profile="legacy")
        enhanced = backend_strategy.build_observation("AIA_AR_sample", None, {}, profile="enhanced")
        self.assertEqual(backend_strategy.summarize([legacy])["profile"], "legacy")
        self.assertEqual(backend_strategy.summarize([enhanced])["profile"], "enhanced")
        self.assertEqual(backend_strategy.summarize([legacy, enhanced])["profile"], "mixed")

    def test_formal_signals_require_independent_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "a.npy"
            image.write_bytes(b"array-placeholder")
            sidecar = root / "a.npy.zattrs"
            sidecar.write_text(json.dumps({"EXPTIME": 0.01}), encoding="utf-8")
            observation = backend_strategy.build_observation("AIA_AR_sample", image, {}, profile="enhanced")
            signals = backend_strategy.formal_signals("AIA_AR_sample", observation,
                                                       label_path="label.json", data_path=str(image))
            self.assertEqual(len(signals), 1)
            self.assertEqual(signals[0]["checkId"], "C06")
            self.assertEqual(signals[0]["status"], "不通过")
            display_only = backend_strategy.build_observation("AIA_AR_sample", root / "display.png", {}, profile="enhanced")
            self.assertEqual(backend_strategy.formal_signals("AIA_AR_sample", display_only,
                                                              label_path="label.json", data_path="display.png"), [])

    def test_priors_preserves_findings_with_enhanced_observation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "a.png"
            path.write_bytes(b"not-a-png")
            label = {"basic_annotation": {"fileName": "a.png"}}
            legacy = priors.evaluate_sample("AIA_AR_sample", label, path, policy=priors.Policy(backend_profile="legacy"))
            enhanced = priors.evaluate_sample("AIA_AR_sample", label, path, policy=priors.Policy(backend_profile="enhanced"))
            self.assertEqual(legacy["findings"], enhanced["findings"])
            self.assertEqual(legacy["confirmed_bits"], enhanced["confirmed_bits"])
            self.assertEqual(enhanced["observations"]["backend_strategy"]["scoreEffect"], "none")


if __name__ == "__main__":
    unittest.main()
