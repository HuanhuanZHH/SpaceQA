import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmark"))
sys.path.insert(0, str(ROOT / "tests"))

from benchmark_multifamily import run_benchmark
from multifamily_fixture import FAMILIES, build_multifamily_fixture


class MultifamilyBenchmarkTests(unittest.TestCase):
    def test_fixture_has_six_families_and_five_tiers(self):
        with tempfile.TemporaryDirectory() as temp:
            publish, truth = build_multifamily_fixture(Path(temp))
            self.assertEqual(len(truth), 30)
            self.assertEqual({item["family"] for item in truth.values()}, set(FAMILIES))
            self.assertEqual(len([path for path in publish.iterdir() if path.is_dir()]), 30)

    def test_routes_have_expected_order_and_integrity(self):
        with tempfile.TemporaryDirectory() as temp:
            result = run_benchmark(Path(temp))
            candidates = {row["candidate"]: row for row in result["candidates"]}
            a = candidates["A_schema_pairing"]["packageClassPresence"]["recall"]
            b = candidates["B_single_package_rules"]["packageClassPresence"]["recall"]
            c = candidates["C_rules_plus_sibling_diff"]["packageClassPresence"]["recall"]
            self.assertLess(a, b)
            self.assertLess(b, c)
            # The synthetic generator adds annotations to images it then
            # deletes. Their L3-07 changes remain true injected differences,
            # but cannot be scientifically confirmed without the image.
            # Keep those false negatives in the benchmark rather than
            # redefining the truth or requiring unsafe 100% confirmation.
            class_result = candidates["C_rules_plus_sibling_diff"]["packageClassPresence"]
            self.assertEqual(class_result["truePositive"], 104)
            self.assertEqual(class_result["falseNegative"], 14)
            self.assertEqual(class_result["falsePositive"], 0)
            self.assertAlmostEqual(c, 104 / 118)
            self.assertEqual(
                {missed for package in class_result["packages"] for missed in package["missed"]},
                {"L3-07"},
            )
            self.assertGreater(candidates["C_rules_plus_sibling_diff"]["pendingCount"], 0)
            self.assertEqual(candidates["C_rules_plus_sibling_diff"]["audit"]["status"], "pass")
            self.assertEqual(candidates["C_rules_plus_sibling_diff"]["cleanFailures"]["count"], 0)
            self.assertEqual(candidates["C_rules_plus_sibling_diff"]["tierPairwise"]["correctPairs"], 60)
            self.assertEqual(candidates["C_rules_plus_sibling_diff"]["tierPairwise"]["ties"], 0)
            self.assertNotIn(
                "SAMPLE_DUPLICATE",
                candidates["C_rules_plus_sibling_diff"]["cleanFailures"]["byType"],
            )
            self.assertTrue(result["determinism"]["repeatFindingsIdentical"])
            self.assertEqual(candidates["D_weak_clean_reference_anomaly"]["cleanVsPollutedAuc"], 1.0)


if __name__ == "__main__":
    unittest.main()
